from pathlib import Path
import logging
from collections import OrderedDict
import os.path as osp
import random
import numpy as np
import torch
from torch.utils.data import get_worker_info
from torch_geometric.data import Dataset, Data
from torch_cluster import random_walk
# optional single-file storage backend
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

logger = logging.getLogger(__name__)


def _is_all_token(value):
    return isinstance(value, str) and value.strip().lower() in {'all', 'full', 'auto_all'}


def _parse_positive_int_or_all(value, name):
    if _is_all_token(value):
        return 'all'
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f'{name} must be a positive integer or "all".')
    return parsed


class RWDataset(Dataset):
    def __init__(
        self,
        root,
        batch_size,
        n_walks,
        walk_length,
        device,
        p=1,
        q=1,
        transform=None,
        pre_transform=None,
        save_batch_size=None,
        save_to_h5=False,
        h5_filename=None,
        h5_chunk_bytes=16 * 1024 * 1024,
        graph_cache_size=8,
        h5_use_gzip=True,
        show_progress=True,
        sampling_strategy='auto',
        on_the_fly=False,
        on_the_fly_num_shards=None,
        on_the_fly_seed=2025,
        on_the_fly_graphs_per_shard=1,
        on_the_fly_device='cpu',
    ):
        # core config
        self.batch_size = batch_size
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.device = device
        self.p = p
        self.q = q
        # allow saving smaller shards even if compute batch is large
        self.save_batch_size = save_batch_size or batch_size
        # source graph files
        self._files = sorted(Path(root).glob('*.pt'))
        logger.info(f'Number of graphs: {len(self._files)}')

        # on-the-fly mode skips materialized RW shards and emits the same
        # Data(x=<unique features>, map_index=<walk remap>) contract from get().
        self.on_the_fly = bool(on_the_fly)
        self.on_the_fly_num_shards = (
            None if on_the_fly_num_shards is None else int(on_the_fly_num_shards)
        )
        self.on_the_fly_seed = int(on_the_fly_seed)
        self.on_the_fly_graphs_per_shard = _parse_positive_int_or_all(
            on_the_fly_graphs_per_shard,
            'on_the_fly_graphs_per_shard',
        )
        self.on_the_fly_device = on_the_fly_device
        graph_order_generator = torch.Generator(device='cpu')
        graph_order_generator.manual_seed(self.on_the_fly_seed)
        self._on_the_fly_graph_order = torch.randperm(
            len(self._files),
            generator=graph_order_generator,
        ).tolist()
        if self.on_the_fly:
            if self.on_the_fly_graphs_per_shard == 'all':
                self._source_cache_size = len(self._files)
            else:
                self._source_cache_size = min(int(self.on_the_fly_graphs_per_shard), len(self._files))
        else:
            self._source_cache_size = 0
        self._graph_cache = None
        self._num_nodes_per_graph = None
        self._union_edge_cache = OrderedDict()

        # storage/backend options
        self.save_to_h5 = save_to_h5
        self.sampling_strategy = self._resolve_sampling_strategy(root, sampling_strategy)
        self.h5_filename = h5_filename or f'{self._prefix}.h5'
        self.h5_chunk_bytes = h5_chunk_bytes
        self.graph_cache_size = int(graph_cache_size)
        self.h5_use_gzip = bool(h5_use_gzip)
        self.show_progress = bool(show_progress)

        # persistent per-process HDF5 handle (each worker process gets its own)
        self._h5 = None  # type: ignore
        self._h5_finalizer = None  # type: ignore

        super().__init__(root, transform, pre_transform)

    # Ensure dataset is picklable for DataLoader workers: drop any open h5 handle on pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        # close and remove any existing handle before pickling
        h5 = state.get('_h5', None)
        try:
            if h5 is not None:
                try:
                    h5.close()
                except Exception:
                    pass
        finally:
            state['_h5'] = None
            state['_h5_finalizer'] = None
            state['_graph_cache'] = None
            state['_union_edge_cache'] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self) -> None:
        """Clean up HDF5 file handle on object deletion."""
        if getattr(self, '_h5', None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def _prefix_for_strategy(self, sampling_strategy):
        if sampling_strategy == 'full_coverage':
            return f'RW_{self.walk_length}_{self.p}_{self.q}_{self.n_walks}_{self.save_batch_size}'
        if sampling_strategy == 'slice_balanced':
            return f'RW_slice_balanced_{self.walk_length}_{self.p}_{self.q}_{self.n_walks}_{self.save_batch_size}'
        raise ValueError(f'Unsupported sampling_strategy: {sampling_strategy}')

    def _has_processed_cache(self, root, sampling_strategy):
        processed_dir = Path(root) / 'processed'
        prefix = self._prefix_for_strategy(sampling_strategy)
        meta_path = processed_dir / f'{prefix}_meta.pt'
        h5_path = processed_dir / f'{prefix}.h5'
        data_pattern = f'{prefix}_data_*.pt'
        if self.save_to_h5:
            return h5_path.exists()
        return meta_path.exists() or any(processed_dir.glob(data_pattern))

    def _resolve_sampling_strategy(self, root, sampling_strategy):
        if sampling_strategy not in {'auto', 'slice_balanced', 'full_coverage'}:
            raise ValueError(
                "sampling_strategy must be one of {'auto', 'slice_balanced', 'full_coverage'}."
            )

        if self.on_the_fly and sampling_strategy == 'auto':
            return 'slice_balanced'

        if sampling_strategy != 'auto':
            return sampling_strategy

        for candidate in ('slice_balanced', 'full_coverage'):
            if self._has_processed_cache(root, candidate):
                logger.info('Using existing %s cache for RWDataset.', candidate)
                return candidate

        return 'slice_balanced'
        
    @property
    def _prefix(self):
        return self._prefix_for_strategy(self.sampling_strategy)

    @property
    def raw_file_names(self):
        # No formal raw files managed by PyG; we load existing *.pt graphs from `root` directly.
        # Returning an empty list tells PyG there's nothing to download/copy into raw_dir.
        return []

    @property
    def processed_file_names(self):
        # Use a single meta marker file so PyG knows when processing is complete
        if self.on_the_fly:
            return [f'{self._prefix}_{self._on_the_fly_config_token()}_meta.pt']
        return [f'{self._prefix}_meta.pt']
    
    @property
    def processed_file_paths(self):
        # Helper for debugging: full paths to all data shard files
        if self.on_the_fly:
            return []
        if self.save_to_h5:
            return [str(Path(self.processed_dir) / self.h5_filename)]
        pattern = f'{self._prefix}_data_*.pt'
        return [str(p) for p in sorted(Path(self.processed_dir).glob(pattern))]

    def _open_h5(self, mode='a'):
        path = Path(self.processed_dir) / self.h5_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return h5py.File(path, mode)

    def _get_h5(self):
        """Return a persistent HDF5 file handle for this process (worker).
        Open lazily on first use and reuse for subsequent get() calls.
        """
        if not self.save_to_h5:
            return None
        if h5py is None:
            raise ImportError("h5py is required when save_to_h5=True. Please install h5py or set save_to_h5=False.")
        if getattr(self, "_h5", None) is None:
            path = Path(self.processed_dir) / self.h5_filename
            # Read-only handle; avoid open/close on every sample
            # Use a larger raw data chunk cache to speed up random reads
            # rdcc_nbytes ~ 256MB, rdcc_nslots ~ 1e6, rdcc_w0 ~ 0.75
            self._h5 = h5py.File(path, 'r', rdcc_nbytes=256*1024*1024, rdcc_nslots=1000003, rdcc_w0=0.75)
            try:
                import weakref
                self._h5_finalizer = weakref.finalize(self, self._h5.close)
            except Exception:
                # Best-effort; if finalizer can't be set, rely on process teardown
                self._h5_finalizer = None
        return self._h5

    def _ensure_h5_datasets(self, h5f):
        # Creates four extensible index datasets if not exist:
        # - nodes: int64 of concatenated unique_node arrays per shard
        # - node_offsets: int64 starting index per shard in nodes array; size = num_shards+1
        # - maps: int64 of concatenated map_index arrays per shard
        # - map_offsets: int64 starting index per shard in maps array; size = num_shards+1
        # Use int32 for nodes/maps to save space; offsets remain int64. Gzip is optional.
        if 'nodes' not in h5f:
            ds_kwargs = dict(chunks=True)
            if self.h5_use_gzip:
                ds_kwargs.update(dict(compression='gzip', compression_opts=4, shuffle=True))
            h5f.create_dataset('nodes', shape=(0,), maxshape=(None,), dtype='int32', **ds_kwargs)
        if 'node_offsets' not in h5f:
            h5f.create_dataset('node_offsets', shape=(1,), maxshape=(None,), dtype='int64', chunks=True)
            h5f['node_offsets'][0] = 0
        if 'maps' not in h5f:
            ds_kwargs = dict(chunks=True)
            if self.h5_use_gzip:
                ds_kwargs.update(dict(compression='gzip', compression_opts=4, shuffle=True))
            h5f.create_dataset('maps', shape=(0,), maxshape=(None,), dtype='int32', **ds_kwargs)
        if 'map_offsets' not in h5f:
            h5f.create_dataset('map_offsets', shape=(1,), maxshape=(None,), dtype='int64', chunks=True)
            h5f['map_offsets'][0] = 0
        # 'x_global' (the concatenated feature table) is created once separately

    def _h5_init_x_global(self, h5f, total_nodes: int, feat_dim: int, np_dtype):
        # Create once; prefer fast decompression (or none) for read-time speed
        if 'x_global' not in h5f:
            h5f.create_dataset(
                'x_global', shape=(total_nodes, feat_dim), maxshape=(total_nodes, feat_dim),
                dtype=np_dtype, chunks=True, compression='lzf'
            )

    def _h5_append_shard(self, h5f, unique_node, unique_index, x_chunk=None):
        nodes_ds = h5f['nodes']
        node_offs = h5f['node_offsets']
        maps_ds = h5f['maps']
        map_offs = h5f['map_offsets']
        # append nodes
        cur_n = nodes_ds.shape[0]
        add_n = unique_node.numel()
        nodes_ds.resize((cur_n + add_n,))
        nodes_np = unique_node.detach().cpu().numpy()
        if nodes_np.dtype != np.int32:
            nodes_np = nodes_np.astype(np.int32, copy=False)
        nodes_ds[cur_n:cur_n + add_n] = nodes_np
        # update node offsets
        last_node_off = int(node_offs[-1])
        cur_shards = node_offs.shape[0] - 1
        node_offs.resize((cur_shards + 2,))
        node_offs[-1] = last_node_off + add_n
        # append maps
        cur_m = maps_ds.shape[0]
        add_m = unique_index.numel()
        maps_ds.resize((cur_m + add_m,))
        maps_np = unique_index.detach().cpu().numpy()
        if maps_np.dtype != np.int32:
            maps_np = maps_np.astype(np.int32, copy=False)
        maps_ds[cur_m:cur_m + add_m] = maps_np
        last_map_off = int(map_offs[-1])
        map_offs.resize((map_offs.shape[0] + 1,))
        map_offs[-1] = last_map_off + add_m
        # shard index is consistent across both offsets now
        return cur_shards

    def _h5_get_shard(self, idx, h5f=None):
        # Accept an optional, already-open HDF5 file handle to avoid reopen overhead
        close_after = False
        if h5f is None:
            h5f = self._open_h5('r')
            close_after = True
        try:
            nodes = h5f['nodes']
            node_offsets = h5f['node_offsets']
            maps = h5f['maps']
            map_offsets = h5f['map_offsets']
            xg = h5f['x_global']
            ns = int(node_offsets[idx])
            ne = int(node_offsets[idx + 1])
            ms = int(map_offsets[idx])
            me = int(map_offsets[idx + 1])
            unique_node_np = nodes[ns:ne]
            unique_index_np = maps[ms:me]
            # gather features from x_global for the unique nodes
            # Optimize read locality: sort indices and read contiguous runs in chunks, then scatter back
            u64 = unique_node_np.astype(np.int64, copy=False)
            n = u64.size
            if n > 1:
                order = np.argsort(u64, kind='mergesort')
                inv = np.empty_like(order)
                inv[order] = np.arange(order.size)
                sorted_u = u64[order]
                feat_dim = xg.shape[1]
                x_np = np.empty((n, feat_dim), dtype=xg.dtype)
                i = 0
                # limit block size to keep memory bounded
                max_block = 65536
                while i < n:
                    j = i + 1
                    # extend j while indices are contiguous and within block size
                    while j < n and (j - i) < max_block and sorted_u[j] == sorted_u[j - 1] + 1:
                        j += 1
                    # read contiguous slice [sorted_u[i], sorted_u[j-1]]
                    start = int(sorted_u[i])
                    end = int(sorted_u[j - 1])
                    block = xg[start:end + 1, :]
                    # original positions for this sorted segment: inv[i:j]
                    orig_idx = inv[i:j]
                    x_np[orig_idx, :] = block
                    i = j
            else:
                x_np = xg[u64, :]
            return (
                torch.from_numpy(unique_node_np).long(),
                torch.from_numpy(unique_index_np).long(),
                torch.from_numpy(x_np),
            )
        finally:
            if close_after and h5f is not None:
                h5f.close()

    def _on_the_fly_len(self):
        if self.on_the_fly_num_shards is not None:
            return self.on_the_fly_num_shards
        if len(self._files) == 0:
            return 0
        num_nodes_per_graph = self._load_on_the_fly_num_nodes()
        total_nodes = int(sum(num_nodes_per_graph))
        return (total_nodes + self.save_batch_size - 1) // self.save_batch_size

    def _on_the_fly_config_token(self):
        graphs_per_shard = self._on_the_fly_graphs_per_shard_meta()
        num_shards = 'auto' if self.on_the_fly_num_shards is None else int(self.on_the_fly_num_shards)
        return (
            f'on_the_fly_gps_{graphs_per_shard}'
            f'_n_{num_shards}'
        )

    def _on_the_fly_meta_path(self):
        return Path(self.processed_dir) / f'{self._prefix}_{self._on_the_fly_config_token()}_meta.pt'

    def _resolved_on_the_fly_graphs_per_shard(self):
        if self.on_the_fly_graphs_per_shard == 'all':
            return len(self._files)
        return min(int(self.on_the_fly_graphs_per_shard), len(self._files))

    def _on_the_fly_graphs_per_shard_meta(self):
        return (
            'all'
            if self.on_the_fly_graphs_per_shard == 'all'
            else int(self.on_the_fly_graphs_per_shard)
        )

    def _load_on_the_fly_num_nodes(self):
        if self._num_nodes_per_graph is not None:
            return self._num_nodes_per_graph

        graph_paths = [str(p) for p in self._files]
        meta_path = self._on_the_fly_meta_path()
        if meta_path.exists():
            try:
                meta = torch.load(str(meta_path), weights_only=False)
                cached_counts = meta.get('num_nodes_per_graph')
                cached_graphs = meta.get('graphs')
                if cached_counts is not None and cached_graphs == graph_paths:
                    self._num_nodes_per_graph = [int(n) for n in cached_counts]
                    return self._num_nodes_per_graph
            except Exception:
                logger.warning("Could not read on-the-fly metadata cache at %s", meta_path, exc_info=True)

        counts = []
        for path in graph_paths:
            try:
                data = torch.load(path, map_location='meta', weights_only=False)
                n = int(getattr(data, 'num_nodes', data.x.size(0)))
            except Exception:
                logger.warning(
                    "Falling back to CPU load to inspect graph metadata for %s",
                    path,
                    exc_info=True,
                )
                data = torch.load(path, map_location='cpu', weights_only=False)
                n = int(getattr(data, 'num_nodes', data.x.size(0)))
            counts.append(n)

        self._num_nodes_per_graph = counts
        return self._num_nodes_per_graph

    def _write_on_the_fly_meta(self):
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        num_nodes_per_graph = None
        total_nodes = None
        if self.on_the_fly_num_shards is None:
            num_nodes_per_graph = self._load_on_the_fly_num_nodes()
            total_nodes = int(sum(num_nodes_per_graph))
        meta = {
            'num_shards': int(self._on_the_fly_len()),
            'walk_length': int(self.walk_length),
            'p': float(self.p),
            'q': float(self.q),
            'compute_batch_size': int(self.batch_size),
            'save_batch_size': int(self.save_batch_size),
            'n_walks': int(self.n_walks),
            'save_to_h5': False,
            'on_the_fly': True,
            'on_the_fly_seed': int(self.on_the_fly_seed),
            'on_the_fly_graphs_per_shard': self._on_the_fly_graphs_per_shard_meta(),
            'on_the_fly_graphs_per_shard_resolved': int(self._resolved_on_the_fly_graphs_per_shard()),
            'on_the_fly_device': str(self.on_the_fly_device),
            'sampling_strategy': str(self.sampling_strategy),
            'graphs': [str(p) for p in self._files],
        }
        if num_nodes_per_graph is not None:
            meta['num_nodes_per_graph'] = [int(n) for n in num_nodes_per_graph]
            meta['total_nodes'] = int(total_nodes)
            meta['num_shards_source'] = 'node_budget'
        else:
            meta['num_shards_source'] = 'explicit'
        torch.save(meta, self._on_the_fly_meta_path())

    def _get_source_graph(self, gid: int):
        if self._graph_cache is None:
            self._graph_cache = OrderedDict()

        if gid in self._graph_cache:
            x_cpu, edge_index_cpu, num_nodes = self._graph_cache.pop(gid)
            self._graph_cache[gid] = (x_cpu, edge_index_cpu, num_nodes)
            return x_cpu, edge_index_cpu, num_nodes

        graph_path = str(self._files[gid])
        data = torch.load(graph_path, map_location='cpu', weights_only=False)
        x_cpu = data.x.detach().cpu()
        edge_index_cpu = data.edge_index.detach().cpu().long()
        num_nodes = int(getattr(data, 'num_nodes', x_cpu.size(0)))

        if self._source_cache_size > 0:
            while len(self._graph_cache) >= self._source_cache_size:
                self._graph_cache.popitem(last=False)
            self._graph_cache[gid] = (x_cpu, edge_index_cpu, num_nodes)

        return x_cpu, edge_index_cpu, num_nodes

    def warm_on_the_fly_graph_cache(self, rw_device=None, shard_idx=0):
        if not self.on_the_fly or self._source_cache_size <= 0:
            return 0
        gids = self._select_on_the_fly_graphs(int(shard_idx))
        rw_device = torch.device(rw_device) if rw_device is not None else None
        for gid in gids:
            self._get_source_graph(gid)
        if self._can_use_union_random_walk():
            rw_device = rw_device if rw_device is not None else self._on_the_fly_rw_device()
            self._get_union_edge_cache(rw_device, gids)
        return len(gids)

    def _can_use_union_random_walk(self):
        return (
            self.on_the_fly
            and self._resolved_on_the_fly_graphs_per_shard() > 1
            and len(self._files) > 0
        )

    def _get_union_edge_cache(self, device, gids):
        device = torch.device(device)
        gids = tuple(int(gid) for gid in gids)
        cache_key = (str(device), gids)
        if self._union_edge_cache is None:
            self._union_edge_cache = OrderedDict()
        if cache_key in self._union_edge_cache:
            cache_value = self._union_edge_cache.pop(cache_key)
            self._union_edge_cache[cache_key] = cache_value
            return cache_value

        offsets = [0]
        edge_parts = []
        empty_edge_gids = []
        for gid in gids:
            _, edge_index_cpu, num_nodes = self._get_source_graph(gid)
            graph_offset = offsets[-1]
            offsets.append(graph_offset + int(num_nodes))
            if edge_index_cpu.numel() == 0:
                empty_edge_gids.append(gid)
                continue
            edge_parts.append(edge_index_cpu.long() + graph_offset)

        if len(edge_parts) > 0:
            edge_index = torch.cat(edge_parts, dim=1).to(device, non_blocking=False)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        offsets_cpu = torch.tensor(offsets, dtype=torch.long)
        gids_cpu = torch.tensor(gids, dtype=torch.long)
        cache_value = {
            'edge_index': edge_index,
            'offsets_cpu': offsets_cpu,
            'offsets_device': offsets_cpu.to(device, non_blocking=False),
            'gids_cpu': gids_cpu,
            'empty_edge_gids': set(empty_edge_gids),
        }
        while len(self._union_edge_cache) >= 1:
            self._union_edge_cache.popitem(last=False)
        self._union_edge_cache[cache_key] = cache_value
        return cache_value

    def _on_the_fly_rw_device(self):
        requested = self.on_the_fly_device or 'cpu'
        requested = str(requested)
        if requested.startswith('cuda') and get_worker_info() is not None:
            # DataLoader workers should not initialize CUDA contexts. Keep RW
            # sampling on CPU unless the caller runs with num_workers=0.
            return torch.device('cpu')
        return torch.device(requested)

    def _select_on_the_fly_graphs(self, idx: int):
        n_graphs = len(self._files)
        if n_graphs == 0:
            return []

        graphs_per_shard = self._resolved_on_the_fly_graphs_per_shard()
        if graphs_per_shard == n_graphs:
            return list(self._on_the_fly_graph_order)

        num_windows = (n_graphs + graphs_per_shard - 1) // graphs_per_shard
        start = (int(idx) % num_windows) * graphs_per_shard
        return [
            self._on_the_fly_graph_order[(start + i) % n_graphs]
            for i in range(graphs_per_shard)
        ]

    def _split_seed_counts(self, gids, idx: int):
        if len(gids) == 0:
            return {}
        base = self.save_batch_size // len(gids)
        rem = self.save_batch_size - base * len(gids)
        counts = {gid: base for gid in gids}
        for offset in range(rem):
            counts[gids[(idx + offset) % len(gids)]] += 1
        return counts

    def _draw_on_the_fly_nodes(self, num_nodes: int, k: int, seed: int):
        if k <= 0 or num_nodes <= 0:
            return torch.empty(0, dtype=torch.long)

        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)

        if k >= num_nodes:
            parts = []
            remaining = k
            while remaining > 0:
                perm = torch.randperm(num_nodes, generator=generator)
                take = min(remaining, num_nodes)
                parts.append(perm[:take])
                remaining -= take
            return torch.cat(parts, dim=0)

        # k is normally small (save_batch_size split over source graphs). Avoid
        # randperm(num_nodes) for million-node graphs, while still keeping the
        # selected seeds unique within this shard/graph.
        selected = []
        seen = set()
        draw_size = max(k * 2, 32)
        while len(selected) < k:
            candidates = torch.randint(num_nodes, (draw_size,), generator=generator)
            for value in candidates.tolist():
                if value in seen:
                    continue
                seen.add(value)
                selected.append(value)
                if len(selected) == k:
                    break
            if len(seen) > num_nodes:
                break
        return torch.tensor(selected, dtype=torch.long)

    def _on_the_fly_shard_seed(self, idx: int, salt: int = 0):
        return self.on_the_fly_seed + idx * 1000003 + salt

    def _draw_on_the_fly_seed_map(self, gids, counts, idx: int):
        seed_parts = []
        gid_parts = []
        for salt, gid in enumerate(gids):
            k = counts.get(gid, 0)
            if k <= 0:
                continue
            _, _, num_nodes = self._get_source_graph(gid)
            idxs = self._draw_on_the_fly_nodes(
                num_nodes,
                k,
                self._on_the_fly_shard_seed(int(idx), salt=gid + salt),
            )
            if idxs.numel() == 0:
                continue
            seed_parts.append(idxs)
            gid_parts.append(torch.full((idxs.numel(),), gid, dtype=torch.long))

        if len(seed_parts) == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return torch.cat(seed_parts, dim=0), torch.cat(gid_parts, dim=0)

    def _x_chunk_from_unique_union(self, unique_union_cpu, offsets_cpu, gids_cpu):
        part_u = torch.bucketize(unique_union_cpu, offsets_cpu[1:], right=True)
        first_gid = int(gids_cpu[int(part_u[0])])
        x0_cpu, _, _ = self._get_source_graph(first_gid)
        x_chunk = torch.empty(
            (unique_union_cpu.numel(), x0_cpu.size(1)),
            dtype=x0_cpu.dtype,
        )

        for part_tensor_value in part_u.unique(sorted=True):
            part = int(part_tensor_value.item())
            gid = int(gids_cpu[part])
            mask = part_u == part
            local_nodes = (unique_union_cpu[mask] - offsets_cpu[part]).long()
            x_cpu, _, _ = self._get_source_graph(gid)
            x_chunk[mask] = x_cpu[local_nodes]

        return x_chunk

    def _generate_on_the_fly_union_shard(self, idx: int):
        gids = self._select_on_the_fly_graphs(int(idx))
        counts = self._split_seed_counts(gids, int(idx))
        rw_device = self._on_the_fly_rw_device()
        union_cache = self._get_union_edge_cache(rw_device, gids)
        edge_index = union_cache['edge_index']
        offsets_cpu = union_cache['offsets_cpu']
        gids_cpu = union_cache['gids_cpu']
        empty_edge_gids = union_cache['empty_edge_gids']

        seed_nodes_cpu, seed_gids_cpu = self._draw_on_the_fly_seed_map(gids, counts, idx)
        if seed_nodes_cpu.numel() == 0:
            raise RuntimeError(f'Could not generate on-the-fly shard {idx}.')

        gid_to_part = {gid: part for part, gid in enumerate(gids)}
        seed_parts_cpu = torch.tensor(
            [gid_to_part[int(gid)] for gid in seed_gids_cpu.tolist()],
            dtype=torch.long,
        )
        union_seed_cpu = seed_nodes_cpu + offsets_cpu[seed_parts_cpu]
        has_edge_mask_cpu = torch.tensor(
            [int(gid.item()) not in empty_edge_gids for gid in seed_gids_cpu],
            dtype=torch.bool,
        )

        cur_patterns = torch.empty(
            (self.n_walks, seed_nodes_cpu.numel(), self.walk_length + 1),
            dtype=torch.long,
        )
        if torch.any(has_edge_mask_cpu):
            has_edge_pos = has_edge_mask_cpu.nonzero(as_tuple=False).view(-1)
            seeds = union_seed_cpu[has_edge_mask_cpu].to(rw_device).long().repeat(self.n_walks)
            if edge_index.numel() > 0:
                rw_seed = self._on_the_fly_shard_seed(int(idx), salt=1009)
                fork_devices = []
                if rw_device.type == 'cuda':
                    fork_devices = [
                        rw_device.index
                        if rw_device.index is not None
                        else torch.cuda.current_device()
                    ]
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(rw_seed)
                    if rw_device.type == 'cuda':
                        torch.cuda.manual_seed_all(rw_seed)
                    ptn = random_walk(
                        edge_index[0].long(),
                        edge_index[1].long(),
                        start=seeds,
                        walk_length=self.walk_length,
                        p=self.p,
                        q=self.q,
                    ).view(self.n_walks, -1, self.walk_length + 1)
            else:
                ptn = seeds.view(self.n_walks, -1, 1).repeat(1, 1, self.walk_length + 1)
            cur_patterns[:, has_edge_pos, :] = ptn.detach().cpu()

        if torch.any(~has_edge_mask_cpu):
            no_edge_pos = (~has_edge_mask_cpu).nonzero(as_tuple=False).view(-1)
            seeds = union_seed_cpu[~has_edge_mask_cpu].long().repeat(self.n_walks)
            ptn = seeds.view(self.n_walks, -1, 1).repeat(1, 1, self.walk_length + 1)
            cur_patterns[:, no_edge_pos, :] = ptn

        unique_union, inverse = torch.unique(
            cur_patterns.reshape(-1).long(),
            sorted=True,
            return_inverse=True,
        )
        x_chunk = self._x_chunk_from_unique_union(unique_union.cpu(), offsets_cpu, gids_cpu)
        return Data(x=x_chunk, map_index=inverse.long())

    def _generate_on_the_fly_shard(self, idx: int):
        if len(self._files) == 0:
            raise IndexError('RWDataset has no source graph .pt files.')

        if self._can_use_union_random_walk():
            return self._generate_on_the_fly_union_shard(idx)

        gids = self._select_on_the_fly_graphs(int(idx))
        counts = self._split_seed_counts(gids, int(idx))
        rw_device = self._on_the_fly_rw_device()

        patterns_parts = []
        gid_parts = []
        for salt, gid in enumerate(gids):
            k = counts.get(gid, 0)
            if k <= 0:
                continue

            _, edge_index_cpu, num_nodes = self._get_source_graph(gid)
            idxs = self._draw_on_the_fly_nodes(
                num_nodes,
                k,
                self._on_the_fly_shard_seed(int(idx), salt=gid + salt),
            )
            if idxs.numel() == 0:
                continue

            seeds = idxs.to(rw_device).long().repeat(self.n_walks)
            edge_index = edge_index_cpu.to(rw_device, non_blocking=False)
            if edge_index.numel() > 0:
                rw_seed = self._on_the_fly_shard_seed(int(idx), salt=gid + 1009)
                fork_devices = []
                if rw_device.type == 'cuda':
                    fork_devices = [
                        rw_device.index
                        if rw_device.index is not None
                        else torch.cuda.current_device()
                    ]
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(rw_seed)
                    if rw_device.type == 'cuda':
                        torch.cuda.manual_seed_all(rw_seed)
                    ptn = random_walk(
                        edge_index[0].long(),
                        edge_index[1].long(),
                        start=seeds,
                        walk_length=self.walk_length,
                        p=self.p,
                        q=self.q,
                    ).view(self.n_walks, -1, self.walk_length + 1)
            else:
                ptn = seeds.view(self.n_walks, -1, 1).repeat(1, 1, self.walk_length + 1)

            patterns_parts.append(ptn.detach().cpu())
            gid_parts.append(
                torch.full(
                    ptn.shape,
                    fill_value=gid,
                    dtype=torch.long,
                    device='cpu',
                )
            )

        if len(patterns_parts) == 0:
            raise RuntimeError(f'Could not generate on-the-fly shard {idx}.')

        cur_patterns = torch.cat(patterns_parts, dim=1)
        gid_tensor = torch.cat(gid_parts, dim=1)

        flat_gid = gid_tensor.reshape(-1).long()
        flat_node = cur_patterns.reshape(-1).long()
        unique_pairs, inverse = torch.unique(
            torch.stack((flat_gid, flat_node), dim=1),
            dim=0,
            sorted=True,
            return_inverse=True,
        )

        first_gid = int(unique_pairs[0, 0])
        x0_cpu, _, _ = self._get_source_graph(first_gid)
        x_chunk = torch.empty(
            (unique_pairs.size(0), x0_cpu.size(1)),
            dtype=x0_cpu.dtype,
        )

        for gid_tensor_value in unique_pairs[:, 0].unique(sorted=True):
            gid = int(gid_tensor_value.item())
            mask = unique_pairs[:, 0] == gid
            local_nodes = unique_pairs[mask, 1].long()
            x_cpu, _, _ = self._get_source_graph(gid)
            x_chunk[mask] = x_cpu[local_nodes]

        return Data(x=x_chunk, map_index=inverse.long())

    def process(self):
        if self.on_the_fly:
            self._write_on_the_fly_meta()
            return

        total_shards = 0
        h5f = None
        if self.save_to_h5:
            if h5py is None:
                raise ImportError("h5py is required when save_to_h5=True. Please install h5py or set save_to_h5=False.")
            h5f = self._open_h5('w')  # fresh file for this processing run
            self._ensure_h5_datasets(h5f)
        # Ensure processed_dir exists for saving .pt shards
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        graph_files = [str(p) for p in self._files]
        num_nodes_per_graph = []
        perms = []
        pos = []
        for gf in graph_files:
            g = torch.load(gf, weights_only=False)
            n = int(g.num_nodes)
            num_nodes_per_graph.append(n)
            perms.append(torch.randperm(n))
            pos.append(0)
        # 2) Precompute global ID offsets per graph
        offsets = [0]
        for n in num_nodes_per_graph:
            offsets.append(offsets[-1] + n)
        offsets_cpu = torch.tensor(offsets, dtype=torch.long)
        offsets_dev = offsets_cpu.to(self.device)

    # LRU cache for graphs: store x on CPU; row/col on device for RW
        from collections import OrderedDict
        graph_cache: OrderedDict[int, tuple] = OrderedDict()

        def get_graph(gid: int):
            if gid in graph_cache:
                # refresh LRU
                x_cpu, row, col = graph_cache.pop(gid)
                graph_cache[gid] = (x_cpu, row, col)
                return x_cpu, row, col
            # load from disk
            data = torch.load(graph_files[gid], weights_only=False)
            x_cpu = data.x.detach().cpu()
            if data.edge_index.numel() > 0:
                rowcol = data.edge_index.to(self.device)
                row, col = rowcol[0].long(), rowcol[1].long()
            else:
                row = col = None
            # evict if needed
            if len(graph_cache) >= self.graph_cache_size:
                graph_cache.popitem(last=False)
            graph_cache[gid] = (x_cpu, row, col)
            return x_cpu, row, col

        active = [gid for gid, n in enumerate(num_nodes_per_graph) if n > 0]
        graph_ids = active.copy()
        total_seed_budget = int(sum(num_nodes_per_graph))

        def draw_seed_indices(gid: int, k: int):
            n = num_nodes_per_graph[gid]
            if k <= 0 or n <= 0:
                return torch.empty(0, dtype=torch.long)

            parts = []
            remaining = k
            while remaining > 0:
                available = n - pos[gid]
                if available == 0:
                    perms[gid] = torch.randperm(n)
                    pos[gid] = 0
                    available = n

                take_now = min(remaining, available)
                parts.append(perms[gid][pos[gid]:pos[gid] + take_now])
                pos[gid] += take_now
                remaining -= take_now

            if len(parts) == 1:
                return parts[0]
            return torch.cat(parts, dim=0)

        # Progress bar setup over total nodes
        total_nodes_total = int(sum(num_nodes_per_graph))
        pbar = None
        if self.show_progress:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=total_nodes_total, desc=f"RWDataset[{self._prefix}]", unit="node")
            except Exception:
                pbar = None

        # Initialize and write x_global once (HDF5 backend only)
        if self.save_to_h5 and h5f is not None:
            # determine feature dim and dtype from first graph
            # use get_graph to leverage caching/device move
            # build x_global by concatenating per-graph x in the same order as offsets
            first_data = torch.load(graph_files[0], weights_only=False) if len(graph_files) > 0 else None
            if first_data is not None:
                feat_dim = int(first_data.x.size(1))
                np_dtype = first_data.x.detach().cpu().numpy().dtype
                self._h5_init_x_global(h5f, total_nodes_total, feat_dim, np_dtype)
                xg = h5f['x_global']
                base = 0
                for gid, gf in enumerate(graph_files):
                    data = torch.load(gf, weights_only=False)
                    x_cpu = data.x.detach().cpu().numpy()
                    n = x_cpu.shape[0]
                    xg[base:base + n, :] = x_cpu
                    base += n

        def save_shard(cur_patterns, gid_tensor, used_gids):
            nonlocal total_shards

            global_ids = cur_patterns + offsets_dev[gid_tensor]
            flat_global = global_ids.reshape(-1)
            unique_global, inverse = torch.unique(flat_global, return_inverse=True)

            unique_global_cpu = unique_global.detach().cpu()
            gid_u = torch.bucketize(unique_global_cpu, offsets_cpu[1:], right=True)
            first_gid = used_gids[0]
            x0_cpu, _, _ = get_graph(first_gid)
            feat_dim = int(x0_cpu.size(1))
            x_chunk = torch.empty((unique_global_cpu.numel(), feat_dim), dtype=x0_cpu.dtype)
            uniq_gids = gid_u.unique().tolist()
            for g in uniq_gids:
                mask = (gid_u == g)
                if not torch.any(mask):
                    continue
                xg_cpu, _, _ = get_graph(g)
                locals_g = (unique_global_cpu[mask] - offsets_cpu[g]).long()
                x_chunk[mask] = xg_cpu[locals_g]

            inverse_cpu = inverse.detach().cpu()
            if self.save_to_h5:
                self._h5_append_shard(h5f, unique_global_cpu, inverse_cpu)
            else:
                sub_graph = Data(x=x_chunk, map_index=inverse_cpu)
                torch.save(sub_graph, osp.join(self.processed_dir, f'{self._prefix}_data_{total_shards}.pt'))
            total_shards += 1

        try:
            with torch.no_grad():
                if self.sampling_strategy == 'full_coverage':
                    while len(active) > 0:
                        # Full-coverage mixer: visit every node exactly once and build each shard from multiple graphs.
                        m = self.save_batch_size
                        ng = len(active)
                        take_base = m // ng if ng > 0 else m
                        rem = m - take_base * ng
                        order = active.copy()
                        random.shuffle(order)
                        take = {gid: take_base for gid in order}
                        for gid in order[:rem]:
                            take[gid] += 1
                        for gid in list(take.keys()):
                            take[gid] = min(take[gid], num_nodes_per_graph[gid] - pos[gid])
                        if sum(take.values()) == 0:
                            for gid in order:
                                if pos[gid] < num_nodes_per_graph[gid]:
                                    take[gid] = min(1, num_nodes_per_graph[gid] - pos[gid])
                                    if sum(take.values()) >= m:
                                        break

                        patterns_parts = []
                        gid_parts = []
                        used_gids = []
                        for gid in order:
                            k = take.get(gid, 0)
                            if k <= 0:
                                continue
                            idxs = perms[gid][pos[gid]:pos[gid] + k]
                            pos[gid] += k
                            _, row_g, col_g = get_graph(gid)
                            seeds = idxs.to(self.device).long().repeat(self.n_walks)
                            if row_g is not None:
                                ptn = random_walk(
                                    row_g, col_g, start=seeds,
                                    walk_length=self.walk_length, p=self.p, q=self.q
                                ).view(self.n_walks, -1, self.walk_length + 1)
                            else:
                                ptn = seeds.view(self.n_walks, -1, 1).repeat(1, 1, self.walk_length + 1)
                            patterns_parts.append(ptn)
                            gid_parts.append(torch.full_like(ptn, fill_value=gid, dtype=torch.long))
                            used_gids.append(gid)

                        if len(patterns_parts) == 0:
                            active = [gid for gid in active if pos[gid] < num_nodes_per_graph[gid]]
                            continue

                        cur_patterns = torch.cat(patterns_parts, dim=1)
                        gid_tensor = torch.cat(gid_parts, dim=1)
                        save_shard(cur_patterns, gid_tensor, used_gids)

                        seeds_in_shard = int(sum(take.values()))
                        if pbar is not None:
                            pbar.update(seeds_in_shard)
                        active = [gid for gid in active if pos[gid] < num_nodes_per_graph[gid]]
                else:
                    # Slice-balanced mixer: keep every graph in rotation for the full run.
                    seeds_generated = 0
                    while seeds_generated < total_seed_budget and len(graph_ids) > 0:
                        # Nodes are sampled uniformly within each graph without replacement, and
                        # small graphs reshuffle when exhausted so late shards do not drift
                        # toward only large graphs.
                        m = min(self.save_batch_size, total_seed_budget - seeds_generated)
                        ng = len(graph_ids)
                        take_base = m // ng if ng > 0 else m
                        rem = m - take_base * ng
                        order = graph_ids.copy()
                        random.shuffle(order)
                        take = {gid: take_base for gid in order}
                        for gid in order[:rem]:
                            take[gid] += 1

                        patterns_parts = []
                        gid_parts = []
                        used_gids = []
                        for gid in order:
                            k = take.get(gid, 0)
                            if k <= 0:
                                continue
                            idxs = draw_seed_indices(gid, k)
                            if idxs.numel() == 0:
                                continue
                            _, row_g, col_g = get_graph(gid)
                            seeds = idxs.to(self.device).long().repeat(self.n_walks)
                            if row_g is not None:
                                ptn = random_walk(
                                    row_g, col_g, start=seeds,
                                    walk_length=self.walk_length, p=self.p, q=self.q
                                ).view(self.n_walks, -1, self.walk_length + 1)
                            else:
                                ptn = seeds.view(self.n_walks, -1, 1).repeat(1, 1, self.walk_length + 1)
                            patterns_parts.append(ptn)
                            gid_parts.append(torch.full_like(ptn, fill_value=gid, dtype=torch.long))
                            used_gids.append(gid)

                        if len(patterns_parts) == 0:
                            continue

                        cur_patterns = torch.cat(patterns_parts, dim=1)
                        gid_tensor = torch.cat(gid_parts, dim=1)
                        save_shard(cur_patterns, gid_tensor, used_gids)

                        seeds_in_shard = int(sum(take.values()))
                        if pbar is not None:
                            pbar.update(seeds_in_shard)
                        seeds_generated += seeds_in_shard
        finally:
            if h5f is not None:
                h5f.close()
            if pbar is not None:
                pbar.close()

        # Write a small meta file as the processed marker
        meta = {
            'num_shards': int(total_shards),
            'walk_length': int(self.walk_length),
            'p': float(self.p),
            'q': float(self.q),
            'compute_batch_size': int(self.batch_size),
            'save_batch_size': int(self.save_batch_size),
            'n_walks': int(self.n_walks),
            'save_to_h5': bool(self.save_to_h5),
            'h5_filename': str(self.h5_filename),
            'x_stored_globally': bool(self.save_to_h5),
            'h5_use_gzip': bool(self.h5_use_gzip),
            'sampling_strategy': str(self.sampling_strategy),
            # for reference only
            'graphs': [str(p) for p in self._files],
        }
        torch.save(meta, osp.join(self.processed_dir, f'{self._prefix}_meta.pt'))

    def len(self):
        if self.on_the_fly:
            return int(self._on_the_fly_len())
        if self.save_to_h5:
            # read num_shards from meta if available; else derive from offsets
            meta_path = Path(self.processed_dir) / f'{self._prefix}_meta.pt'
            if meta_path.exists():
                meta = torch.load(str(meta_path), weights_only=False)
                return int(meta.get('num_shards', 0))
            with self._open_h5('r') as h5f:
                return int(h5f['node_offsets'].shape[0] - 1)
        pattern = f'{self._prefix}_data_*.pt'
        return sum(1 for _ in Path(self.processed_dir).glob(pattern))

    def get(self, idx):
        if self.on_the_fly:
            return self._generate_on_the_fly_shard(idx)
        if self.save_to_h5:
            # Load unique_node, map_index and x for shard using persistent handle when available
            h5f = self._get_h5()
            unique_node, unique_index, x = self._h5_get_shard(idx, h5f=h5f)
            return Data(x=x, map_index=unique_index)
        else:
            data = torch.load(osp.join(self.processed_dir, f'{self._prefix}_data_{idx}.pt'), weights_only=False)
            return data
