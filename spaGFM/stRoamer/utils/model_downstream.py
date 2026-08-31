import logging
from spaGFM.stRoamer.utils.metric_eval import evaluate, normalize_metric_names
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, one_hot

logger = logging.getLogger(__name__)
CLASSIFICATION_METRICS = {'acc', 'acc_macro', 'f1', 'f1_macro', 'mcc'}
REGRESSION_METRICS = {'rmse', 'mae'}
CLUSTERING_METRICS = {'ari', 'nmi', 'silhouette'}


def _encode_class_labels(y, params):
    y_cpu = y.detach().cpu()
    unique_labels = torch.unique(y_cpu).tolist()
    class_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    y_encoded_cpu = y_cpu.clone().long()
    for label, idx in class_to_idx.items():
        y_encoded_cpu[y_cpu == label] = idx

    num_classes = params.get('output_dim', len(class_to_idx))
    if num_classes < len(class_to_idx):
        raise ValueError(
            f"output_dim ({num_classes}) is smaller than number of classes ({len(class_to_idx)})."
        )

    return y_encoded_cpu.to(y.device), num_classes


def _is_regression_metric(metric_name):
    return metric_name in REGRESSION_METRICS


def _regression_prediction(logits):
    return logits.squeeze(-1) if logits.ndim > 1 and logits.size(-1) == 1 else logits


def _is_better_metric(current, best, metric_name):
    return current < best if _is_regression_metric(metric_name) else current > best


def _get_metric_names(params):
    return normalize_metric_names(params['metric'])


def _get_primary_metric(params):
    return _get_metric_names(params)[0]


def _validate_metrics(metric_names, allowed_metrics, context):
    invalid_metrics = [metric for metric in metric_names if metric not in allowed_metrics]
    if invalid_metrics:
        raise ValueError(
            f"{context} does not support metric(s) {invalid_metrics}. Supported metrics: {sorted(allowed_metrics)}."
        )


def _summarize_metric_results(metric_names, val_scores_by_metric, test_scores_by_metric):
    if len(metric_names) == 1:
        metric_name = metric_names[0]
        return {
            'train': 0,
            'train_std': 0,
            'val': np.mean(val_scores_by_metric[metric_name]),
            'val_std': np.std(val_scores_by_metric[metric_name]),
            'test': np.mean(test_scores_by_metric[metric_name]),
            'test_std': np.std(test_scores_by_metric[metric_name]),
            'metric': metric_name,
        }

    return {
        'metric': metric_names[0],
        'metrics': {
            metric_name: {
                'train': 0,
                'train_std': 0,
                'val': np.mean(val_scores_by_metric[metric_name]),
                'val_std': np.std(val_scores_by_metric[metric_name]),
                'test': np.mean(test_scores_by_metric[metric_name]),
                'test_std': np.std(test_scores_by_metric[metric_name]),
                'metric': metric_name,
            }
            for metric_name in metric_names
        },
    }


def _evaluate_metric_bundle(pred, y, params, embeddings=None):
    metric_names = _get_metric_names(params)
    return {
        metric_name: evaluate(pred, y, params=params, embeddings=embeddings, metric_name=metric_name)
        for metric_name in metric_names
    }

def node_linear_prob(x, x_slot, y, y_slot, data_name, params):
    train, test = x
    train = torch.tensor(train.obsm[x_slot], dtype=torch.float)

    y_train, y_test = y
    y_train = torch.tensor(y_train.obs[y_slot].values, dtype=torch.long)

    num_classes = params['output_dim']
    device = params['device']

    classifier = nn.Linear(train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=params['linear_probe_lr'], weight_decay=params['linear_probe_weight_decay'])

    # Train for a fixed number of epochs
    num_epochs = params['linear_probe_epochs']

    for epoch in range(num_epochs):
        classifier.train()
        
        # Training in mini-batches
        train_batch_size = 256
        train_idx = torch.randperm(train.shape[0])
        train_loss = 0
        num_train_batches = (train.shape[0] + train_batch_size - 1) // train_batch_size

        for i in range(num_train_batches):
            optimizer.zero_grad()
            batch_idx = train_idx[i * train_batch_size:(i + 1) * train_batch_size]
            batch_X = train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            batch_logits = classifier(batch_X)
            if params.get('num_tasks') is not None:
                if params['num_tasks'] == 1:
                    loss = F.binary_cross_entropy_with_logits(batch_logits.squeeze(), batch_y.float())
                else:
                    loss = F.binary_cross_entropy_with_logits(batch_logits, batch_y.float())
            elif _is_regression_metric(params['metric']):
                loss = F.mse_loss(_regression_prediction(batch_logits), batch_y.float())
            else:
                loss = F.cross_entropy(batch_logits, batch_y)

            loss.backward()
            train_loss += loss.item()

            optimizer.step()

    # Evaluation in mini-batches
    classifier.eval()
    test_batch_size = 256

    res_dict = {}
    for test_d_name, test, y_test in zip(data_name, test, y_test):

        test = torch.tensor(test.obsm[x_slot], dtype=torch.float)
        y_test = torch.tensor(y_test.obs[y_slot].values, dtype=torch.long)

        with torch.no_grad():
            # Test
            test_logits_list = []
            num_test_batches = (test.shape[0] + test_batch_size - 1) // test_batch_size
            for i in range(num_test_batches):
                start_idx = i * test_batch_size
                end_idx = min((i + 1) * test_batch_size, test.shape[0])
                batch_logits = classifier(test[start_idx:end_idx].to(device))
                if _is_regression_metric(params['metric']):
                    batch_logits = _regression_prediction(batch_logits)
                test_logits_list.append(batch_logits)
            test_logits = torch.cat(test_logits_list, dim=0)

            test_acc = evaluate(test_logits, y_test.to(device), params=params)
            res_dict[test_d_name] = {'train': 0, 'train_std': 0, 'test': test_acc, 'test_std': 0, 'metric': params['metric']}

    return res_dict

def linear_probe_node(embeddings, y, splits, params, device):
    """Linear probe evaluation for node classification"""

    metric_names = _get_metric_names(params)
    _validate_metrics(metric_names, CLASSIFICATION_METRICS | REGRESSION_METRICS, 'linear_probe_node')
    is_regression = any(_is_regression_metric(metric_name) for metric_name in metric_names)
    if is_regression and any(metric_name not in REGRESSION_METRICS for metric_name in metric_names):
        raise ValueError('linear_probe_node does not support mixing regression and classification metrics.')
    if (not is_regression) and any(metric_name not in CLASSIFICATION_METRICS for metric_name in metric_names):
        raise ValueError('linear_probe_node does not support clustering metrics.')
    primary_metric = metric_names[0]
    best_val_accs = {metric_name: [] for metric_name in metric_names}
    best_test_accs = {metric_name: [] for metric_name in metric_names}
    y_eval = y
    if not is_regression and params.get('num_tasks') is None:
        y_eval, num_classes = _encode_class_labels(y, params)
    else:
        num_classes = params['output_dim']

    for split in splits:
        # Train linear classifier
        train_mask = split['train']
        val_mask = split['val']
        test_mask = split['test']

        X_train = embeddings[train_mask]
        y_train = y_eval[train_mask]

        classifier = nn.Linear(embeddings.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=params['linear_probe_lr'], weight_decay=params['linear_probe_weight_decay'])

        # Train for a fixed number of epochs
        num_epochs = params['linear_probe_epochs']
        best_val_acc = float('inf') if _is_regression_metric(primary_metric) else float('-inf')
        best_val_scores = None
        best_test_scores = None

        for epoch in range(num_epochs):
            classifier.train()
            optimizer.zero_grad()

            # Training in mini-batches
            train_batch_size = 200000
            train_idx = torch.randperm(X_train.size(0))
            train_loss = 0
            num_train_batches = (X_train.size(0) + train_batch_size - 1) // train_batch_size

            for i in range(num_train_batches):
                batch_idx = train_idx[i * train_batch_size:(i + 1) * train_batch_size]
                batch_X = X_train[batch_idx].to(device)
                batch_y = y_train[batch_idx].to(device)

                batch_logits = classifier(batch_X)
                if params.get('num_tasks') is not None:
                    if params['num_tasks'] == 1:
                        loss = F.binary_cross_entropy_with_logits(batch_logits.squeeze(), batch_y.float())
                    else:
                        loss = F.binary_cross_entropy_with_logits(batch_logits, batch_y.float())
                elif is_regression:
                    loss = F.mse_loss(_regression_prediction(batch_logits), batch_y.float())
                else:
                    loss = F.cross_entropy(batch_logits, batch_y)

                loss.backward()
                train_loss += loss.item()

            optimizer.step()

            # Evaluation in mini-batches
            classifier.eval()
            val_batch_size = 200000
            test_batch_size = 200000

            with torch.no_grad():
                # Validation
                val_logits_list = []
                num_val_batches = (embeddings[val_mask].size(0) + val_batch_size - 1) // val_batch_size
                for i in range(num_val_batches):
                    start_idx = i * val_batch_size
                    end_idx = min((i + 1) * val_batch_size, embeddings[val_mask].size(0))
                    batch_logits = classifier(embeddings[val_mask][start_idx:end_idx].to(device))
                    if is_regression:
                        batch_logits = _regression_prediction(batch_logits)
                    val_logits_list.append(batch_logits)
                val_logits = torch.cat(val_logits_list, dim=0)

                # Test
                test_logits_list = []
                num_test_batches = (embeddings[test_mask].size(0) + test_batch_size - 1) // test_batch_size
                for i in range(num_test_batches):
                    start_idx = i * test_batch_size
                    end_idx = min((i + 1) * test_batch_size, embeddings[test_mask].size(0))
                    batch_logits = classifier(embeddings[test_mask][start_idx:end_idx].to(device))
                    if is_regression:
                        batch_logits = _regression_prediction(batch_logits)
                    test_logits_list.append(batch_logits)
                test_logits = torch.cat(test_logits_list, dim=0)

                val_scores = _evaluate_metric_bundle(val_logits, y_eval[val_mask].to(device), params=params)
                test_scores = _evaluate_metric_bundle(test_logits, y_eval[test_mask].to(device), params=params)
                val_acc = val_scores[primary_metric]

                if _is_better_metric(val_acc, best_val_acc, primary_metric):
                    best_val_acc = val_acc
                    best_val_scores = val_scores
                    best_test_scores = test_scores
        for metric_name in metric_names:
            best_val_accs[metric_name].append(best_val_scores[metric_name])
            best_test_accs[metric_name].append(best_test_scores[metric_name])

    return _summarize_metric_results(metric_names, best_val_accs, best_test_accs)

def knn_node(embeddings, y, splits, params, device, k):
    """kNN evaluation using sklearn KNN estimators."""

    if k <= 0:
        raise ValueError('k must be a positive integer.')

    metric_names = _get_metric_names(params)
    _validate_metrics(metric_names, CLASSIFICATION_METRICS | REGRESSION_METRICS, 'knn_node')
    is_regression = any(_is_regression_metric(metric_name) for metric_name in metric_names)
    if is_regression and any(metric_name not in REGRESSION_METRICS for metric_name in metric_names):
        raise ValueError('knn_node does not support mixing regression and classification metrics.')
    if (not is_regression) and any(metric_name not in CLASSIFICATION_METRICS for metric_name in metric_names):
        raise ValueError('knn_node does not support clustering metrics.')
    best_val_accs = {metric_name: [] for metric_name in metric_names}
    best_test_accs = {metric_name: [] for metric_name in metric_names}
    knn_weights = params.get('knn_weights', 'uniform')
    knn_metric = params.get('knn_metric', 'minkowski')
    knn_p = params.get('knn_p', 2)
    knn_n_jobs = params.get('knn_n_jobs', -1)

    if not is_regression:
        y_encoded_cpu, num_classes = _encode_class_labels(y, params)
        y_encoded_cpu = y_encoded_cpu.detach().cpu()
    else:
        y_cpu = y.detach().cpu()
    embeddings_cpu = embeddings.detach().cpu()

    for split in splits:
        train_mask = split['train']
        val_mask = split['val']
        test_mask = split['test']

        train_embeddings = embeddings_cpu[train_mask]
        train_labels = y_cpu[train_mask] if is_regression else y_encoded_cpu[train_mask]

        if train_embeddings.size(0) == 0:
            raise ValueError('Each split must contain at least one training sample for kNN.')

        k_neighbors = min(k, train_embeddings.size(0))

        if is_regression:
            estimator = KNeighborsRegressor(
                n_neighbors=k_neighbors,
                weights=knn_weights,
                metric=knn_metric,
                p=knn_p,
                n_jobs=knn_n_jobs,
            )
            estimator.fit(train_embeddings.numpy(), train_labels.numpy())

            val_preds = estimator.predict(embeddings_cpu[val_mask].numpy())
            test_preds = estimator.predict(embeddings_cpu[test_mask].numpy())

            val_logits = torch.tensor(val_preds, dtype=torch.float, device=y.device)
            test_logits = torch.tensor(test_preds, dtype=torch.float, device=y.device)
            y_val_eval = y[val_mask]
            y_test_eval = y[test_mask]
        else:
            estimator = KNeighborsClassifier(
                n_neighbors=k_neighbors,
                weights=knn_weights,
                metric=knn_metric,
                p=knn_p,
                n_jobs=knn_n_jobs,
            )
            estimator.fit(train_embeddings.numpy(), train_labels.numpy())

            val_probs = estimator.predict_proba(embeddings_cpu[val_mask].numpy())
            test_probs = estimator.predict_proba(embeddings_cpu[test_mask].numpy())

            val_probs_full = np.zeros((val_probs.shape[0], num_classes), dtype=np.float32)
            test_probs_full = np.zeros((test_probs.shape[0], num_classes), dtype=np.float32)
            val_probs_full[:, estimator.classes_] = val_probs
            test_probs_full[:, estimator.classes_] = test_probs

            val_logits = torch.tensor(val_probs_full, dtype=torch.float, device=y.device)
            test_logits = torch.tensor(test_probs_full, dtype=torch.float, device=y.device)
            y_val_eval = y_encoded_cpu[val_mask].to(device)
            y_test_eval = y_encoded_cpu[test_mask].to(device)

        val_scores = _evaluate_metric_bundle(val_logits, y_val_eval, params=params)
        test_scores = _evaluate_metric_bundle(test_logits, y_test_eval, params=params)

        for metric_name in metric_names:
            best_val_accs[metric_name].append(val_scores[metric_name])
            best_test_accs[metric_name].append(test_scores[metric_name])

    return _summarize_metric_results(metric_names, best_val_accs, best_test_accs)


def kmeans_node(embeddings, y, splits, params, device):
    """KMeans evaluation on the full dataset across repeated random seeds."""

    metric_names = _get_metric_names(params)
    _validate_metrics(metric_names, CLUSTERING_METRICS, 'kmeans_node')
    test_scores_by_metric = {metric_name: [] for metric_name in metric_names}

    y_encoded_cpu, num_classes = _encode_class_labels(y, params)
    y_encoded_cpu = y_encoded_cpu.detach().cpu()
    embeddings_array = embeddings.detach().cpu().numpy()

    default_clusters = num_classes
    requested_clusters = params.get('kmeans_num_clusters', default_clusters)
    kmeans_n_init = params.get('kmeans_n_init', 10)
    kmeans_max_iter = params.get('kmeans_max_iter', 500)
    kmeans_random_state = params.get('kmeans_random_state', 0)
    num_runs = int(params.get('kmeans_num_runs', len(splits) if splits else 10))
    if num_runs < 1:
        raise ValueError('kmeans_num_runs must be at least 1.')

    num_clusters = min(int(requested_clusters), embeddings_array.shape[0])
    if num_clusters < 2:
        raise ValueError('KMeans evaluation requires at least 2 clusters.')

    if params.get('PCA_embeddings', params.get('pca_embeddings', True)):
        pca_n_components = int(params.get('PCA_n_components', params.get('pca_n_components', 64)))
        if pca_n_components < 1:
            raise ValueError('PCA_n_components must be at least 1.')
        if embeddings_array.shape[1] > pca_n_components:
            pca = PCA(
                n_components=min(pca_n_components, embeddings_array.shape[0], embeddings_array.shape[1]),
                random_state=kmeans_random_state,
            )
            embeddings_array = pca.fit_transform(embeddings_array)

    clustering_embeddings = torch.tensor(embeddings_array, dtype=torch.float, device=y.device)
    y_eval = y_encoded_cpu.to(device)

    for run_idx in range(num_runs):
        run_seed = int(kmeans_random_state) + run_idx
        estimator = KMeans(
            n_clusters=num_clusters,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=run_seed,
        )
        cluster_assignments = estimator.fit_predict(embeddings_array)
        predictions = torch.tensor(cluster_assignments, dtype=torch.long, device=y.device)

        test_scores = {
            metric_name: evaluate(
                predictions,
                y_eval,
                params=params,
                embeddings=clustering_embeddings if metric_name == 'silhouette' else None,
                metric_name=metric_name,
            )
            for metric_name in metric_names
        }
        for metric_name in metric_names:
            test_scores_by_metric[metric_name].append(test_scores[metric_name])

    zero_val_scores = {metric_name: [0] * num_runs for metric_name in metric_names}
    return _summarize_metric_results(metric_names, zero_val_scores, test_scores_by_metric)

def get_node_neighbors_sparse(graph, graph_f_tensor):
    
    num_nodes = graph.size(0)
    adj = torch.sparse_coo_tensor(graph.edge_index, torch.ones(graph.edge_index.size(1)),
                                  (num_nodes, num_nodes))
    agg_neighbors = torch.sparse.mm(adj, one_hot(graph_f_tensor)).softmax(dim=-1)
    graph.agg_neighbors = agg_neighbors
    logger.debug(f'graph:{graph}')

    return graph

def build_radius_graph(x, coord, radius, max_neighbors = 1000):
    radius_edge_index = radius_graph(coord, 
                                     r=radius, loop=False, max_num_neighbors=max_neighbors)
    graph = Data(x=x, edge_index=coalesce(radius_edge_index))

    logger.debug(f'graph:{graph}')
    logger.debug(f'if undirected graph:{graph.is_undirected()}')

    return graph
