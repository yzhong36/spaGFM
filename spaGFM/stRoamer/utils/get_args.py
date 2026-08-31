import argparse
import json

def _json_dict(s: str) -> dict:
    """Parse a JSON string into a dict. Used as argparse type for dict arguments."""
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")

def _json_or_raw(s: str):
    """Parse JSON when provided, otherwise keep the raw string."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s

def _str_to_bool(s):
    """Parse bool-like CLI values without argparse's type=bool footgun."""
    if isinstance(s, bool):
        return s
    value = str(s).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {s}")

def get_default_pretraining_args():
    parser = argparse.ArgumentParser(description="Default arguments for pretraining")
    ## Initial settings
    parser.add_argument(
        "--master_port",
        type=str,
        default="12345",
        help="Port used for torch.distributed rendezvous (set differently to run multiple local jobs)",
    )
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--dataset", type=str)

    ## Model settings
    parser.add_argument(
        "--mode",
        type=str,
        help="Run mode: pretrain, adaptation, or inference",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Required when --mode adaptation; checkpoint path used to initialize model weights",
    )
    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--n_total_rw", type=int)
    parser.add_argument("--n_visible_rw", type=int)
    parser.add_argument("--walk_length", type=int)
    parser.add_argument("--rw_encoder", type=_json_dict, default='{}')
    parser.add_argument("--subgraph_encoder", type=_json_dict, default='{}')
    parser.add_argument("--subgraph_decoder", type=_json_dict, default='{}')
    parser.add_argument("--mask_feature_ratio", type=float, default=0.2)
    parser.add_argument("--mask_node_ratio", type=float, default=0.2)
    parser.add_argument("--variance_loss", type=_str_to_bool, default=True)
    parser.add_argument("--variance_target", type=float, default=1.0)
    parser.add_argument("--covariance_loss", type=_str_to_bool, default=False)
    parser.add_argument("--covariance_weight", type=float, default=0.0)
    parser.add_argument("--loss_fn", type=str, default="l2")

    ## Training settings
    parser.add_argument("--n_cpu", type=int, default=10)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_epoch_wise_size", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--save_step_wise", type=_str_to_bool, default=True)
    parser.add_argument("--save_step_wise_size", type=int, default=20000)

    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--scale_lr_by_world_size", type=_str_to_bool, default=True)
    parser.add_argument("--warmup_in_global_steps", type=_str_to_bool, default=False)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--stable_steps", type=int)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument('--opt_beta1', type=float, default=0.9)
    parser.add_argument('--opt_beta2', type=float, default=0.995)
    parser.add_argument('--opt_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument("--adamw_decay_all_params", type=_str_to_bool, default=False)
    parser.add_argument("--ema_update_every", type=int, default=10)
    parser.add_argument("--ema_alpha", type=float, default=0.99)

    ## Online evaluation settings
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--eval_label", type=_json_or_raw, default=None)
    parser.add_argument("--eval_embedding_type", type=_json_or_raw, default="subgraph_emb")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_split", type=str, default="low")
    parser.add_argument("--eval_split_repeat", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, default=2025)
    parser.add_argument("--eval_metric", type=str, default="f1_macro")
    parser.add_argument("--min-label-ratio", type=float, default=0.01)
    parser.add_argument("--min-label-count", type=int, default=100)
    parser.add_argument("--linear_probe_lr", type=float, default=0.01)
    parser.add_argument("--linear_probe_weight_decay", type=float, default=0.001)
    parser.add_argument("--linear_probe_epochs", type=int, default=100)
    parser.add_argument("--knn_k", type=int, default=5)
    
    ## Mixed Precision settings
    parser.add_argument("--use_amp", type=_str_to_bool, default=False, help="Enable automatic mixed precision training")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="AMP dtype: float16 or bfloat16")
    
    return vars(parser.parse_args())
