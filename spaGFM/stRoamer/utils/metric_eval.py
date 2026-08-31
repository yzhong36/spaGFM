import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassMatthewsCorrCoef

def normalize_metric_names(metric):
    if isinstance(metric, str):
        metric_list = [part.strip() for part in metric.split(',') if part.strip()]
    elif isinstance(metric, (list, tuple)):
        metric_list = [str(part).strip() for part in metric if str(part).strip()]
    else:
        raise TypeError(f"Unsupported metric specification type: {type(metric)}")

    if not metric_list:
        raise ValueError("At least one metric must be provided.")
    return metric_list


def evaluate(pred, y, params=None, embeddings=None, metric_name=None):
    metric = metric_name or params['metric']

    if metric == 'acc':
        return eval_acc(pred, y)
    elif metric == 'acc_macro':
        return eval_acc_macro(pred, y)
    elif metric == 'f1':
        return eval_f1(pred, y)
    elif metric == 'f1_macro':
        return eval_f1_macro(pred, y)
    elif metric == 'mcc':
        return eval_mcc(pred, y)
    elif metric == 'ari':
        return eval_ari(pred, y)
    elif metric == 'nmi':
        return eval_nmi(pred, y)
    elif metric == 'silhouette':
        return eval_silhouette(pred, embeddings)
    elif metric == 'rmse':
        return eval_rmse(pred, y)
    elif metric == 'mae':
        return eval_mae(pred, y)
    else:
        raise ValueError(f"Metric {metric} is not supported.")


def _as_label_array(y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(dim=-1)
    return y_pred.detach().cpu().numpy()


def _as_target_array(y_true):
    if y_true.ndim == 2:
        y_true = y_true.squeeze()
    return y_true.detach().cpu().numpy()

# Only works for single task classification
def eval_acc(y_pred, y_true):
    device = y_pred.device
    y_true = y_true.to(device)
    num_classes = y_pred.size(1)

    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    return evaluator(y_pred, y_true).item()

def eval_acc_macro(y_pred, y_true):
    device = y_pred.device
    y_true = y_true.to(device)
    num_classes = y_pred.size(1)

    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    evaluator = Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)

    return evaluator(y_pred, y_true).item()

def eval_f1(y_pred, y_true):
    device = y_pred.device
    y_true = y_true.to(device)
    num_classes = y_pred.size(1)

    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    evaluator = F1Score(task="multiclass", num_classes=num_classes).to(device)

    return evaluator(y_pred, y_true).item()

def eval_f1_macro(y_pred, y_true):
    device = y_pred.device
    y_true = y_true.to(device)
    num_classes = y_pred.size(1)

    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    evaluator = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)

    return evaluator(y_pred, y_true).item()

def eval_mcc(y_pred, y_true):
    device = y_pred.device
    y_true = y_true.to(device)
    num_classes = y_pred.size(1)

    if y_true.ndim == 2:
        y_true = y_true.squeeze()

    evaluator = MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device)

    return evaluator(y_pred, y_true).item()


def eval_ari(y_pred, y_true):
    return adjusted_rand_score(_as_target_array(y_true), _as_label_array(y_pred))


def eval_nmi(y_pred, y_true):
    return normalized_mutual_info_score(_as_target_array(y_true), _as_label_array(y_pred))


def eval_silhouette(y_pred, embeddings):
    if embeddings is None:
        raise ValueError("Silhouette score requires embeddings to be passed into evaluate().")

    cluster_labels = _as_label_array(y_pred)
    unique_clusters = np.unique(cluster_labels)
    if unique_clusters.size < 2 or unique_clusters.size >= cluster_labels.shape[0]:
        return float("nan")

    embedding_array = embeddings.detach().cpu().numpy()
    return float(silhouette_score(embedding_array, cluster_labels))

def eval_rmse(y_pred, y_true):
    if len(y_true.shape) == 1:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        return rmse

    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return sum(rmse_list) / len(rmse_list)

def eval_mae(y_pred, y_true):
    if len(y_true.shape) == 1:
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        mae = np.abs(y_true - y_pred).mean()
        return mae

    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    mae_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        mae_list.append(np.abs(y_true[is_labeled, i] - y_pred[is_labeled, i]).mean())

    return sum(mae_list) / len(mae_list)
