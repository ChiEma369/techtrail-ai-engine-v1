
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

METRIC_COLS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']


def compute_metrics(y_true, y_pred):
    """Accuracy, Precision/Recall/F1 (macro)."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(p),
        'recall_macro': float(r),
        'f1_macro': float(f1),
    }


def per_class_report(y_true, y_pred, labels=None):
    """Bảng precision / recall / f1 / support cho từng lớp."""
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    index = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    return pd.DataFrame({'precision': p, 'recall': r, 'f1': f1, 'support': s},
                        index=index)


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix',
                          normalize=True, save_path=None, figsize=(10, 8)):
    """Vẽ confusion matrix (mặc định chuẩn hóa theo hàng = recall từng lớp)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_show = cm.astype(float)
    if normalize:
        cm_show = cm_show / np.maximum(cm_show.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_show, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    thresh = cm_show.max() / 2 if cm_show.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f'{cm_show[i, j]:.2f}' if normalize else f'{int(cm[i, j])}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if cm_show[i, j] > thresh else 'black')
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return cm


def most_confused_pairs(y_true, y_pred, labels, top_k=10):
    """Các cặp (lớp thật -> lớp dự đoán sai) hay nhầm nhất, phục vụ error analysis."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = [(labels[i], labels[j], int(cm[i, j]))
            for i in range(len(labels)) for j in range(len(labels))
            if i != j and cm[i, j] > 0]
    df = pd.DataFrame(rows, columns=['true_label', 'pred_label', 'count'])
    return df.sort_values('count', ascending=False).head(top_k).reset_index(drop=True)


def save_predictions(path, ids, y_true, y_pred):
    """Lưu dự đoán theo quy ước: id, y_true, y_pred."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pd.DataFrame({'id': list(ids), 'y_true': list(y_true),
                  'y_pred': list(y_pred)}).to_csv(path, index=False)


def load_predictions(path):
    """Đọc file dự đoán, trả về DataFrame có cột id, y_true, y_pred."""
    df = pd.read_csv(path)
    missing = {'id', 'y_true', 'y_pred'} - set(df.columns)
    if missing:
        raise ValueError(f'{path} thiếu cột: {sorted(missing)}')
    return df


def evaluate_model(name, y_true, y_pred, labels, out_dir='results',
                   plot=True, normalize=True):
    """Đánh giá đầy đủ 1 mô hình: in report, vẽ/lưu confusion matrix, lưu metrics JSON."""
    metrics = compute_metrics(y_true, y_pred)
    print(f'\n=== {name} ===')
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print('Tổng hợp:', {k: round(v, 4) for k, v in metrics.items()})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    per_class_report(y_true, y_pred, labels).to_csv(
        os.path.join(out_dir, f'{name}_per_class.csv'))
    if plot:
        plot_confusion_matrix(
            y_true, y_pred, labels, title=f'Confusion Matrix - {name}',
            normalize=normalize,
            save_path=os.path.join(out_dir, f'{name}_confusion_matrix.png'))
    return metrics


def compare_models(results):
    """
    results: {tên mô hình: (y_true, y_pred)}
    Trả về bảng so sánh Accuracy / Precision / Recall / F1-macro.
    """
    rows = {name: compute_metrics(yt, yp) for name, (yt, yp) in results.items()}
    return pd.DataFrame(rows).T[METRIC_COLS]


def plot_comparison(comp_df, title='So sánh hiệu năng mô hình', save_path=None):
    """Biểu đồ cột nhóm cho bảng so sánh từ compare_models()."""
    ax = comp_df.plot(kind='bar', figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='lower right')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
