import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, precision_recall_curve, average_precision_score)
    
def plot_history(history, type, name='stage 1'):
    n_cols = 3 if type == 'whole' else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols,4))

    # Accuracy
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    ax = axes[0] if n_cols > 1 else axes

    epochs = range(1,len(accuracy) + 1)

    ax.plot(epochs, accuracy,"o-", label="Training accuracy")
    ax.plot(epochs, val_accuracy,"o-", label="Validation accuracy")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True,ls='--')

    # Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    ax = axes[-1] if n_cols > 1 else axes
    ax.plot(epochs, loss,"o-", label="Training loss")
    ax.plot(epochs, val_loss,"o-", label="Validation loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True,ls='--')

    # AUC (binary only)
    if type == 'whole':
        auc = history.history['auc']
        val_auc = history.history['val_auc']
        ax = axes[1]
        ax.plot(epochs, auc,"o-", label="Training AUC")
        ax.plot(epochs, val_auc,"o-", label="Validation AUC")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True,ls='--')

    plt.savefig(name)
    plt.tight_layout()
    plt.show()

def plot_conf_mat(cm, class_names, normalize=False, title="Confusion matrix"):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,f"{cm[i,j]:.2f}" if normalize else f"{cm[i,j]:d}",
                    ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show() 

def evaluate_multiclass(model, val_ds, num_classes, class_names):
    y_true = None
    y_prob = None

    try:
        y_true = val_ds.classes.astype(int)
        y_prob = model.predict(val_ds, verbose=0)
    except AttributeError:
        y_prob_list, y_true_list = [],[]
        for x, y in val_ds:
            p = model.predict(x, verbose=0)
            y_prob_list.append(p)
            y_true_list.append(np.array(y))

        y_prob = np.concatenate(y_prob_list, axis=0)
        y_true_raw = np.concatenate(y_true_list, axis=0)

        if y_true_raw.ndim == 2 and y_true_raw.shape[1] == y_prob.shape[1]:
            y_true = np.argmax(y_true_raw, axis=1)
        else:
            y_true = y_true_raw.astype(int).ravel()

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    y_pred = np.argmax(y_prob, axis=1)
    cm = np.array(confusion_matrix(y_true, y_pred))

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)

    plot_conf_mat(cm, class_names=class_names, normalize=False, title="Confusion matrix (counts)")
    plot_conf_mat(cm, class_names=class_names, normalize=True, title="Confusion matrix (normalized)")

    return {
        "confusion_maxtrix": cm.tolist(),
        "classification_report": report
    }





        