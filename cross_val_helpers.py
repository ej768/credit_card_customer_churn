import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_cv_roc_curve(tprs, fprs, aucs, title="ROC Curve (Cross-Validation)"):
    """
    Plots the ROC curve with cross-validation folds and mean AUC.

    Parameters:
    - tprs: list of interpolated TPRs from each fold
    - aucs: list of AUC scores for each fold
    - fprs: list of FPRs from each fold
    - title: title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Compute the mean True Positive Rate (TPR)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure that the last point is at the top right corner

    # Compute the mean AUC
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot individual folds
    for i, (fpr, tpr, roc_auc) in enumerate(zip(fprs, tprs, aucs)):
        plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {i+1} AUC = {roc_auc:.2f}")

    # Plot the mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.2f})", lw=2)
    
    # Plot the random classifier (diagonal line)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    # Adding labels, title, and grid
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_labels=["Negative", "Positive"]):
    """
    Plots a confusion matrix using seaborn heatmap.

    Parameters:
    - cm: 2D array (confusion matrix)
    - class_labels: list of class names for axes
    - title: title of the plot
    - normalize: whether to normalize by row
    """
    # Normalize the inputs
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    # Create Seaborn Heatmap
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)

    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def run_stratified_cv(mdl, skf, X, y):
    """
    Performs stratified K-fold cross-validation on a given classification model and collects performance metrics.

    Parameters:
    - mdl: A scikit-learn compatible classifier with `fit`, `predict`, and `predict_proba` methods.

    Returns:
    - tprs (list of np.ndarray): Interpolated true positive rates for each fold (used for plotting mean ROC).
    - fprs (list of np.ndarray): False positive rate values (common across folds for plotting).
    - aucs (list of float): Area Under the Curve (AUC) values for each fold.
    - conf_matrix (np.ndarray): Summed confusion matrix over all folds.

    Notes:
    - This function assumes binary classification (i.e., labels are 0 and 1).
    - Requires the global variables `X`, `y`, and `skf` to be defined:
        * `X` (pd.DataFrame): Feature matrix
        * `y` (pd.Series or np.array): Binary target variable
        * `skf` (StratifiedKFold): StratifiedKFold instance used for splitting data
    """
    # Lists to store metrics across folds
    tprs = []        # Interpolated True Positive Rates for each fold (for ROC curve)
    fprs = []        # False Positive Rate x-axis (mean_fpr used for all folds)
    aucs = []        # AUC values for each fold
    mean_fpr = np.linspace(0, 1, 100)  # Fixed FPR range for interpolation and averaging

    # Initialize an empty confusion matrix (2x2 for binary classification)
    conf_matrix = np.zeros((2, 2))

    # Perform stratified K-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        # Split data into train and test sets using fold indices
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model on training data
        mdl.fit(X_train, y_train)

        # Predict class probabilities on test set
        y_proba = mdl.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

        # Compute ROC curve for this fold
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        # Interpolate TPR to match fixed FPR points and store
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        fprs.append(mean_fpr)  # Use consistent FPR for all folds (helps in plotting)

        # Calculate AUC for this fold and store
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Predict class labels and update confusion matrix
        y_pred = mdl.predict(X_test)
        conf_matrix += confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Return all metrics collected across folds
    return tprs, fprs, aucs, conf_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def summarize_cv_results(name, aucs, conf_matrix):
    """
    Summarize the performance of a model across cross-validation folds.

    Parameters:
    - name (str): Model name
    - aucs (list): List of AUC scores per fold
    - conf_matrix (np.ndarray): Summed confusion matrix from all folds

    Returns:
    - dict: Summary metrics for the model
    """
    tn, fp, fn, tp = conf_matrix.ravel()
    
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Model": name,
        "Mean AUC": np.mean(aucs),
        "Std AUC": np.std(aucs),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def plot_all_models_roc(models_roc_data, title="Model Comparison - ROC Curves"):
    """
    Plots the mean ROC curve for multiple models on the same graph.

    Parameters:
    - models_roc_data (list of tuples): Each tuple is 
        (model_name, tprs, mean_fpr, aucs)
    - title (str): Plot title
    """
    plt.figure(figsize=(10, 7))

    for model_name, tprs, mean_fpr, aucs in models_roc_data:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()