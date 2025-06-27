"""
Utility functions for evaluating and visualizing binary classification models.

Includes:

- Metric evaluation:

    * evaluate_model_at_threshold: Compute key metrics (accuracy, sensitivity, specificity, etc.) at a given threshold.
    * bootstrap_metric: Compute confidence intervals via bootstrapping.
    * find_best_threshold: Search for threshold that optimizes a metric.
    * print_metrics: Nicely formats and prints evaluation results.

- Visualization:

    * plot_confusion_mat: Confusion matrix heatmap.
    * plot_auc: Interactive ROC curve with threshold info.
    * radar_metrics_multi: Radar plot to compare multiple models.
    * plot_feature_importances: Bar plot for feature importances/coefs across models.

- Statistical utilities:

    * check_normality: Shapiro-Wilk normality test for DataFrame columns.
    * boxcox_transform_df: Apply Box-Cox transformation to numeric columns.
    * plot_shapiro_pvalues: Visualize Shapiro test results.

"""

from sklearn.metrics import roc_curve
from sklearn.metrics import (balanced_accuracy_score, recall_score, 
                             precision_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import shapiro, boxcox
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def npv_score(y_true, y_pred):
    tn, _, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0

def auc_score(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)

def bootstrap_metric(y_true, y_input, metric_func, n_bootstrap=1000, alpha=0.05, seed=0):
    np.random.seed(seed)
    stats = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        stats.append(metric_func(y_true[idx], y_input[idx]))
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return (lower, upper)

def evaluate_model_at_threshold(model, X_test, y_test, threshold=0.5, bootstrap=False, n_bootstrap=1000):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results = {
        "threshold": threshold,
        "accuracy": balanced_accuracy_score(y_test, y_pred),
        "sensitivity": recall_score(y_test, y_pred),  # TPR
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0),  # TNR
        "ppv": precision_score(y_test, y_pred),  # PPV
        "npv": float(tn / (tn + fn) if (tn + fn) > 0 else 0),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    if bootstrap:
        y_test_np = np.array(y_test)
        y_pred_np = np.array(y_pred)
        y_proba_np = np.array(y_proba)
        results["ci"] = {
            "accuracy": bootstrap_metric(y_test_np, y_pred_np, balanced_accuracy_score, n_bootstrap),
            "sensitivity": bootstrap_metric(y_test_np, y_pred_np, recall_score, n_bootstrap),
            "specificity": bootstrap_metric(y_test_np, y_pred_np, specificity_score, n_bootstrap),
            "ppv": bootstrap_metric(y_test_np, y_pred_np, precision_score, n_bootstrap),
            "npv": bootstrap_metric(y_test_np, y_pred_np, npv_score, n_bootstrap),
            "f1": bootstrap_metric(y_test_np, y_pred_np, f1_score, n_bootstrap),
            "auc": bootstrap_metric(y_test_np, y_proba_np, auc_score, n_bootstrap)
        }

    return results

def find_best_threshold(model, X_test, y_test, metric='auc', thresholds=np.linspace(0, 1, 101), bootstrap=False, n_bootstrap=100):
    
    best_result = None
    best_score = float('inf')
    target_value=0.9

    for t in thresholds:
        result = evaluate_model_at_threshold(model, X_test, y_test, threshold=t,
                                             bootstrap=bootstrap, n_bootstrap=n_bootstrap)
        score = result[metric]
        
        diff = abs(score - target_value)
        if diff < best_score:
            best_score = diff
            best_result = result

    return best_result

def print_metrics(metrics_dict):
    print('=' * 30)
    print(f"Best threshold: {metrics_dict['threshold']:.2f}")
    print('=' * 30)

    def format_ci(name):
        if "ci" in metrics_dict and name in metrics_dict["ci"] and metrics_dict["ci"][name] is not None:
            ci_low, ci_high = metrics_dict["ci"][name]
            return f" [{ci_low:.3f}, {ci_high:.3f}]"
        return ""

    print(f"Accuracy   : {metrics_dict['accuracy']:.3f}{format_ci('accuracy')}")
    print(f"Sensitivity: {metrics_dict['sensitivity']:.3f} (TPR){format_ci('sensitivity')}")
    print(f"Specificity: {metrics_dict['specificity']:.3f} (TNR){format_ci('specificity')}")
    print(f"Precision  : {metrics_dict['ppv']:.3f} (PPV){format_ci('ppv')}")
    print(f"NPV        : {metrics_dict['npv']:.3f}{format_ci('npv')}")
    print(f"F1 Score   : {metrics_dict['f1']:.3f}{format_ci('f1')}")
    print(f"AUC        : {metrics_dict['auc']:.3f}{format_ci('auc')}")
    print('=' * 30)

def plot_confusion_mat(model, X_test, y_test, threshold=0.5, labels=None, title="Confusion Matrix"):
    """
    Plots confusion matrix for predictions at a given threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{title} (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.show()

def plot_auc(y_true, y_pred_proba):
    
    """Interactive ROC graph"""

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    accuracies = []
    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        accuracies.append(balanced_accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    hover_text = [
        f"Threshold: {thresh:.2f}<br>Accuracy: {acc:.3f}<br>F1: {f1:.3f}"
        for thresh, acc, f1 in zip(thresholds, accuracies, f1_scores)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines+markers',
        text=hover_text,
        hoverinfo='text',
        name='ROC Curve'
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        width = 800,
        height = 500
    )
    return fig

def radar_metrics_multi(metrics_dicts, model_names):
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc', 'f1']
    categories = metrics_to_plot + [metrics_to_plot[0]]
    fig = go.Figure()
    for metrics, name in zip(metrics_dicts, model_names):
        values = [metrics[m] for m in metrics_to_plot]
        values += [values[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            name=name,
            line=dict(width=3),
            opacity=1
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='Performance Metrics Comparison',
        width=700,
        height=500
    )

    return fig

def plot_feature_importances(models, model_names, feature_names):
    """
    Plot feature importances or coefficients for multiple models.
    """
    importances = []
    for model in models:
        if hasattr(model, 'coef_'):
            imp = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'named_steps'):
            last_step = list(model.named_steps.values())[-1]
            if hasattr(last_step, 'coef_'):
                imp = np.abs(last_step.coef_[0])
            elif hasattr(last_step, 'feature_importances_'):
                imp = last_step.feature_importances_
            else:
                imp = np.full(len(feature_names), np.nan)
        else:
            imp = np.full(len(feature_names), np.nan)
        importances.append(imp)
    importances = np.array(importances)

    fig = go.Figure()
    for i, name in enumerate(model_names):
        fig.add_trace(go.Bar(
            x=feature_names,
            y=importances[i],
            name=name
        ))
    fig.update_layout(
        barmode='group',
        title='Feature Importances / Coefficients by Model',
        xaxis_title='Feature',
        yaxis_title='Importance (absolute value)',
        width=900,
        height=500
    )
    fig.show()

def plot_auc_paper(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, lw=1.5, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC Curve', fontsize=11)
    plt.tick_params(axis='both', labelsize=9)
    plt.legend(loc='lower right', fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()

def radar_metrics_multi_matplotlib(metrics_dicts, model_names):
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc', 'f1']
    num_vars = len(metrics_to_plot)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], fontsize=10, weight='bold')

    ax.set_rlabel_position(180 / num_vars)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.set_ylim(0, 1)

    for metrics, name in zip(metrics_dicts, model_names):
        values = [metrics[m] for m in metrics_to_plot]
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, label=name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=9)
    plt.title("Model Performance Comparison", size=14, weight='bold', pad=20)
    plt.tight_layout()
    return fig
