# Utility Functions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import bootstrap, shapiro, probplot
from typing import List

def plot_unit_distance_vs_sensor_area(bounding_area_df: pd.DataFrame, 
                                      unit_id: int, camera_id: str, 
                                      feature_columns : List):
    
    """Plot Unit Trajectory vs Sensor Bounding Box Area"""

    df = bounding_area_df.copy()
    df[df[feature_columns] == 0] = np.nan
    
    unit_df = df[df.unit_id == unit_id]
    unit_df = unit_df.dropna(subset=[camera_id]).reset_index(drop=True)

    fig, axs = plt.subplots(1, 3, figsize=(21, 5))

    axs[0].scatter(
        unit_df[camera_id], unit_df['x_position'],
        alpha=0.6, s=60, edgecolors='k', linewidths=1
    )
    axs[0].plot(
        unit_df[camera_id], unit_df['x_position'],
        color='tab:blue', alpha=0.5, linewidth=2
    )
    axs[0].set_xlabel(f'{camera_id} bounding box area')
    axs[0].set_ylabel(f'#{unit_id} X-position on map')
    axs[0].set_title(f'X Position of unit #{unit_id} vs bounding box area')
    axs[0].grid(True)

    axs[1].scatter(
        unit_df[camera_id], unit_df['y_position'],
        alpha=0.6, s=60, edgecolors='k', linewidths=1
    )
    axs[1].plot(
        unit_df[camera_id], unit_df['y_position'],
        color='tab:orange', alpha=0.5, linewidth=2
    )
    axs[1].set_xlabel(f'{camera_id} bounding box area')
    axs[1].set_ylabel(f'#{unit_id} Y-position on map')
    axs[1].set_title(f'Y Position of unit #{unit_id} vs bounding box area')
    axs[1].grid(True)

    axs[2].scatter(
        unit_df[camera_id], unit_df['unit_distance_from_origin'],
        alpha=0.6, s=60, edgecolors='k', linewidths=1
    )
    axs[2].plot(
        unit_df[camera_id], unit_df['unit_distance_from_origin'],
        color='tab:green', alpha=0.5, linewidth=2
    )
    axs[2].set_xlabel(f'{camera_id} bounding box area')
    axs[2].set_ylabel(f'#{unit_id} distance from origin')
    axs[2].set_title(f'Euclidean distance of unit #{unit_id} vs bounding box area')
    axs[2].grid(True)

    fig.tight_layout()
    plt.show()

def sensor_distance_scatter(bounding_area_df: pd.DataFrame, 
                            camera_id: str, 
                            feature_columns : List):
    
    """Scatter Plot for a given sensor"""

    df = bounding_area_df.copy()
    df[df[feature_columns] == 0] = np.nan

    df = df.dropna(subset=[camera_id]).reset_index(drop=True)

    fig, axs = plt.subplots(2, 3, figsize=(21, 10))

    axs[0, 0].scatter(
        df[camera_id], df['x_position'],
        alpha=0.6, s=60, edgecolors='k', linewidths=1
    )
    axs[0, 0].set_xlabel(f'{camera_id} bounding box area')
    axs[0, 0].set_ylabel(f'Units X-position on map')
    axs[0, 0].set_title(f'X Position of units vs. bounding box area')
    axs[0, 0].grid(True)

    axs[0, 1].scatter(
        df[camera_id], df['y_position'],
        alpha=0.6, s=60, edgecolors='k', linewidths=1
    )
    axs[0, 1].set_xlabel(f'{camera_id} bounding box area')
    axs[0, 1].set_ylabel(f'Units Y-position on map')
    axs[0, 1].set_title(f'Y Position of units vs bounding box area')
    axs[0, 1].grid(True)

    axs[0, 2].scatter(
        df[camera_id], df['unit_distance_from_origin'],
        alpha=0.6, s=60, edgecolors='k', linewidths=1
    )
    axs[0, 2].set_xlabel(f'{camera_id} bounding box area')
    axs[0, 2].set_ylabel(f'Units distance from origin')
    axs[0, 2].set_title(f'Euclidean distance of units vs bounding box area')
    axs[0, 2].grid(True)

    axs[1, 0].hist(df['x_position'].dropna(), bins=30, color='tab:blue', alpha=0.7, edgecolor='k')
    axs[1, 0].set_xlabel('X-position')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Histogram of X-position')
    axs[1, 0].grid(True)

    axs[1, 1].hist(df['y_position'].dropna(), bins=30, color='tab:orange', alpha=0.7, edgecolor='k')
    axs[1, 1].set_xlabel('Y-position')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Histogram of Y-position')
    axs[1, 1].grid(True)

    axs[1, 2].hist(df['unit_distance_from_origin'].dropna(), bins=30, color='tab:green', alpha=0.7, edgecolor='k')
    axs[1, 2].set_xlabel('Distance from origin')
    axs[1, 2].set_ylabel('Frequency')
    axs[1, 2].set_title('Histogram of distance from origin')
    axs[1, 2].grid(True)

    fig.tight_layout()
    plt.show()

def model_analytics(y_test, y_pred, model_name):
    """Regression Model Residual Analysis"""

    res = y_test - y_pred
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    __, shapiro_p = shapiro(res)

    fig, axs = plt.subplots(3, 1, figsize=(5, 9))
    
    # Residual histogram
    sns.histplot(res, bins=30, kde=True, ax=axs[0], color="tab:blue")
    axs[0].set_title(f"{model_name} Residuals Histogram")
    axs[0].axvline(0, color='k', linestyle='--')
    axs[0].set_xlabel("Residual")
    axs[0].set_ylabel("Frequency")
    
    # Q-Q plot
    probplot(res, dist="norm", plot=axs[1])
    axs[1].set_title(f"{model_name} Residuals Q-Q Plot")
    
    # Predicted vs Actual scatter plot
    sns.scatterplot(x=y_test, y=y_pred, ax=axs[2], color="tab:purple", alpha=0.6)
    axs[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axs[2].set_title(f"{model_name} Predicted vs Actual")
    axs[2].set_xlabel("Actual")
    axs[2].set_ylabel("Predicted")
    
    plt.tight_layout()
    plt.show()
    
    # stats summary
    print("="*60)
    print(f"        Model: {model_name}")
    print("="*60)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Residual Mean: {np.mean(res):.4f}, Residual Std: {np.std(res):.4f}")
    print(f"Shapiro-Wilk Test p-value: {shapiro_p:.6f} (normality {'not rejected' if shapiro_p > 0.05 else 'rejected'})")
    print("="*60)