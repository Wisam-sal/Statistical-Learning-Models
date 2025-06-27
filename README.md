# Statistical Learning Models

This project contains implementations and experiments with various statistical learning models, focusing mainly on supervised learning techniques. The goal is to explore, analyze, and compare different algorithms for predictive modeling and data-driven inference.

The data used is taken from Nvidia's https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces. 

## Project Overview

The project aims to:

- Investigate the performance of classical statistical learning algorithms.
- Apply these models to smartspace dataset for classification and regression tasks.
- Evaluate model performance metrics.

## Methods and Models

- **Polynomial Regression** and **Logistic Regression** 
- **Genral Additive Model (GAMs)**
- **Tree-based Ensemble Methods: Random Forests** and **Gradient Boosting Machine (GBM)**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Multilayer Perceptron (MLP)**

## Workflow
**Extract, Transform, and Load**:
- The Ground Truth JSON files are loaded into 3 main classes:
    - **SmartSpace Class**: This class models the overall environment or context (e.g., a warehouse, hospital, or area) where sensors are deployed and unit move freely.
    - **Sensor Class**: represents individual sensors (e.g., cameras) within the smartspace. handles sensor metadata (location, type, calibration, etc.).
    - **Unit Class**: represents a moving unit (e.g., Forklift, NovaCarter, Person, Transporter). Contains the trajectory of the unit as well as sensors' 2D and 3D bounding-boxes capturing the unit's trajectory per frame. 

- Various engineered features are then processed and saved into CSV files.
**Data Exploration, models performance metrics**:
- The two Notebooks explore the data and benchmark the performance of various machine learning models (regression and classification).

## Getting Started
For a quick plug-n-play: Download deliverables/reproducability to get a single folder containing the preprocessed data and analysis Notebooks.

To reproduce the entire project:

1. Clone the repository.
2. Install required dependencies (see `requirements.txt`).
3. run the ETL.py script to reproduce the data processing part.
4. run Notebooks/2D_position_regression.ipynb for data exploration & regression analysis.
5. run Notebooks/human_robot_classification.ipynb for data exploration & classification analysis.
