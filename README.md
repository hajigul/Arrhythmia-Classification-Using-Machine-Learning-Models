# Arrhythmia-Classification-Using-Machine-Learning-Models

A machine learning project aimed at classifying cardiac arrhythmias using clinical data. The dataset contains 452 instances across 16 classes, with 279 features related to ECG readings, patient demographics, and health metrics.


Objective : Predict whether a person has arrhythmia and classify it into one of 12 available groups.

Goal: 
To build and evaluate multiple classification models to accurately detect and classify different types of cardiac arrhythmias using minimal feature sets and high recall/sensitivity.


Arrhythmia-Classification/
│
├── data_loader.py            # Data loading and preprocessing
├── ml_model.py               # Classical ML model definitions
├── dl_model.py               # Deep learning model (placeholder)
├── evaluation.py             # Model evaluation, plotting, and saving results
├── config.py                 # Configuration and paths
├── main.py                   # Main script to run full pipeline
├── README.md                 # This file
└── requirements.txt          # Python dependencies
⚙️ Requirements
Make sure you have these packages installed:

bash


1
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn


Clone the repo:
git clone https://github.com/yourusername/Arrhythmia-Classification.git 
cd Arrhythmia-Classification

Place the dataset:
Make sure your dataset (data.csv) is located in the correct folder:


1
D:\Preparation_for_Github\2. Classification of Arrhythmia\Data\data.csv
Run the main script:
bash

