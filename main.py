# main.py
import pandas as pd
import os
from data_loader import load_and_preprocess_data
from ml_model import train_knn, train_logistic_regression, train_svm_linear, train_kernelized_svm, train_decision_tree, train_random_forest
from evaluation import evaluate_and_save
import config

def save_results_to_file(results, filename=config.RESULT_FILE):
    with open(filename, 'w') as f:
        for res in results:
            f.write(f"Model: {res['Model']}\n")
            f.write(f"Train Accuracy: {res['Train Accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {res['Test Accuracy']:.4f}\n")
            f.write(f"Train Recall: {res['Train Recall']:.4f}\n")
            f.write(f"Test Recall: {res['Test Recall']:.4f}\n")
            f.write(f"Train Precision: {res['Train Precision']:.4f}\n")
            f.write(f"Test Precision: {res['Test Precision']:.4f}\n")
            f.write("-" * 50 + "\n")

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config.DATA_PATH)

    results = []

    models = [
        ('KNN', train_knn),
        ('Logistic Regression', train_logistic_regression),
        ('SVM Linear', train_svm_linear),
        ('Kernelized SVM', train_kernelized_svm),
        ('Decision Tree', train_decision_tree),
        ('Random Forest', train_random_forest),
    ]

    for name, trainer in models:
        print(f"Training {name}...")
        model, _ = trainer(X_train, y_train)
        result = evaluate_and_save(model, X_train, y_train, X_test, y_test, name, config.PLOT_SAVE_DIR)
        results.append(result)

    # Save results
    save_results_to_file(results)
    print("All results saved successfully.")

if __name__ == "__main__":
    main()