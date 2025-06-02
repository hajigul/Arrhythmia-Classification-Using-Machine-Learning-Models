# evaluation.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    confusion_matrix, classification_report
)

def evaluate_and_save(model, X_train, y_train, X_test, y_test, model_name, save_path):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_rec = recall_score(y_test, y_pred_test, average='weighted')
    test_rec = recall_score(y_test, y_pred_test, average='weighted')
    train_prec = precision_score(y_test, y_pred_test, average='weighted')
    test_prec = precision_score(y_test, y_pred_test, average='weighted')

    print(f"{model_name}:")
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Train Recall: {train_rec:.4f}, Test Recall: {test_rec:.4f}")
    print(f"Train Precision: {train_prec:.4f}, Test Precision: {test_prec:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    plt.close()

    # Classification report
    report = classification_report(y_test, y_pred_test, output_dict=False)
    print(report)

    return {
        'Model': model_name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Train Recall': train_rec,
        'Test Recall': test_rec,
        'Train Precision': train_prec,
        'Test Precision': test_prec
    }