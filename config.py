# config.py
import os

DATA_PATH = r"D:\Preparation_for_Github\2. Classification of Arrhythmia\Data\data.csv"
PLOT_SAVE_DIR = r"D:\Preparation_for_Github\2. Classification of Arrhythmia\Data\plots"
RESULT_FILE = r"D:\Preparation_for_Github\2. Classification of Arrhythmia\Data\results\model_results.txt"

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)