# ml_model.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model, None  # No epochs/history for KNN

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='saga', max_iter=1000)
    model.fit(X_train, y_train)
    return model, None

def train_svm_linear(X_train, y_train):
    model = LinearSVC(C=0.01)
    model.fit(X_train, y_train)
    return model, None

def train_kernelized_svm(X_train, y_train):
    model = SVC(kernel='sigmoid', C=10, gamma=0.001)
    model.fit(X_train, y_train)
    return model, None

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    model.fit(X_train, y_train)
    return model, None

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300, criterion='gini')
    model.fit(X_train, y_train)
    return model, None