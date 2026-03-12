from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier

# simple linear model to act as baseline
def baseline(x_train, y_train, x_test):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)
    return model

# 1 Decision Tree -> Random forest, xgboost
def tree(x_train, y_train):
    max_depth = [5, 10, 15]
    min_sample_split = [100, 500]
    min_samples_leaf = [10, 50, 100]

    dt = DecisionTreeClassifier()
    
    # 1.1 soft voting

# 1.2 bagging
    bagging_models = {
        "10 class 20%:", BaggingClassifier(n_estimators= 10, max_samples = 0.2),
        BaggingClassifier(n_estimators= 10, max_samples = 0.2),
        BaggingClassifier(n_estimators= 10, max_samples = 0.2, bootstrap=True),
    }

#1.3 grad boost