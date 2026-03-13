from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier)

RAND_STATE = 67

# simple linear model to act as baseline
def baseline(x_train, y_train, class_weight="balanced"):
    model = LogisticRegression(
      max_iter=1000,
      class_weight=class_weight,
      random_state=RAND_STATE
      )
    model.fit(x_train, y_train)
    return model

# 1 SVM
def svm_model(x_train, y_train, C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"):
  model = SVC(
    C=C,
    kernel=kernel,
    gamma=gamma,
    class_weight=class_weight,
    probability=True,
    random_state=RAND_STATE
  )

  model.fit(x_train, y_train)
  
  return model

# 2 Decision Tree -> Random forest, xgboost
def decision_tree(x_train, y_train, max_depth=10, min_samples_split=100, min_samples_leaf=10, class_weight="balanced"):
  model = DecisionTreeClassifier(
      max_depth=max_depth,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      class_weight=class_weight,
      random_state=RAND_STATE
  )

  model.fit(x_train, y_train)
  
  return model

# 3 Bagging Tree
def bagging_tree(x_train, y_train, max_depth=10, min_samples_split=100,
                min_samples_leaf=10, n_estimators=20, max_samples=0.8, class_weight="balanced"):
    base_tree = DecisionTreeClassifier(
      max_depth=max_depth,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      class_weight=class_weight,
      random_state=RAND_STATE
    )

    model = BaggingClassifier(
      estimator=base_tree,
      n_estimators=n_estimators,
      max_samples=max_samples,
      bootstrap=True,
      random_state=RAND_STATE
    )

    model.fit(x_train, y_train)

    return model
    
# 4 Random Forest
def random_forest(x_train, y_train, n_estimators=200, max_depth=None,
                min_samples_split=100,min_samples_leaf=10, class_weight="balanced"):
  model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    class_weight=class_weight,
    random_state=RAND_STATE,
    n_jobs=-1
    )
  
  model.fit(x_train, y_train)

  return model

# 5 Grad Boost
def gradient_boost(x_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
  model = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=RAND_STATE
  )
  
  model.fit(x_train, y_train)

  return model

# # 6 Soft Voting
def soft_voting(x_train, y_train, tree_max_depth=10, class_weight="balanced"):
  lr = LogisticRegression(
    max_iter=2000,
    class_weight=class_weight,
    random_state=RAND_STATE
  )

  dt = DecisionTreeClassifier(
    max_depth=tree_max_depth,
    min_samples_split=100,
    min_samples_leaf=10,
    class_weight=class_weight,
    random_state=RAND_STATE
  )

  svm = SVC(
    probability=True,
    class_weight=class_weight,
    random_state=RAND_STATE
  )

  model = VotingClassifier(
    estimators=[
      ("lr", lr),
      ("dt", dt),
      ("svm", svm),
    ],
    voting="soft"
  )

  model.fit(x_train, y_train)

  return model