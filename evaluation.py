import matplotlib.pyplot as plt 

from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  confusion_matrix,
  classification_report, 
  roc_curve
)

def evaluate_model_base(model, x_test, y_test):
  y_pred = model.predict(x_test)

  y_test_bin = (y_test != "benign").astype(int)
  y_scores = model.predict_proba(x_test)[:,1]
  fpr, tpr, thresholds = roc_curve(y_test_bin, y_scores)

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
  rec = recall_score(y_test, y_pred, average="macro")
  f1 = f1_score(y_test, y_pred, average="macro")
  cm = confusion_matrix(y_test, y_pred)

  print(f"Accuracy: {acc:.2f}")
  print(f"Precision: {prec:.2f}")
  print(f"Recall: {rec:.2f}")
  print(f"F1: {f1:.2f}")
  print("\nConfusion Matrix")
  print(confusion_matrix(y_test, y_pred))
  print("\nClassification Report")
  print(classification_report(y_test, y_pred))

  return acc, prec, rec, f1, fpr, tpr

import numpy as np

def plot_recall(models):
  fprs = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

  def recall_at_fpr(fpr, tpr, target_fpr):
      fpr = np.array(fpr)
      tpr = np.array(tpr)
      valid = fpr <= target_fpr
      if not valid.any():
          return 0.0
      return tpr[valid].max()

  recall = {model: [] for model in models}
  for model, (fpr, tpr) in models.items():
     for targ_fpr in fprs:
        recall[model].append(recall_at_fpr(fpr, tpr, targ_fpr))


  for model_name, recalls in recall.items():
      plt.plot(fprs, recalls, marker='o', label=model_name)

  plt.xlabel("False Positive Rate")
  plt.ylabel("Recall (TPR)")
  plt.title("Recall at Fixed FPR for Multiple Models")
  plt.grid(True)
  plt.legend()
  plt.show()

