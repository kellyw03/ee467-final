from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  confusion_matrix,
  classification_report
)

def evaluate_model_base(model, x_test, y_test):
  y_pred = model.predict(x_test)

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, average="macro")
  rec = recall_score(y_test, y_pred, average="macro")
  f1 = f1_score(y_test, y_pred, average="macro")

  print(f"Accuracy: {acc:.2f}")
  print(f"Precision: {prec:.2f}")
  print(f"Recall: {rec:.2f}")
  print(f"F1: {f1:.2f}")
  print("\nConfusion Matrix")
  print(confusion_matrix(y_test, y_pred))
  print("\nClassification Report")
  print(classification_report(y_test, y_pred))

  return acc, prec, rec, f1