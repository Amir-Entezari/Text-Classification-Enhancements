from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluation(y_true, y_pred, pos_label='positive'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {"Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "confusion-matrix": conf_matrix}
