from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score


def evaluate(data_response):
    label, predict_score = [], []
    for v in data_response.values():
        label.append(v["score_ground_truth"])
        predict_score.append(v["score_predict"])

    acc = accuracy_score(y_true=label, y_pred=predict_score)
    f1 = f1_score(y_true=label, y_pred=predict_score, average="weighted")
    QWK = cohen_kappa_score(label, predict_score, weights="quadratic")

    return {
        "QWK": QWK,
        "F1": f1,
        "ACC": acc
    }
