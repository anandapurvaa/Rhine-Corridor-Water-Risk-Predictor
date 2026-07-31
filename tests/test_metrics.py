from modeling.metrics import classification_metrics


def test_classification_metrics_basic():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.4, 0.2]

    metrics = classification_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics