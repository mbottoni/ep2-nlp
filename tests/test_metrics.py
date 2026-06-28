from bertimbau_probing.metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = [0.1, 0.4, 0.6, 0.9]
    m = regression_metrics(y, y)
    assert m["RMSE"] == 0.0
    assert m["MAE"] == 0.0
    assert m["R2"] == 1.0
    assert abs(m["Pearson"] - 1.0) < 1e-9


def test_regression_metrics_has_error_when_wrong():
    m = regression_metrics([0.0, 1.0], [1.0, 0.0])
    assert m["RMSE"] > 0
    assert m["MAE"] == 1.0


def test_classification_metrics_perfect():
    y = [0, 1, 2, 1, 0]
    m = classification_metrics(y, y, num_classes=3)
    assert m["accuracy"] == 1.0
    assert m["per_class_accuracy"] == [1.0, 1.0, 1.0]


def test_classification_metrics_confusion_shape():
    m = classification_metrics([0, 1, 2], [0, 2, 1], num_classes=3)
    assert len(m["confusion_matrix"]) == 3
    assert 0.0 <= m["accuracy"] <= 1.0
