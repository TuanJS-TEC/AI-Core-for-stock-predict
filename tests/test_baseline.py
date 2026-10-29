import joblib, pytest
from nlp_module.sentiment.baseline import Baseline, class_weights, fit_prior, train

X = ["lãi tăng mạnh", "lợi nhuận tăng", "họp cổ đông", "công bố thông tin", "lỗ nặng", "thua lỗ kéo dài"] * 3
Y = ["positive", "positive", "neutral", "neutral", "negative", "negative"] * 3

def test_sqrt_weights():
    w = class_weights(["neutral"] * 8 + ["negative"] * 2, "sqrt")
    assert w["neutral"] == pytest.approx((10 / 8) ** 0.5) and w["negative"] == pytest.approx((10 / 2) ** 0.5)
    assert class_weights(Y, "balanced") == "balanced"

def test_prior_shifts_probabilities_and_keeps_sum(tmp_path):
    b = train(X, Y, "t")
    p0 = b.predict(["họp cổ đông"])[0]
    b.prior = (3.0, 1.0, 1.0)
    p1 = b.predict(["họp cổ đông"])[0]
    assert sum(p1) == pytest.approx(1.0) and p1[0] > p0[0]

def test_missing_class_stays_zero_under_nonones_prior_and_row_sums_to_one():
    keep = [i for i, l in enumerate(Y) if l != "negative"]
    b = train([X[i] for i in keep], [Y[i] for i in keep], "t")
    b.prior = (3.0, 1.0, 2.0)
    for p in b.predict(["lỗ nặng", "họp cổ đông"]):
        assert p[2] == 0.0 and sum(p) == pytest.approx(1.0)

def test_v1_two_tuple_bundle_still_loads(tmp_path):
    b = train(X, Y, "v1")
    joblib.dump((b.model, b.version), tmp_path / "old.joblib")
    old = Baseline.load(tmp_path / "old.joblib")
    assert old.prior == (1.0, 1.0, 1.0) and old.predict(["lỗ nặng"]) == b.predict(["lỗ nặng"])

def test_v2_features_train_and_roundtrip(tmp_path):
    b = train(X, Y, "v2", features="v2", weights="sqrt", C=3.0, prior=(1.5, 1.0, 1.2))
    b.save(tmp_path / "b.joblib")
    assert Baseline.load(tmp_path / "b.joblib").predict(X[:2]) == b.predict(X[:2])


def test_unknown_features_or_weights_raise():
    with pytest.raises(ValueError):
        train(X, Y, "t", features="v3")
    with pytest.raises(ValueError):
        class_weights(Y, "sqr")


def test_fit_prior_recovers_underweighted_positive():
    # positives scored just under neutral: a_pos > 1 flips them, negatives untouched
    probs = [(0.45, 0.55, 0.0)] * 5 + [(0.1, 0.8, 0.1)] * 5 + [(0.0, 0.3, 0.7)] * 2
    labels = ["positive"] * 5 + ["neutral"] * 5 + ["negative"] * 2
    assert fit_prior(probs, labels) == (1.25, 1.0, 1.0)     # smallest shift that fixes all positives (needs a_pos > 1.22)

def test_fit_prior_keeps_identity_when_already_right():
    assert fit_prior([(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)], ["positive", "neutral", "negative"]) == (1.0, 1.0, 1.0)

from nlp_module.evaluate import bootstrap_macro_f1, metrics

def test_bootstrap_ci_brackets_point_and_is_deterministic():
    labels = ["positive", "neutral", "negative", "neutral"] * 10
    probs = [(0.6, 0.3, 0.1), (0.2, 0.7, 0.1), (0.5, 0.2, 0.3), (0.4, 0.5, 0.1)] * 10
    lo, hi = bootstrap_macro_f1(labels, probs)
    assert lo <= metrics(labels, probs)["macro_f1"] <= hi
    assert (lo, hi) == bootstrap_macro_f1(labels, probs)
    perfect = [(1, 0, 0) if l == "positive" else (0, 1, 0) if l == "neutral" else (0, 0, 1) for l in labels]
    assert bootstrap_macro_f1(labels, perfect) == (1.0, 1.0)
