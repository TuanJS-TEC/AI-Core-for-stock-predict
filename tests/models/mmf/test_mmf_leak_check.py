from ai_core.models.mmf import experiment as X, leak_check as L
from .test_mmf_experiment import FOLD, SYMS, fake_frame


def test_leak_check_scores_true_labels_then_each_permutation(tmp_path, monkeypatch):
    b = X.bundle_from_frame(fake_frame(), 5, FOLD, "phobert-foldB", "tinydataset", SYMS)
    seeds, real = [], X.shuffle_train_labels
    monkeypatch.setattr(X, "shuffle_train_labels", lambda b, s: (seeds.append(s), real(b, s))[1])
    d = L.run(None, (5,), 2, "cpu", build=lambda p, h, f: b, max_epochs=1)
    assert list(d["fold"]) == ["A", "B", "C"] and (d["h"] == 5).all() and seeds == [0, 1] * 3
    auc = d[L.HEADER[2:7]].to_numpy()
    assert ((auc >= 0) & (auc <= 1)).all() and (d[L.HEADER[5]] <= d[L.HEADER[3]]).all() and (d[L.HEADER[3]] <= d[L.HEADER[6]]).all()
    text = L.write(d, tmp_path / "x/leak.md").read_text()
    assert text.count("\n") == 2 + 3 and "AUC nhãn xáo trộn: trung bình" in text
