from ai_core.models.mmf import config as C
from ai_core.prices.splits import folds_from_config

def test_fold_nlp_maps_every_fold_and_labels_precede_train_end():
    assert {f.name for f in folds_from_config()} == set(C.FOLD_NLP) - {"F"}
    assert C.FOLD_NLP == {"A": "phobert-foldB", "B": "phobert-foldB", "C": "phobert-foldC", "F": "phobert-phobert-v1"}

def test_final_fold_ends_where_dev_ends():
    assert C.FINAL_FOLD == ("F", "2026-05-04", "2026-07-30", "2026-10-30")
    assert C.HOLDOUT_AFTER == "2026-10-30"

def test_arms_cover_every_ablation():
    assert C.ARMS["fusion_concat"] == (("price", "fund", "text"), "concat")
    assert C.ARMS["fusion_attention"] == (("price", "fund", "text"), "attention")
    assert C.ARMS["no_price"][0] == ("fund", "text")
    assert C.ARMS["no_fund"][0] == ("price", "text")
    assert C.ARMS["no_text"][0] == ("price", "fund")
    assert {"dl_price", "dl_fund", "dl_text"} <= set(C.ARMS)
