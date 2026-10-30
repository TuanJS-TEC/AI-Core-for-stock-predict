"""Walk-forward splits: NLP's FOLDS and h-session purge, reused as-is so fusion and the text-only study are comparable."""
from nlp_module.forecast.walk_forward import FOLDS, _split


def folds(df, h):
    """Yield (name, train, val, train+val, test) per fold. df needs ret1/ret5/vol10 and y{h} (as _split drops NaN rows)."""
    for name, *bounds in FOLDS:
        yield (name, *_split(df, h, *bounds))
