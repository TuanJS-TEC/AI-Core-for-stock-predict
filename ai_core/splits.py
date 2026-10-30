"""Walk-forward splits: NLP's FOLDS (fold C's end frozen at TEST_END) and its purge, so fusion and the text-only study stay comparable."""
from nlp_module.forecast.walk_forward import _split

from . import config as C


def folds(df, h, purge=None, folds=C.FOLDS):
    """Yield (name, train, val, train+val, test) per fold. df needs ret1/ret5/vol10 and y{h} (as _split drops NaN rows).
    purge=None means h sessions."""
    for name, *bounds in folds:
        yield (name, *_split(df, h, *bounds, purge=purge))
