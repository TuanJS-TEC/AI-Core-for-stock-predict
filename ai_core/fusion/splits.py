"""Walk-forward splits: NLP's FOLDS (fold C's end frozen at TEST_END) and its purge, so fusion and the text-only study stay comparable."""
from . import config as C
from ai_core.models.volcast import walk_forward
from ai_core.models.volcast.walk_forward import _split

FOLDS = [(n, a, b, c, C.TEST_END if n == "C" else e) for n, a, b, c, e in walk_forward.FOLDS]


def folds(df, h, purge=None, folds=FOLDS):
    """Yield (name, train, val, train+val, test) per fold. df needs ret1/ret5/vol10 and y{h} (as _split drops NaN rows).
    purge=None means h sessions."""
    for name, *bounds in folds:
        yield (name, *_split(df, h, *bounds, purge=purge))
