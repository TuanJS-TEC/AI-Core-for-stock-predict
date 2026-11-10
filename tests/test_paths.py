"""The restructured tree resolves every data path through ai_core.paths."""
from ai_core import paths
from ai_core.fusion import config as fusion_config
from ai_core.models.volcast import volcast
from ai_core.news import storage
from ai_core.prices import config as prices_config


def test_paths_resolve():
    assert (paths.ROOT / "pyproject.toml").exists()
    assert (paths.PRICES_HOME / "data" / "hose_holidays.csv").exists()
    assert prices_config.STOCK_DIR == paths.PRICES_HOME / "stock"
    assert storage.DB_PATH == paths.NEWS_HOME / "data" / "nlp.db"
    assert volcast.STOCK_DIR == paths.PRICES_HOME / "stock"
    assert fusion_config.STOCK_DIR == paths.PRICES_HOME / "stock"
