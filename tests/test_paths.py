"""The tree resolves every data path through ai_core.paths."""
from ai_core import paths
from ai_core.fusion import config as fusion_config
from ai_core.models.volcast import volcast
from ai_core.news import entities, storage
from ai_core.prices import config as prices_config


def test_paths_resolve():
    assert (paths.ROOT / "pyproject.toml").exists()
    assert (paths.CONFIG / "hose_holidays.csv").exists()
    assert (paths.CONFIG / "companies.csv").exists()
    assert prices_config.STOCK_DIR == paths.PRICES == paths.DATA / "prices"
    assert prices_config.HOLIDAYS_FILE == paths.CONFIG / "hose_holidays.csv"
    assert storage.DB_PATH == paths.NEWS / "nlp.db" and paths.NEWS == paths.DATA / "news"
    assert entities.COMPANIES_CSV == paths.CONFIG / "companies.csv"
    assert volcast.STOCK_DIR == paths.PRICES
    assert volcast.BUNDLE.startswith(str(paths.NEWS)) and volcast.REPORT.startswith(str(paths.NEWS))
    assert fusion_config.STOCK_DIR == paths.PRICES


def test_no_legacy_home_dirs():
    assert not (paths.ROOT / "NLP_Module").exists() and not (paths.ROOT / "Pipeline_V1.2").exists()
