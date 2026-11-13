"""One-way dependencies: prices and news know nothing of each other, models read both, fusion sits on top."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "ai_core"
LAYER = {"prices": 0, "news": 0, "models": 1, "fusion": 2}


def imports_of(path: Path) -> set[str]:
    """Absolute module names imported anywhere in the file (also inside functions); relative imports are resolved."""
    pkg = ".".join(["ai_core", *path.relative_to(PKG).parent.parts])
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                parts = pkg.split(".")[: len(pkg.split(".")) - (n.level - 1)]
                base = ".".join([*parts, *([n.module] if n.module else [])])
            out.add(base)
            out |= {f"{base}.{a.name}" for a in n.names}
    return out


def layer_of(mod: str):
    p = mod.split(".")
    return p[1] if p[0] == "ai_core" and len(p) > 1 and p[1] in LAYER else None


def violations(forbid):
    bad = []
    for f in sorted(PKG.rglob("*.py")):
        mine = f.relative_to(PKG).parts[0]
        for m in imports_of(f):
            if mine in LAYER and layer_of(m) and forbid(mine, layer_of(m)):
                bad.append(f"{f.relative_to(PKG)} imports {m}")
    return bad


def test_prices_and_news_do_not_import_each_other_or_upward():
    assert not violations(lambda a, b: a in ("prices", "news") and b != a)


def test_models_do_not_import_fusion():
    assert not violations(lambda a, b: a == "models" and b == "fusion")


def test_finhay_is_http_only():
    bad = {m for m in imports_of(PKG / "fusion" / "finhay.py")
           if layer_of(m) in ("news", "models") or m.split(".")[0] in ("torch", "transformers")}
    assert not bad, bad


def test_package_inits_stay_empty():
    for d in ("", "prices", "fusion"):
        src = (PKG / d / "__init__.py").read_text(encoding="utf-8")
        assert not ast.parse(src).body, f"ai_core/{d}/__init__.py must stay empty (cron imports are minimal)"


def test_no_sys_path_mutation():
    for f in PKG.rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            assert not (isinstance(n, ast.Attribute) and n.attr == "path" and isinstance(n.value, ast.Name)
                        and n.value.id == "sys"), f"{f.relative_to(PKG)} touches sys.path"
