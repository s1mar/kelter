"""--help prints usage, exits 0 and does nothing for every file with a __main__ block, found by scanning the tree.
No GPU, no network."""
import ast
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"tests", ".venv", "venv", "third_party", "checkpoints", "data", "results", "build", "dist", "node_modules", ".git"}


def main_files():
    out = []
    for f in sorted(ROOT.rglob("*.py")):
        rel = f.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts) or any(p.endswith(".egg-info") for p in rel.parts):
            continue
        if '__name__ == "__main__"' in f.read_text(encoding="utf-8") and rel.name != "__init__.py":
            out.append(rel)
    return out


def command(rel):
    if rel.parts[0] == "kelter":  # run as the documented python -m kelter.x.y
        return [sys.executable, "-m", ".".join(rel.with_suffix("").parts)]
    return [sys.executable, rel.as_posix()]


def status():
    return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout


def guarded_by_ast(path):
    """Fallback for a file whose --help cannot run here (an import needs a package or weights this machine lacks): the
    __main__ block must call no_arguments or argparse itself, or a main() whose body parses arguments before anything else."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            block = ast.get_source_segment(text, node) or ""
            if "no_arguments(" in block or "parse_args" in block:
                return True
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in node.body:
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):  # past the docstring
                    return "parse_args" in (ast.get_source_segment(text, stmt) or "") or "ArgumentParser" in (ast.get_source_segment(text, stmt) or "")
    return False


def test_the_scan_finds_the_known_entry_points():
    names = {r.as_posix() for r in main_files()}
    for must in ("kelter/gpu_snapshot.py", "kelter/server.py", "kelter/eval/table.py", "scripts/download_weights.py",
                 "scripts/make_benchmark_figure.py"):
        assert must in names, must
    assert len(names) >= 25


def test_help_prints_usage_exits_0_and_does_nothing_for_every_main_file():
    files = main_files()
    before = status()

    def run(rel):
        r = subprocess.run(command(rel) + ["--help"], cwd=ROOT, capture_output=True, text=True, timeout=240)
        return rel, r
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(run, files))
    bad = []
    for rel, r in results:
        if r.returncode == 0 and "usage" in r.stdout.lower():
            continue
        if r.returncode != 0 and ("ModuleNotFoundError" in r.stderr or "ImportError" in r.stderr) and guarded_by_ast(ROOT / rel):
            continue  # cannot import here: judged by the AST check
        bad.append((rel.as_posix(), r.returncode, r.stdout[-120:], r.stderr[-200:]))
    assert not bad, bad
    assert status() == before  # nothing was written, in the tree or in tracked outputs
