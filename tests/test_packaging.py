"""Packaging: where files live in a checkout and in an installed wheel, the kelter command,
and the download from a hand-placed archive. No GPU, no network."""
import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path

import pytest

from kelter import __version__, paths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_home(monkeypatch):
    monkeypatch.delenv("KELTER_HOME", raising=False)


@pytest.fixture
def installed(tmp_path, monkeypatch, no_home):
    """Installed mode: no checkout, package assets copied as setup.py copies them, KELTER_HOME a temp folder."""
    assets = tmp_path / "assets"
    for src, dst in paths.ASSET_FILES.items():
        (assets / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, assets / dst)
    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    monkeypatch.setattr(paths, "ASSETS", assets)
    home = tmp_path / "home"
    monkeypatch.setenv("KELTER_HOME", str(home))
    return home


def test_checkout_mode_keeps_todays_paths(no_home):
    assert paths.checkout_root() == ROOT and paths.is_checkout()
    assert paths.home() == ROOT
    assert paths.asset("policy_table.json") == ROOT / "policy_table.json"
    assert paths.asset("results/ensemble/E1.json") == ROOT / "results" / "ensemble" / "E1.json"
    assert paths.under_home("checkpoints/r2") == ROOT / "checkpoints" / "r2"
    assert paths.command("serve") == "python -m kelter.server"
    from kelter.engine import load_policy
    assert load_policy()["_sha256"] == (ROOT / "policy_table.sha256").read_text(encoding="utf-8").strip()


def test_kelter_home_overrides(tmp_path, monkeypatch, no_home):
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    assert paths.home() == tmp_path
    assert paths.asset("policy_table.json") == ROOT / "policy_table.json"  # read-only files stay in the checkout


def test_installed_mode_uses_assets_and_user_folder(tmp_path, monkeypatch, no_home):
    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    assert paths.home() == paths.user_home()
    assert paths.asset("weights/MANIFEST.json") == paths.ASSETS / "weights" / "MANIFEST.json"
    assert paths.asset("results/ensemble/E3.json") == paths.ASSETS / "ensemble" / "E3.json"
    assert paths.command("serve") == "kelter serve"
    assert paths.user_home({"LOCALAPPDATA": "X:/la"}, "win32") == Path("X:/la") / "kelter"
    assert paths.user_home({"XDG_DATA_HOME": "/xdg"}, "linux") == Path("/xdg") / "kelter"
    assert paths.user_home({}, "linux") == Path.home() / ".local" / "share" / "kelter"
    assert paths.user_home({}, "darwin") == Path.home() / "Library" / "Application Support" / "kelter"


def test_asset_sources_exist_and_policy_loads_installed(installed):
    for src in paths.ASSET_FILES:
        assert (ROOT / src).is_file(), src
    from kelter.engine import load_policy
    assert load_policy()["_sha256"] == (ROOT / "policy_table.sha256").read_text(encoding="utf-8").strip()


def test_checkout_only_line(installed, capsys):
    with pytest.raises(SystemExit) as e:
        paths.require_checkout("training (python -m kelter.train)")
    assert e.value.code == 2
    err = capsys.readouterr().err.strip()
    assert "\n" not in err and "clone of the repository" in err
    from kelter import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["train"])
    assert e.value.code == 2


def test_every_script_module_is_guarded_or_runtime():
    """Run as python -m from a wheel, every checkout-only module prints the one line instead of failing on paths."""
    runtime = {"cli", "download", "examples", "gpu_snapshot", "server", "__main__"}
    for p in sorted((ROOT / "kelter").rglob("*.py")):
        s = p.read_text(encoding="utf-8")
        rel = p.with_suffix("").relative_to(ROOT / "kelter").as_posix()
        if '__name__ == "__main__"' in s and rel not in runtime:
            tail = s[s.index('__name__ == "__main__"'):]
            assert "paths.require_checkout(" in tail.split("\n", 3)[2], rel


def test_cli_version(capsys):
    from kelter import cli
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"kelter {__version__}"


def test_cli_status_without_weights(installed, monkeypatch, capsys):
    from kelter import cli
    monkeypatch.setattr(cli, "_gpu", lambda: "no CUDA GPU found")
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "installed package" in out and f"Folder: {installed}" in out
    assert "default not downloaded" in out and "lower-latency not downloaded" in out
    assert "Next: kelter download --variant lower-latency" in out
    monkeypatch.setattr(cli, "_gpu", lambda: "GPU X, 11.0 of 12.0 GB free")
    cli.main([])
    assert "Next: kelter download\n" in capsys.readouterr().out
    for n in ("r2", "Q1"):
        (installed / "checkpoints" / n).mkdir(parents=True)
        (installed / "checkpoints" / n / "config.json").write_text("{}")
    cli.main([])
    assert "default present" in (out := capsys.readouterr().out) and "Next: kelter serve, then" in out


def test_serve_without_weights_one_line(installed):
    from kelter import server
    with pytest.raises(server.MissingWeights) as e:
        server.main([])
    msg = str(e.value)
    assert "\n" not in msg and "weights not found" in msg and "kelter download" in msg
    assert str(installed / "checkpoints") in msg


def test_witness_and_ensemble_paths_under_home(installed):
    """Q1's witness_ckpt resolves under the writable folder, as the variant checkpoints do."""
    from kelter import server
    ck = installed / "checkpoints" / "Q1"
    (ck / "adapter").mkdir(parents=True)
    (ck / "config.json").write_text(json.dumps({"llm": {}, "witness_ckpt": "checkpoints/r2"}))
    with pytest.raises(server.MissingWeights) as e:
        server.check_weights(server.default_ckpt_path("checkpoints/Q1"))
    assert "r2" in str(e.value) and "kelter download" in str(e.value)
    (installed / "checkpoints" / "r2").mkdir()
    (installed / "checkpoints" / "r2" / "config.json").write_text("{}")
    server.check_weights(server.default_ckpt_path("checkpoints/Q1"))


def _tar(name):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        data = b"{}"
        info = tarfile.TarInfo(f"{name}/config.json")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_uses_hand_placed_archive(installed, monkeypatch, capsys):
    from kelter import download
    blob = _tar("r2")
    manifest = {"tag": "v0", "repo": "o/r", "variants": {"default": ["r2"]},
                "assets": {"r2": {"file": "r2.tar", "sha256": hashlib.sha256(blob).hexdigest(), "size_mb": 0}}}
    (paths.ASSETS / "weights" / "MANIFEST.json").write_text(json.dumps(manifest))
    arch = installed / "weights" / "archives"
    arch.mkdir(parents=True)
    (arch / "r2.tar").write_bytes(blob)

    def no_network(*a, **k):
        raise AssertionError("downloaded although the archive was in place")
    monkeypatch.setattr(download, "fetch", no_network)
    assert download.main([]) == 0
    assert (installed / "checkpoints" / "r2" / "config.json").exists() and not (arch / "r2.tar").exists()
    assert str(installed / "checkpoints") in capsys.readouterr().out
    assert download.main([]) == 0  # already present: skipped


def test_pyproject_declares_package():
    s = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "(.+)"', s, re.M).group(1) == __version__
    assert 'build-backend = "setuptools.build_meta"' in s and 'kelter = "kelter.cli:main"' in s
    assert 'license-files = ["LICENSE", "WEIGHTS-LICENSE.md"]' in s
    req = {re.split(r"[=\[ ]", l)[0]: l.split("==")[1].split()[0] for l in
           (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if "==" in l and not l.startswith("#")}
    for dep in re.findall(r'"([A-Za-z0-9_.\-]+)(?:\[[a-z]+\])?>=([0-9.]+)"', s):
        if dep[0] in ("torch", "setuptools"):  # torch: from the PyTorch index; setuptools: the build backend
            continue
        assert req.get(dep[0]) == dep[1], dep
