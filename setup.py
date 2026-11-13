"""Build step for the package assets.

The read-only files the runtime needs (kelter.paths.ASSET_FILES) live in the repository outside the package, where a
clone uses them. This copies them into kelter/_assets in the wheel, so each file has one source. Everything else is in
pyproject.toml.
"""
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kelter.paths import ASSET_FILES  # noqa: E402  (kelter.paths imports only the standard library and kelter)


class build_py_with_assets(build_py):
    def run(self):
        super().run()
        for src, dst in ASSET_FILES.items():
            out = Path(self.build_lib) / "kelter" / "_assets" / dst
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / src, out)


setup(cmdclass={"build_py": build_py_with_assets})
