from pathlib import Path
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as BuildPy
from setuptools.command.editable_wheel import editable_wheel as EditableWheel

ROOT = Path(__file__).resolve().parent


def _vendor_tiktoken_cache() -> None:
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from career_agent.agent.tiktoken_assets import populate_bundled_tiktoken_cache

    populate_bundled_tiktoken_cache()


class build_py(BuildPy):
    def run(self) -> None:
        _vendor_tiktoken_cache()
        super().run()


class editable_wheel(EditableWheel):
    def run(self) -> None:
        _vendor_tiktoken_cache()
        super().run()


setup(cmdclass={"build_py": build_py, "editable_wheel": editable_wheel})
