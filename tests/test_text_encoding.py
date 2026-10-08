from __future__ import annotations

import ast
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "src"
_PATH_TEXT_METHODS = {"read_text", "write_text"}


def _text_calls_without_encoding(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if any(keyword.arg == "encoding" for keyword in node.keywords):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _PATH_TEXT_METHODS:
            lines.append(node.lineno)
            continue
        is_open = (
            isinstance(function, ast.Name) and function.id == "open"
            or isinstance(function, ast.Attribute) and function.attr == "open"
            and not (isinstance(function.value, ast.Name) and function.value.id in {"os", "sqlite3"})
        )
        if not is_open:
            continue
        mode = next((k.value for k in node.keywords if k.arg == "mode"), None)
        if mode is None:
            # Built-in open takes mode second; Path.open takes it first.
            position = 1 if isinstance(function, ast.Name) else 0
            mode = node.args[position] if len(node.args) > position else None
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value:
            continue
        lines.append(node.lineno)
    return lines


def test_source_never_relies_on_the_platform_text_encoding() -> None:
    # Windows defaults to the locale code page (for example GBK), macOS to
    # UTF-8; text files must name their encoding to behave the same on both.
    offenders = [
        f"{path.relative_to(_SOURCE.parent)}:{line}"
        for path in sorted(_SOURCE.rglob("*.py"))
        for line in _text_calls_without_encoding(
            ast.parse(path.read_text(encoding="utf-8"))
        )
    ]

    assert offenders == []
