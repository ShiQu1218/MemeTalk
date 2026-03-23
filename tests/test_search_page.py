import ast
from pathlib import Path


def test_search_page_does_not_reference_undefined_container_name() -> None:
    page_path = Path(__file__).resolve().parents[1] / "pages" / "3_🔍_Search.py"
    module = ast.parse(page_path.read_text(encoding="utf-8"))

    loaded_container_names = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and node.id == "container" and isinstance(node.ctx, ast.Load)
    ]

    assert loaded_container_names == []
