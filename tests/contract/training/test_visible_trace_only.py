from __future__ import annotations

import ast
from pathlib import Path


def test_training_tree_has_no_final_trace_import_or_hidden_evaluation_literal() -> None:
    for path in Path("src/ratemem/training").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ratemem.evaluation.final_trace"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "final-test-envelope" not in node.value
                assert node.value != "final_evaluation"
