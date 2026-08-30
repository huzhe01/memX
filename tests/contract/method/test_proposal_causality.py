from __future__ import annotations

import ast
import inspect

import ratemem.method.proposal as proposal_module
from ratemem.method.proposal import CausalCandidateProposer


def test_proposal_api_has_no_future_trace_or_query_images() -> None:
    parameters = set(inspect.signature(CausalCandidateProposer.propose).parameters)
    assert parameters == {
        "self",
        "state",
        "handle",
        "current_target_code",
        "event_index",
    }


def test_proposal_source_cannot_import_evaluator_or_final_trace() -> None:
    tree = ast.parse(inspect.getsource(proposal_module))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("ratemem.evaluation.final_trace") for name in imported)
    assert not any("evaluator" in name for name in imported)
