from __future__ import annotations

import json
import operator
import subprocess
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from ratemem.artifacts.schema import AttemptManifest

_REPOSITORY = Path(__file__).resolve().parents[2]
_SYNTHETIC_SUFFIX = "SYNTHETIC_TEST_VALUE_123456789"


class _ManifestEnvelope(BaseModel):
    manifest: AttemptManifest


def _synthetic_credential(prefix: str = "ak-") -> str:
    return prefix + _SYNTHETIC_SUFFIX


def _valid_input() -> dict[str, Any]:
    return {
        "run_id": "cpu-smoke",
        "git_revision": "f" * 40,
        "config_hash": "a" * 64,
        "status": "passed",
        "notes": "",
    }


def _exception_surfaces(error: BaseException) -> tuple[str, ...]:
    surfaces = [
        str(error),
        repr(error),
        operator.mod("%s", error),
        operator.mod("%r", error),
        "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        repr(error.args),
        repr(error.__dict__),
        repr(error.__cause__),
        repr(error.__context__),
    ]
    if isinstance(error, ValidationError):
        surfaces.extend((repr(error.errors()), error.json()))
    return tuple(surfaces)


def _assert_not_leaked(error: BaseException, credential: str) -> None:
    assert all(credential not in surface for surface in _exception_surfaces(error))


def _assert_credential_rejected(
    call: Callable[[], object], credential: str
) -> None:
    with pytest.raises(ValueError, match="credential-shaped") as raised:
        call()
    _assert_not_leaked(raised.value, credential)


def _assert_serialization_rejected(
    call: Callable[[], object], credential: str
) -> None:
    try:
        call()
    except Exception as error:
        assert "credential-shaped" in str(error)
        _assert_not_leaked(error, credential)
    else:
        raise AssertionError("credential-bearing instance was serialized")


def _forged_manifest(**updates: object) -> AttemptManifest:
    values = _valid_input()
    values.update(updates)
    construct = BaseModel.model_construct.__func__
    return construct(AttemptManifest, _fields_set=set(values), **values)


def _bare_forged_manifest(**updates: object) -> AttemptManifest:
    values = _valid_input()
    values.update(updates)
    forged = object.__new__(AttemptManifest)
    object.__setattr__(forged, "__dict__", values)
    return forged


def _call_validation_entrypoint(
    entrypoint: str, values: dict[str, Any]
) -> AttemptManifest:
    if entrypoint == "constructor":
        return AttemptManifest(**values)
    if entrypoint == "model_validate":
        return AttemptManifest.model_validate(values)
    serialized = json.dumps(values)
    if entrypoint == "model_validate_json_str":
        return AttemptManifest.model_validate_json(serialized)
    return AttemptManifest.model_validate_json(serialized.encode("utf-8"))


def _assign_attribute(
    manifest: AttemptManifest, name: str, value: object
) -> None:
    setattr(manifest, name, value)


def _delete_attribute(manifest: AttemptManifest, name: str) -> None:
    delattr(manifest, name)


def test_manifest_rejects_secret_shaped_values() -> None:
    credential = _synthetic_credential()
    _assert_credential_rejected(
        lambda: AttemptManifest(
            run_id="cpu-smoke",
            git_revision="f" * 40,
            config_hash="a" * 64,
            status="passed",
            notes="token " + credential,
        ),
        credential,
    )


@pytest.mark.parametrize("field", ["run_id", "git_revision", "config_hash", "notes"])
@pytest.mark.parametrize("prefix", ["ak-", "as-"])
def test_manifest_rejects_every_credential_shape_without_echoing_input(
    field: str, prefix: str
) -> None:
    credential = _synthetic_credential(prefix)
    values = _valid_input()
    values[field] = credential

    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate(values), credential
    )


@pytest.mark.parametrize(
    "entrypoint",
    ["constructor", "model_validate", "model_validate_json_str", "model_validate_json_bytes"],
)
@pytest.mark.parametrize(
    "scenario", ["free-text", "invalid-status", "extra-value", "extra-key"]
)
def test_all_public_validation_entrypoints_redact_rejected_inputs(
    entrypoint: str, scenario: str
) -> None:
    credential = _synthetic_credential("as-")
    values = _valid_input()
    if scenario == "free-text":
        values["notes"] = credential
    elif scenario == "invalid-status":
        values["status"] = credential
    elif scenario == "extra-value":
        values["unexpected"] = credential
    else:
        values[credential] = "safe"

    _assert_credential_rejected(
        lambda: _call_validation_entrypoint(entrypoint, values), credential
    )


@pytest.mark.parametrize("location", ["value", "key"])
def test_json_escaped_credentials_are_preflighted(location: str) -> None:
    credential = _synthetic_credential()
    escaped = "\\u0061\\u006b-" + _SYNTHETIC_SUFFIX
    raw = (
        '{"run_id":"cpu-smoke","git_revision":"'
        + "f" * 40
        + '","config_hash":"'
        + "a" * 64
        + '","status":"passed","notes":""'
    )
    if location == "value":
        raw += ',"unexpected":{"nested":["' + escaped + '"]}}'
    else:
        raw += ',"unexpected":{"' + escaped + '":"safe"}}'

    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate_json(raw), credential
    )


@pytest.mark.parametrize("location", ["value", "key"])
@pytest.mark.parametrize("payload_type", ["str", "bytes", "bytearray"])
def test_public_json_boundary_preflights_malformed_raw_input(
    location: str, payload_type: str
) -> None:
    credential = _synthetic_credential("as-")
    if location == "value":
        raw = '{"notes":"' + credential
    else:
        raw = '{"' + credential + '":'
    payload: str | bytes | bytearray
    if payload_type == "bytes":
        payload = raw.encode("ascii")
    elif payload_type == "bytearray":
        payload = bytearray(raw, "ascii")
    else:
        payload = raw

    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate_json(payload), credential
    )


def test_public_json_boundary_delegates_safe_malformed_input() -> None:
    with pytest.raises(ValidationError, match="Invalid JSON"):
        AttemptManifest.model_validate_json('{"notes":')


@pytest.mark.parametrize(
    "scenario",
    [
        "invalid-status",
        "extra-value",
        "extra-key",
        "nested-value",
        "nested-key",
        "bytes-value",
        "bytes-key",
    ],
)
def test_recursive_preflight_runs_before_ordinary_validation_errors(
    scenario: str,
) -> None:
    prefix = "as-" if scenario.startswith("bytes") else "ak-"
    credential = _synthetic_credential(prefix)
    values = _valid_input()
    if scenario == "invalid-status":
        values["status"] = credential
    elif scenario == "extra-value":
        values["unexpected"] = credential
    elif scenario == "extra-key":
        values[credential] = "safe"
    elif scenario == "nested-value":
        values["unexpected"] = {"rows": [{"safe": credential}]}
    elif scenario == "nested-key":
        values["unexpected"] = {"rows": [{credential: "safe"}]}
    elif scenario == "bytes-value":
        values["unexpected"] = {b"safe": credential.encode("ascii")}
    else:
        values["unexpected"] = {credential.encode("ascii"): b"safe"}

    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate(values), credential
    )


def test_type_adapter_and_nested_validation_share_the_safe_preflight() -> None:
    credential = _synthetic_credential("as-")
    values = _valid_input()
    values["unexpected"] = {"nested": [credential]}
    serialized = json.dumps(values)
    adapter = TypeAdapter(AttemptManifest)

    calls: tuple[Callable[[], object], ...] = (
        lambda: adapter.validate_python(values),
        lambda: adapter.validate_json(serialized),
        lambda: _ManifestEnvelope.model_validate({"manifest": values}),
        lambda: TypeAdapter(_ManifestEnvelope).validate_python(
            {"manifest": values}
        ),
    )
    for call in calls:
        _assert_credential_rejected(call, credential)


def test_manifest_forbids_extra_fields() -> None:
    values = _valid_input()
    values["unexpected"] = "value"
    with pytest.raises(ValidationError):
        AttemptManifest.model_validate(values)


def test_manifest_is_frozen_and_revalidates_instances() -> None:
    manifest = AttemptManifest.model_validate(_valid_input())
    with pytest.raises(ValidationError):
        manifest.notes = "changed"

    credential = _synthetic_credential()
    forged = _forged_manifest(notes=credential)
    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate(forged), credential
    )


def test_frozen_mutation_errors_do_not_retain_credential_input() -> None:
    manifest = AttemptManifest.model_validate(_valid_input())
    credential = _synthetic_credential("as-")

    _assert_credential_rejected(
        lambda: _assign_attribute(manifest, "notes", credential), credential
    )
    _assert_credential_rejected(
        lambda: _assign_attribute(manifest, credential, "safe"), credential
    )
    _assert_credential_rejected(
        lambda: _delete_attribute(manifest, credential), credential
    )


def test_model_copy_validates_updates_and_unchecked_copies() -> None:
    manifest = AttemptManifest.model_validate(_valid_input())
    safe_copy = manifest.model_copy(update={"notes": "safe update"})
    assert safe_copy.notes == "safe update"

    credential = _synthetic_credential("as-")
    _assert_credential_rejected(
        lambda: manifest.model_copy(update={"notes": credential}), credential
    )
    unchecked = BaseModel.model_copy(manifest, update={"notes": credential})
    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate(unchecked), credential
    )


def test_model_copy_preserves_standard_fields_set_semantics() -> None:
    manifest = AttemptManifest(
        run_id="cpu-smoke",
        git_revision="f" * 40,
        config_hash="a" * 64,
        status="passed",
    )
    expected_fields = {
        "run_id",
        "git_revision",
        "config_hash",
        "status",
    }
    expected_dump = manifest.model_dump(exclude_unset=True)

    for copied in (
        manifest.model_copy(),
        manifest.model_copy(deep=True),
        manifest.model_copy(update={}),
    ):
        assert copied.model_fields_set == expected_fields
        assert copied.model_dump(exclude_unset=True) == expected_dump

    updated = manifest.model_copy(update={"notes": "safe update"})
    assert updated.model_fields_set == expected_fields | {"notes"}
    assert updated.model_dump(exclude_unset=True) == {
        **expected_dump,
        "notes": "safe update",
    }


def test_model_construct_is_disabled_without_echoing_arguments() -> None:
    credential = _synthetic_credential()
    values = _valid_input()
    values["notes"] = credential
    with pytest.raises(ValueError, match="model_construct is disabled") as raised:
        AttemptManifest.model_construct(**values)
    _assert_not_leaked(raised.value, credential)


@pytest.mark.parametrize("location", ["value", "key"])
def test_outbound_serializers_reject_forged_credentials(location: str) -> None:
    credential = _synthetic_credential("as-")
    notes: object
    if location == "value":
        notes = {"nested": [credential]}
    else:
        notes = {credential: "safe"}
    forged = _forged_manifest(notes=notes)
    adapter = TypeAdapter(AttemptManifest)
    envelope = BaseModel.model_construct.__func__(
        _ManifestEnvelope,
        _fields_set={"manifest"},
        manifest=forged,
    )
    envelope_adapter = TypeAdapter(_ManifestEnvelope)

    calls: tuple[Callable[[], object], ...] = (
        forged.model_dump,
        forged.model_dump_json,
        lambda: adapter.dump_python(forged),
        lambda: adapter.dump_json(forged),
        envelope.model_dump,
        envelope.model_dump_json,
        lambda: envelope_adapter.dump_python(envelope),
        lambda: envelope_adapter.dump_json(envelope),
    )
    for call in calls:
        _assert_serialization_rejected(call, credential)


def test_minimally_forged_instance_is_rejected_before_internal_access() -> None:
    credential = _synthetic_credential()
    forged = _bare_forged_manifest(notes=credential)

    _assert_credential_rejected(
        lambda: AttemptManifest.model_validate(forged), credential
    )
    _assert_serialization_rejected(forged.model_dump, credential)


def test_artifact_ignore_rule_is_root_anchored() -> None:
    nested = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            "src/ratemem/artifacts/untracked-sentinel",
        ],
        cwd=_REPOSITORY,
        check=False,
    )
    generated = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "-q",
            "--",
            "artifacts/untracked-sentinel",
        ],
        cwd=_REPOSITORY,
        check=False,
    )

    assert nested.returncode == 1
    assert generated.returncode == 0


def test_task8_plan_scans_tracked_and_generated_outputs() -> None:
    plan = (
        _REPOSITORY
        / "docs/superpowers/plans/2026-08-24-ratemem-core-memory.md"
    ).read_text(encoding="utf-8")

    assert "git grep -Iq -E" in plan
    assert "rg --hidden --no-ignore -q" in plan
    for root in ("artifacts", "run_log", "logs", "exports"):
        assert root in plan


def test_core_plan_root_anchors_artifacts_and_lists_gitignore_change() -> None:
    plan = (
        _REPOSITORY
        / "docs/superpowers/plans/2026-08-24-ratemem-core-memory.md"
    ).read_text(encoding="utf-8")
    task_one = plan.split("### Task 1:", maxsplit=1)[1].split(
        "### Task 2:", maxsplit=1
    )[0]
    task_eight = plan.split("### Task 8:", maxsplit=1)[1].split(
        "### Task 9:", maxsplit=1
    )[0]

    assert "\n/artifacts/\n" in task_one
    assert "\nartifacts/\n" not in task_one
    assert "- Modify: `.gitignore`" in task_eight
    assert "nested artifact source and test packages remain tracked and scanned" in task_eight
