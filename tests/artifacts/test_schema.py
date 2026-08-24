from __future__ import annotations

import json
import operator
import subprocess
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError, field_serializer

from ratemem.artifacts.schema import AttemptManifest

_REPOSITORY = Path(__file__).resolve().parents[2]
_SYNTHETIC_SUFFIX = "SYNTHETIC_TEST_VALUE_123456789"


class _ManifestEnvelope(BaseModel):
    manifest: AttemptManifest


class _OverridingStr(str):
    calls: dict[str, int]

    def __new__(
        cls, payload: str, calls: dict[str, int]
    ) -> _OverridingStr:
        instance = str.__new__(cls, payload)
        instance.calls = calls
        return instance

    @property
    def __class__(self) -> type[str]:
        self.calls["class"] += 1
        return str

    def __str__(self) -> str:
        self.calls["str"] += 1
        return "safe"

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<overriding-str>"


class _OverridingBytes(bytes):
    calls: dict[str, int]

    def __new__(
        cls, payload: bytes, calls: dict[str, int]
    ) -> _OverridingBytes:
        instance = bytes.__new__(cls, payload)
        instance.calls = calls
        return instance

    @property
    def __class__(self) -> type[bytes]:
        self.calls["class"] += 1
        return bytes

    def __bytes__(self) -> bytes:
        self.calls["bytes"] += 1
        return b"safe"

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<overriding-bytes>"


class _OverridingBytearray(bytearray):
    calls: dict[str, int]

    def __init__(self, payload: bytes, calls: dict[str, int]) -> None:
        bytearray.__init__(self, payload)
        self.calls = calls

    @property
    def __class__(self) -> type[bytearray]:
        self.calls["class"] += 1
        return bytearray

    def __bytes__(self) -> bytes:
        self.calls["bytes"] += 1
        return b"safe"

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<overriding-bytearray>"


class _HidingDict(dict[str, object]):
    calls: dict[str, int]

    def __init__(
        self, values: dict[str, object], calls: dict[str, int]
    ) -> None:
        dict.__init__(self, values)
        self.calls = calls

    @property
    def __class__(self) -> type[dict[object, object]]:
        self.calls["class"] += 1
        return dict

    def items(self):
        self.calls["items"] += 1
        return {}.items()

    def __iter__(self) -> Iterator[str]:
        self.calls["iter"] += 1
        return iter(())

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<hiding-dict>"


class _RaisingMapping(Mapping[str, object]):
    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls

    @property
    def __class__(self) -> type[dict[object, object]]:
        self.calls["class"] += 1
        return dict

    def __getitem__(self, key: str) -> object:
        self.calls["getitem"] += 1
        raise RuntimeError("mapping getitem must not run")

    def __iter__(self) -> Iterator[str]:
        self.calls["iter"] += 1
        raise RuntimeError("mapping iter must not run")

    def __len__(self) -> int:
        self.calls["len"] += 1
        raise RuntimeError("mapping len must not run")

    def items(self):
        self.calls["items"] += 1
        raise RuntimeError("mapping items must not run")

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<raising-mapping>"


class _SpoofedUnknown:
    advertised_class: ClassVar[type[dict[object, object]]] = dict

    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls

    @property
    def __class__(self) -> type[dict[object, object]]:
        self.calls["class"] += 1
        return self.advertised_class

    def __iter__(self) -> Iterator[object]:
        self.calls["iter"] += 1
        raise RuntimeError("unknown iter must not run")

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<spoofed-unknown>"


class _DeepcopyTrap:
    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        self.calls["deepcopy"] += 1
        raise RuntimeError("deepcopy must not run")

    def __repr__(self) -> str:
        self.calls["repr"] += 1
        return "<deepcopy-trap>"


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


def _assert_markers_not_leaked(
    error: BaseException, markers: tuple[str, ...]
) -> None:
    for marker in markers:
        _assert_not_leaked(error, marker)


def _assert_safe_value_error(
    call: Callable[[], object],
    markers: tuple[str, ...],
    *,
    match: str,
) -> ValueError:
    with pytest.raises(ValueError, match=match) as raised:
        call()
    _assert_markers_not_leaked(raised.value, markers)
    return raised.value


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


def _assert_safe_serialization_error(
    call: Callable[[], object],
    markers: tuple[str, ...],
    *,
    match: str,
) -> BaseException:
    try:
        call()
    except Exception as error:
        assert match in str(error)
        _assert_markers_not_leaked(error, markers)
        return error
    raise AssertionError("unsafe manifest serialization was accepted")


def _fresh_calls(*names: str) -> dict[str, int]:
    return dict.fromkeys(names, 0)


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


@pytest.mark.parametrize("scalar_type", ["str", "bytes", "bytearray"])
def test_overrideable_scalar_subclasses_are_rejected_without_dispatch(
    scalar_type: str,
) -> None:
    credential = _synthetic_credential()
    calls = _fresh_calls("class", "str", "bytes", "repr")
    value: object
    if scalar_type == "str":
        value = _OverridingStr(credential, calls)
    elif scalar_type == "bytes":
        value = _OverridingBytes(credential.encode("ascii"), calls)
    else:
        value = _OverridingBytearray(credential.encode("ascii"), calls)
    values = _valid_input()
    values["notes"] = value

    _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(values),
        (credential,),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


def test_hiding_dict_subclass_is_rejected_without_mapping_dispatch() -> None:
    credential = _synthetic_credential("as-")
    calls = _fresh_calls("class", "items", "iter", "repr")
    values = _valid_input()
    values["notes"] = credential
    hidden = _HidingDict(values, calls)

    _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(hidden),
        (credential,),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


def test_custom_mapping_is_rejected_without_protocol_calls() -> None:
    calls = _fresh_calls("class", "getitem", "iter", "len", "items", "repr")
    mapping = _RaisingMapping(calls)

    _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(mapping),
        ("mapping getitem must not run", "mapping iter must not run"),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


@pytest.mark.parametrize("unsupported_type", ["deque", "path", "namespace", "unknown"])
def test_unsupported_objects_are_rejected_without_repr_or_iteration(
    unsupported_type: str,
) -> None:
    marker = "SYNTHETIC_UNSUPPORTED_VALUE_123456789"
    calls = _fresh_calls("class", "iter", "repr")
    if unsupported_type == "deque":
        unsupported: object = deque([marker])
    elif unsupported_type == "path":
        unsupported = Path(marker)
    elif unsupported_type == "namespace":
        unsupported = SimpleNamespace(value=marker)
    else:
        unsupported = _SpoofedUnknown(calls)
    values = _valid_input()
    values["notes"] = unsupported

    _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(values),
        (marker,),
        match="unsupported manifest input",
    )
    if unsupported_type == "unknown":
        assert not any(calls.values())


def test_nested_custom_mapping_is_rejected_before_handler_dispatch() -> None:
    calls = _fresh_calls("class", "getitem", "iter", "len", "items", "repr")
    values = _valid_input()
    values["unexpected"] = [_RaisingMapping(calls)]

    _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(values),
        ("mapping items must not run",),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


def test_handler_failures_are_replaced_without_original_context() -> None:
    marker = "SYNTHETIC_HANDLER_FAILURE_123456789"
    values = _valid_input()
    values["status"] = marker

    error = _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(values),
        (marker,),
        match="manifest validation failed",
    )
    assert error.__cause__ is None
    assert error.__context__ is None


def test_exact_container_cycles_terminate_before_sanitized_handler_error() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    values = _valid_input()
    values["unexpected"] = cycle

    error = _assert_safe_value_error(
        lambda: AttemptManifest.model_validate(values),
        (),
        match="manifest validation failed",
    )
    assert error.__context__ is None


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


@pytest.mark.parametrize("location", ["value", "key"])
@pytest.mark.parametrize("payload_type", ["str", "bytes"])
def test_public_json_boundary_sanitizes_malformed_unicode_escaped_shapes(
    location: str, payload_type: str
) -> None:
    credential = _synthetic_credential()
    escaped = "".join(f"\\u{ord(character):04x}" for character in credential)
    if location == "value":
        raw = '{"notes":"' + escaped
    else:
        raw = '{"' + escaped + '":'
    payload: str | bytes = raw if payload_type == "str" else raw.encode("ascii")

    error = _assert_safe_value_error(
        lambda: AttemptManifest.model_validate_json(payload),
        (credential, escaped, escaped.replace("\\", "\\\\")),
        match="credential-shaped",
    )
    assert error.__context__ is None


def test_public_json_boundary_sanitizes_safe_parser_failures() -> None:
    marker = "SYNTHETIC_MALFORMED_JSON_123456789"
    raw = '{"notes":"' + marker

    error = _assert_safe_value_error(
        lambda: AttemptManifest.model_validate_json(raw),
        (marker,),
        match="Invalid JSON",
    )
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("payload_type", ["str", "bytes", "bytearray"])
def test_public_json_boundary_rejects_raw_scalar_subclasses(
    payload_type: str,
) -> None:
    credential = _synthetic_credential("as-")
    raw = '{"notes":"' + credential
    calls = _fresh_calls("class", "str", "bytes", "repr")
    payload: object
    if payload_type == "str":
        payload = _OverridingStr(raw, calls)
    elif payload_type == "bytes":
        payload = _OverridingBytes(raw.encode("ascii"), calls)
    else:
        payload = _OverridingBytearray(raw.encode("ascii"), calls)

    _assert_safe_value_error(
        lambda: AttemptManifest.model_validate_json(payload),  # type: ignore[arg-type]
        (credential,),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


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


def test_valid_inputs_work_across_all_supported_schema_paths() -> None:
    values = _valid_input()
    serialized = json.dumps(values)
    manifest = AttemptManifest.model_validate(values)
    expected = AttemptManifest(**values)

    assert manifest == expected
    assert AttemptManifest.model_validate(manifest) == expected
    assert AttemptManifest.model_validate_json(
        serialized,
        strict=True,
        context={"safe": True},
        by_name=True,
    ) == expected
    assert AttemptManifest.model_validate_json(serialized.encode("utf-8")) == expected
    assert AttemptManifest.model_validate_json(bytearray(serialized, "utf-8")) == expected
    assert TypeAdapter(AttemptManifest).validate_python(values) == expected
    assert TypeAdapter(AttemptManifest).validate_json(serialized) == expected
    assert _ManifestEnvelope.model_validate({"manifest": values}).manifest == expected
    assert _ManifestEnvelope.model_validate_json(
        json.dumps({"manifest": values})
    ).manifest == expected


def test_root_rejects_non_manifest_containers_without_scanning_protocols() -> None:
    credential = _synthetic_credential()
    unsupported_roots: tuple[object, ...] = (
        [credential],
        (credential,),
        {credential},
        memoryview(credential.encode("ascii")),
    )

    for value in unsupported_roots:
        _assert_safe_value_error(
            lambda value=value: AttemptManifest.model_validate(value),
            (credential,),
            match="unsupported manifest input",
        )


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


def test_model_copy_rejects_custom_updates_without_mapping_dispatch() -> None:
    manifest = AttemptManifest.model_validate(_valid_input())
    calls = _fresh_calls("class", "getitem", "iter", "len", "items", "repr")
    update = _RaisingMapping(calls)

    _assert_safe_value_error(
        lambda: manifest.model_copy(update=update),
        ("mapping getitem must not run", "mapping iter must not run"),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


def test_model_copy_preflights_forged_fields_before_deepcopy() -> None:
    marker = "SYNTHETIC_DEEPCOPY_VALUE_123456789"
    calls = _fresh_calls("deepcopy", "repr")
    forged = _forged_manifest(notes=_DeepcopyTrap(calls))

    _assert_safe_value_error(
        lambda: forged.model_copy(deep=True),
        (marker, "deepcopy must not run"),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


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


def test_manifest_subclass_cannot_inject_a_field_serializer() -> None:
    credential = _synthetic_credential("as-")
    try:

        class InjectingManifest(AttemptManifest):
            @field_serializer("notes")
            def inject_notes(self, value: str) -> str:
                return credential

    except TypeError as error:
        assert "does not support subclasses" in str(error)
        _assert_not_leaked(error, credential)
    else:
        manifest = InjectingManifest.model_validate(_valid_input())
        _assert_serialization_rejected(manifest.model_dump, credential)


def test_serializer_scans_transformed_handler_output() -> None:
    credential = _synthetic_credential()
    manifest = AttemptManifest.model_validate(_valid_input())

    _assert_serialization_rejected(
        lambda: manifest.serialize_with_credential_guard(
            lambda value: {"notes": [credential]}
        ),
        credential,
    )


def test_serializer_rejects_custom_handler_output_without_protocol_dispatch() -> None:
    manifest = AttemptManifest.model_validate(_valid_input())
    calls = _fresh_calls("class", "getitem", "iter", "len", "items", "repr")

    _assert_safe_serialization_error(
        lambda: manifest.serialize_with_credential_guard(
            lambda value: _RaisingMapping(calls)
        ),
        ("mapping items must not run", "mapping iter must not run"),
        match="unsupported manifest input",
    )
    assert not any(calls.values())


def test_serializer_handler_failures_drop_original_context() -> None:
    marker = "SYNTHETIC_SERIALIZER_FAILURE_123456789"
    manifest = AttemptManifest.model_validate(_valid_input())

    def fail_handler(value: object) -> object:
        raise RuntimeError(marker)

    error = _assert_safe_serialization_error(
        lambda: manifest.serialize_with_credential_guard(fail_handler),
        (marker,),
        match="manifest serialization failed",
    )
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("unsupported_type", ["deque", "path"])
def test_forged_unsupported_fields_are_rejected_before_fallback(
    unsupported_type: str,
) -> None:
    credential = _synthetic_credential("as-")
    unsupported: object
    if unsupported_type == "deque":
        unsupported = deque([credential])
    else:
        unsupported = Path(credential)
    forged = _forged_manifest(notes=unsupported)
    adapter = TypeAdapter(AttemptManifest)
    envelope = BaseModel.model_construct.__func__(
        _ManifestEnvelope,
        _fields_set={"manifest"},
        manifest=forged,
    )
    envelope_adapter = TypeAdapter(_ManifestEnvelope)
    fallback_calls: list[object] = []

    def fallback(value: object) -> str:
        fallback_calls.append(value)
        return "fallback"

    calls: tuple[Callable[[], object], ...] = (
        lambda: forged.model_dump(fallback=fallback),
        lambda: forged.model_dump_json(fallback=fallback),
        lambda: adapter.dump_python(forged, fallback=fallback),
        lambda: adapter.dump_json(forged, fallback=fallback),
        lambda: envelope.model_dump(fallback=fallback),
        lambda: envelope.model_dump_json(fallback=fallback),
        lambda: envelope_adapter.dump_python(envelope, fallback=fallback),
        lambda: envelope_adapter.dump_json(envelope, fallback=fallback),
    )
    for call in calls:
        _assert_safe_serialization_error(
            call,
            (credential,),
            match="unsupported manifest input",
        )
    assert fallback_calls == []


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

    assert "git grep --untracked --exclude-standard -Iq -E" in plan
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
