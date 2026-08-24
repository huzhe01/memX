from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

_SECRET_TEXT = re.compile(r"(?:ak|as)-[A-Za-z0-9_-]{20,}")
_SECRET_BYTES = re.compile(rb"(?:ak|as)-[A-Za-z0-9_-]{20,}")
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_CREDENTIAL_MESSAGE = "credential-shaped value is forbidden"
_UNSUPPORTED_MESSAGE = "unsupported manifest input"
_VALIDATION_MESSAGE = "manifest validation failed"
_SERIALIZATION_MESSAGE = "manifest serialization failed"
_JSON_MESSAGE = "Invalid JSON input"
_REDACTED_INPUT = "<redacted>"
_MANIFEST_FIELDS = frozenset(
    {"run_id", "git_revision", "config_hash", "status", "notes"}
)
_MANIFEST_STATUSES = frozenset({"passed", "failed", "interrupted"})


def _safe_error(code: str, message: str) -> ValidationError:
    detail: InitErrorDetails = {
        "type": PydanticCustomError(code, message),
        "loc": (),
        "input": _REDACTED_INPUT,
    }
    return ValidationError.from_exception_data(
        "AttemptManifest", [detail], hide_input=True
    )


def _credential_error() -> ValidationError:
    return _safe_error("credential_shaped", _CREDENTIAL_MESSAGE)


def _unsupported_error() -> ValidationError:
    return _safe_error("unsupported_manifest_input", _UNSUPPORTED_MESSAGE)


def _validation_error() -> ValidationError:
    return _safe_error("manifest_validation_failed", _VALIDATION_MESSAGE)


def _serialization_error() -> ValidationError:
    return _safe_error("manifest_serialization_failed", _SERIALIZATION_MESSAGE)


def _json_error() -> ValidationError:
    return _safe_error("json_invalid", _JSON_MESSAGE)


def _scan_canonical(value: object) -> None:
    """Scan a closed set of exact built-in values without protocol dispatch."""
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        current_type = type(current)

        if current_type is str:
            if _SECRET_TEXT.search(cast(str, current)):
                raise _credential_error()
            continue

        if current_type is bytes or current_type is bytearray:
            raw = memoryview(cast(bytes | bytearray, current)).tobytes()
            if _SECRET_BYTES.search(raw):
                raise _credential_error()
            continue

        if current_type is dict:
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            safe_dict = cast(dict[object, object], current)
            for key, item in dict.items(safe_dict):
                pending.append(key)
                pending.append(item)
            continue

        if current_type is list:
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            safe_list = cast(list[object], current)
            pending.extend(list.__iter__(safe_list))
            continue

        if current_type in (type(None), bool, int, float):
            continue

        raise _unsupported_error()


def _decode_json_input(value: object) -> str:
    value_type = type(value)
    if value_type is str:
        return cast(str, value)
    if value_type is bytes or value_type is bytearray:
        raw = memoryview(cast(bytes | bytearray, value)).tobytes()
        _scan_canonical(raw)
        decode_failed = False
        decoded = ""
        try:
            decoded = raw.decode("utf-8")
        except Exception:
            decode_failed = True
        if decode_failed:
            raise _json_error()
        return decoded
    raise _unsupported_error()


def _expand_json_unicode_escapes(value: str) -> str:
    return _UNICODE_ESCAPE.sub(
        lambda match: chr(int(match.group(1), 16)), value
    )


class AttemptManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    run_id: str
    git_revision: str
    config_hash: str
    status: Literal["passed", "failed", "interrupted"]
    notes: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("AttemptManifest does not support subclasses")

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate untrusted raw JSON without retaining its input in errors.

        This classmethod is the credential-safe boundary for untrusted raw JSON.
        Raw ``TypeAdapter.validate_json`` and enclosing-model JSON parsing happen
        before the model schema and may retain raw input; they are not
        credential-safe.
        """
        if cls is not AttemptManifest:
            raise _unsupported_error()
        raw_text = _decode_json_input(json_data)
        _scan_canonical(raw_text)
        _scan_canonical(_expand_json_unicode_escapes(raw_text))

        parse_failed = False
        parsed: object = None
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parse_failed = True
        if parse_failed:
            raise _json_error()

        _preflight_manifest_input(parsed)
        return cls.model_validate(
            parsed,
            strict=strict,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @model_validator(mode="wrap")
    @classmethod
    def preflight_credentials(
        cls, value: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        if cls is not AttemptManifest:
            raise _unsupported_error()
        _preflight_manifest_input(value)

        handler_failed = False
        validated: Self | None = None
        try:
            validated = handler(value)
        except Exception:
            handler_failed = True
        if handler_failed or validated is None:
            raise _validation_error()

        _inspect_manifest_instance(validated, phase="validation")
        return validated

    @model_serializer(mode="wrap")
    def serialize_with_credential_guard(
        self, handler: SerializerFunctionWrapHandler
    ) -> Any:
        _inspect_manifest_instance(self, phase="serialization")

        handler_failed = False
        serialized: Any = None
        try:
            serialized = handler(self)
        except Exception:
            handler_failed = True
        if handler_failed:
            raise _serialization_error()

        _scan_canonical(serialized)
        return serialized

    def __setattr__(self, name: str, value: Any) -> None:
        _scan_canonical(name)
        _scan_canonical(value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        _scan_canonical(name)
        super().__delattr__(name)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        _inspect_manifest_instance(self, phase="validation")
        fields_set = object.__getattribute__(self, "__pydantic_fields_set__")
        copied_fields = set.copy(fields_set)
        source_values = object.__getattribute__(self, "__dict__")
        values = copy.deepcopy(source_values) if deep else dict.copy(source_values)
        if update is not None:
            if type(update) is not dict:
                raise _unsupported_error()
            safe_update = cast(dict[str, Any], update)
            _scan_canonical(safe_update)
            copied_fields.update(dict.keys(safe_update))
            dict.update(values, safe_update)
        copied = type(self).model_validate(values)
        object.__setattr__(copied, "__pydantic_fields_set__", copied_fields)
        return copied

    @classmethod
    def model_construct(
        cls, _fields_set: set[str] | None = None, **values: Any
    ) -> Self:
        raise ValueError("AttemptManifest.model_construct is disabled")


def _preflight_manifest_input(value: object) -> None:
    value_type = type(value)
    if value_type is dict:
        _scan_canonical(value)
        return
    if value_type is AttemptManifest:
        _inspect_manifest_instance(
            cast(AttemptManifest, value), phase="validation"
        )
        return
    raise _unsupported_error()


def _read_internal_attribute(manifest: AttemptManifest, name: str) -> object:
    read_failed = False
    value: object = None
    try:
        value = object.__getattribute__(manifest, name)
    except Exception:
        read_failed = True
    if read_failed:
        raise _unsupported_error()
    return value


def _inspect_manifest_instance(
    manifest: AttemptManifest,
    *,
    phase: Literal["validation", "serialization"],
) -> None:
    if type(manifest) is not AttemptManifest:
        raise _unsupported_error()

    values = _read_internal_attribute(manifest, "__dict__")
    if type(values) is not dict:
        raise _unsupported_error()
    _scan_canonical(values)
    safe_values = cast(dict[object, object], values)

    if frozenset(dict.keys(safe_values)) != _MANIFEST_FIELDS:
        raise (
            _validation_error()
            if phase == "validation"
            else _serialization_error()
        )
    for field_name in _MANIFEST_FIELDS:
        if type(dict.__getitem__(safe_values, field_name)) is not str:
            raise _unsupported_error()
    if dict.__getitem__(safe_values, "status") not in _MANIFEST_STATUSES:
        raise (
            _validation_error()
            if phase == "validation"
            else _serialization_error()
        )

    extra = _read_internal_attribute(manifest, "__pydantic_extra__")
    private = _read_internal_attribute(manifest, "__pydantic_private__")
    fields_set = _read_internal_attribute(manifest, "__pydantic_fields_set__")
    if extra is not None or private is not None or type(fields_set) is not set:
        raise _unsupported_error()
    safe_fields_set = cast(set[object], fields_set)
    for internal_field in set.__iter__(safe_fields_set):
        if (
            type(internal_field) is not str
            or internal_field not in _MANIFEST_FIELDS
        ):
            raise _unsupported_error()
