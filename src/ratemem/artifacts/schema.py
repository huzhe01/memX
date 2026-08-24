from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Literal, Self

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
_CREDENTIAL_MESSAGE = "credential-shaped value is forbidden"
_REDACTED_INPUT = "<credential-redacted>"


def _contains_credential(value: object) -> bool:
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if _SECRET_TEXT.search(current):
                return True
            continue
        if isinstance(current, bytes | bytearray | memoryview):
            if _SECRET_BYTES.search(bytes(current)):
                return True
            continue

        if isinstance(current, BaseModel):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            pending.append(current.__dict__)
            extra = getattr(current, "__pydantic_extra__", None)
            if extra is not None:
                pending.append(extra)
            continue

        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            for key, item in current.items():
                pending.append(key)
                pending.append(item)
            continue

        if isinstance(current, list | tuple | set | frozenset):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(current)
    return False


def _credential_error() -> ValidationError:
    detail: InitErrorDetails = {
        "type": PydanticCustomError("credential_shaped", _CREDENTIAL_MESSAGE),
        "loc": (),
        "input": _REDACTED_INPUT,
    }
    return ValidationError.from_exception_data(
        "AttemptManifest", [detail], hide_input=True
    )


def _reject_credentials(value: object) -> None:
    if _contains_credential(value):
        raise _credential_error()


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

    @model_validator(mode="wrap")
    @classmethod
    def preflight_credentials(
        cls, value: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        _reject_credentials(value)
        return handler(value)

    @model_serializer(mode="wrap")
    def serialize_with_credential_guard(
        self, handler: SerializerFunctionWrapHandler
    ) -> Any:
        _reject_credentials(self)
        return handler(self)

    def __setattr__(self, name: str, value: Any) -> None:
        _reject_credentials({name: value})
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        _reject_credentials(name)
        super().__delattr__(name)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        values = copy.deepcopy(self.__dict__) if deep else dict(self.__dict__)
        if update is not None:
            values.update(update)
        return type(self).model_validate(values)

    @classmethod
    def model_construct(
        cls, _fields_set: set[str] | None = None, **values: Any
    ) -> Self:
        raise ValueError("AttemptManifest.model_construct is disabled")
