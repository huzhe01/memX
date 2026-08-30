"""Cryptographic boundary for a sealed, one-time final evaluation trace.

The training package must never import this module.  A final trace can only be
opened by a separately invoked evaluation process whose complete frozen context
is approved with an Ed25519 signature.  The attempt ledger is created before
authenticated decryption, so even a failed attempt consumes the envelope.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import canonical_json_bytes, write_json_atomic
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_ALGORITHM: Literal["X25519-HKDF-SHA256-CHACHA20POLY1305"] = (
    "X25519-HKDF-SHA256-CHACHA20POLY1305"
)
_HKDF_INFO = b"ratemem-final-trace-v1"
_PATH_TYPE = type(Path())
_FREEZE_ID = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$"),
]


class FinalTraceAccessDenied(PermissionError):
    """Raised when a caller requests final trace access for a forbidden purpose."""


class FinalTraceAlreadyOpened(FileExistsError):
    """Raised when the one-time attempt ledger already exists."""


class AccessPurpose(str, Enum):
    """Closed set of reasons a process may present when requesting the trace."""

    TRAINING = "training"
    MODEL_SELECTION = "model_selection"
    COMPARATIVE_VALIDATION = "comparative_validation"
    FINAL_EVALUATION = "final_evaluation"


def _decode_canonical_base64(value: str, *, field: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{field} must be an exact str")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{field} must be canonical base64") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{field} must be canonical base64")
    return decoded


def _raw_x25519_public_key(public_key: X25519PublicKey) -> bytes:
    if not isinstance(public_key, X25519PublicKey):
        raise TypeError("public_key must be an X25519PublicKey")
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _raw_ed25519_public_key(public_key: Ed25519PublicKey) -> bytes:
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("approver_public_key must be an Ed25519PublicKey")
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _derive_key(shared_secret: bytes, manifest_sha256: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=manifest_sha256,
        info=_HKDF_INFO,
    ).derive(shared_secret)


def _jsonl_event_count(payload: bytes) -> int:
    if type(payload) is not bytes or not payload:
        raise ValueError("final trace plaintext must be non-empty exact bytes")
    lines = payload.splitlines()
    if not lines:
        raise ValueError("final trace plaintext must contain JSONL events")
    for line in lines:
        if not line:
            raise ValueError("final trace plaintext cannot contain empty JSONL rows")
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("final trace plaintext must be valid UTF-8 JSONL") from error
        if type(event) is not dict:
            raise ValueError("every final trace JSONL row must be an object")
    return len(lines)


class FinalTraceEnvelope(BaseModel):
    """Public authenticated-encryption envelope; it contains no plaintext."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["X25519-HKDF-SHA256-CHACHA20POLY1305"] = _ALGORITHM
    recipient_public_key_sha256: Sha256
    ephemeral_public_key_base64: str
    nonce_base64: str
    ciphertext_base64: str
    plaintext_sha256: Sha256
    associated_manifest_sha256: Sha256
    event_count: PositiveInt

    @field_validator(
        "ephemeral_public_key_base64",
        "nonce_base64",
        "ciphertext_base64",
    )
    @classmethod
    def validate_binary_fields(cls, value: str, info: Any) -> str:
        decoded = _decode_canonical_base64(value, field=info.field_name)
        expected_lengths = {
            "ephemeral_public_key_base64": 32,
            "nonce_base64": 12,
        }
        expected = expected_lengths.get(info.field_name)
        if expected is not None and len(decoded) != expected:
            raise ValueError(f"{info.field_name} must encode exactly {expected} bytes")
        if info.field_name == "ciphertext_base64" and len(decoded) < 16:
            raise ValueError("ciphertext_base64 must include an authentication tag")
        return value

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class FinalTracePublicManifest(BaseModel):
    """Public commitments for a final trace without any lifecycle event payload."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    split: Literal["final_test"] = "final_test"
    dataset_lock_id: Sha256
    trace_builder_revision: Literal["lifecycle_trace_v1"]
    trace_policy_sha256: Sha256
    power_record_sha256: Sha256
    concept_pool_sha256: Sha256
    prompt_pool_sha256: Sha256
    trace_ids: tuple[Sha256, ...]
    generation_seeds: tuple[NonNegativeInt, ...]
    trace_count: PositiveInt
    event_count: PositiveInt
    plaintext_sha256: Sha256

    @model_validator(mode="after")
    def validate_commitments(self) -> FinalTracePublicManifest:
        if (
            not self.trace_ids
            or self.trace_ids != tuple(sorted(set(self.trace_ids)))
            or self.trace_count != len(self.trace_ids)
        ):
            raise ValueError("final trace ids must be non-empty, sorted, unique, and counted")
        if self.generation_seeds != tuple(sorted(set(self.generation_seeds))):
            raise ValueError("final generation seeds must be sorted and unique")
        return self


class FinalEvaluationContext(BaseModel):
    """Every immutable input that an approval signature authorizes."""

    model_config = _MODEL_CONFIG

    git_commit: GitCommit
    clean_diff_sha256: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    method_lock_sha256: Sha256
    method_cpu_gate_sha256: Sha256
    comparative_execution_freeze_sha256: Sha256
    final_envelope_sha256: Sha256
    paid_compute: bool
    scientific_compute_authorization_sha256: Sha256 | None
    scientific_cost_reservation_sha256: Sha256 | None
    scientific_phase_launch_receipt_sha256: Sha256 | None

    @model_validator(mode="after")
    def validate_paid_compute_records(self) -> FinalEvaluationContext:
        records = (
            self.scientific_compute_authorization_sha256,
            self.scientific_cost_reservation_sha256,
            self.scientific_phase_launch_receipt_sha256,
        )
        if self.paid_compute and any(record is None for record in records):
            raise ValueError("paid final evaluation requires all compute approvals")
        if not self.paid_compute and any(record is not None for record in records):
            raise ValueError("unpaid final evaluation cannot name paid-compute approvals")
        return self


class FinalEvaluationPermit(BaseModel):
    """Signed authorization for exactly one frozen final-evaluation context."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    freeze_id: _FREEZE_ID
    context: FinalEvaluationContext
    approved_at_utc: AwareDatetime
    approver_public_key_sha256: Sha256
    ed25519_signature_base64: str

    @field_validator("ed25519_signature_base64")
    @classmethod
    def validate_signature(cls, value: str) -> str:
        if len(_decode_canonical_base64(value, field="ed25519_signature_base64")) != 64:
            raise ValueError("ed25519_signature_base64 must encode exactly 64 bytes")
        return value

    @property
    def unsigned_bytes(self) -> bytes:
        return canonical_json_bytes(
            self.model_dump(mode="json", exclude={"ed25519_signature_base64"})
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class SensitiveBytesIO(io.BytesIO):
    """Best-effort zeroization of the mutable plaintext buffer on close."""

    def close(self) -> None:
        if not self.closed:
            view = self.getbuffer()
            try:
                view[:] = b"\x00" * len(view)
            finally:
                view.release()
        super().close()


def generate_x25519_keypair() -> tuple[X25519PrivateKey, X25519PublicKey]:
    """Generate a fresh recipient keypair without serializing either key."""

    private_key = X25519PrivateKey.generate()
    return private_key, private_key.public_key()


def seal_final_trace(
    plaintext: bytes,
    recipient_public_key: X25519PublicKey,
    *,
    associated_manifest: bytes,
) -> FinalTraceEnvelope:
    """Encrypt exact JSONL trace bytes and authenticate the public manifest."""

    if type(associated_manifest) is not bytes or not associated_manifest:
        raise ValueError("associated manifest must be non-empty exact bytes")
    event_count = _jsonl_event_count(plaintext)
    recipient_raw = _raw_x25519_public_key(recipient_public_key)
    ephemeral_private_key = X25519PrivateKey.generate()
    ephemeral_raw = _raw_x25519_public_key(ephemeral_private_key.public_key())
    manifest_digest = hashlib.sha256(associated_manifest).digest()
    shared_secret = ephemeral_private_key.exchange(recipient_public_key)
    encryption_key = _derive_key(shared_secret, manifest_digest)
    nonce = os.urandom(12)
    ciphertext = ChaCha20Poly1305(encryption_key).encrypt(
        nonce,
        plaintext,
        associated_manifest,
    )
    return FinalTraceEnvelope(
        recipient_public_key_sha256=hashlib.sha256(recipient_raw).hexdigest(),
        ephemeral_public_key_base64=base64.b64encode(ephemeral_raw).decode("ascii"),
        nonce_base64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_base64=base64.b64encode(ciphertext).decode("ascii"),
        plaintext_sha256=hashlib.sha256(plaintext).hexdigest(),
        associated_manifest_sha256=manifest_digest.hex(),
        event_count=event_count,
    )


def open_final_trace(
    envelope: FinalTraceEnvelope,
    recipient_private_key: X25519PrivateKey,
    *,
    associated_manifest: bytes,
) -> SensitiveBytesIO:
    """Authenticate and decrypt one envelope into a wipe-on-close memory stream."""

    if not isinstance(envelope, FinalTraceEnvelope):
        raise TypeError("envelope must be a FinalTraceEnvelope")
    if not isinstance(recipient_private_key, X25519PrivateKey):
        raise TypeError("recipient_private_key must be an X25519PrivateKey")
    if type(associated_manifest) is not bytes or not associated_manifest:
        raise ValueError("associated manifest must be non-empty exact bytes")
    manifest_digest = hashlib.sha256(associated_manifest).hexdigest()
    if not hmac.compare_digest(manifest_digest, envelope.associated_manifest_sha256):
        raise ValueError("associated manifest does not match the sealed envelope")

    ephemeral_raw = _decode_canonical_base64(
        envelope.ephemeral_public_key_base64,
        field="ephemeral_public_key_base64",
    )
    ephemeral_public_key = X25519PublicKey.from_public_bytes(ephemeral_raw)
    shared_secret = recipient_private_key.exchange(ephemeral_public_key)
    encryption_key = _derive_key(shared_secret, bytes.fromhex(manifest_digest))
    plaintext = ChaCha20Poly1305(encryption_key).decrypt(
        _decode_canonical_base64(envelope.nonce_base64, field="nonce_base64"),
        _decode_canonical_base64(
            envelope.ciphertext_base64,
            field="ciphertext_base64",
        ),
        associated_manifest,
    )

    recipient_raw = _raw_x25519_public_key(recipient_private_key.public_key())
    if not hmac.compare_digest(
        hashlib.sha256(recipient_raw).hexdigest(),
        envelope.recipient_public_key_sha256,
    ):
        raise ValueError("recipient private key does not match the envelope")
    if not hmac.compare_digest(
        hashlib.sha256(plaintext).hexdigest(),
        envelope.plaintext_sha256,
    ):
        raise ValueError("decrypted final trace hash does not match its envelope")
    if _jsonl_event_count(plaintext) != envelope.event_count:
        raise ValueError("decrypted final trace event count does not match its envelope")
    return SensitiveBytesIO(plaintext)


def create_final_evaluation_permit(
    *,
    freeze_id: str,
    context: FinalEvaluationContext,
    signing_key: Ed25519PrivateKey,
    approved_at_utc: datetime,
) -> FinalEvaluationPermit:
    """Sign all frozen inputs needed for one final evaluation."""

    if not isinstance(context, FinalEvaluationContext):
        raise TypeError("context must be a FinalEvaluationContext")
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise TypeError("signing_key must be an Ed25519PrivateKey")
    public_raw = _raw_ed25519_public_key(signing_key.public_key())
    # Validate and normalize every claim before signing its canonical form.
    placeholder = FinalEvaluationPermit(
        freeze_id=freeze_id,
        context=context,
        approved_at_utc=approved_at_utc,
        approver_public_key_sha256=hashlib.sha256(public_raw).hexdigest(),
        ed25519_signature_base64=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )
    signature = signing_key.sign(placeholder.unsigned_bytes)
    return placeholder.model_copy(
        update={
            "ed25519_signature_base64": base64.b64encode(signature).decode("ascii")
        }
    )


def _verify_permit(
    permit: FinalEvaluationPermit,
    *,
    approver_public_key: Ed25519PublicKey,
    current_context: FinalEvaluationContext,
    envelope: FinalTraceEnvelope,
) -> None:
    if not isinstance(permit, FinalEvaluationPermit):
        raise TypeError("permit must be a FinalEvaluationPermit")
    if not isinstance(current_context, FinalEvaluationContext):
        raise TypeError("current_context must be a FinalEvaluationContext")
    public_raw = _raw_ed25519_public_key(approver_public_key)
    if not hmac.compare_digest(
        hashlib.sha256(public_raw).hexdigest(),
        permit.approver_public_key_sha256,
    ):
        raise FinalTraceAccessDenied("permit was signed by a different approver")
    try:
        approver_public_key.verify(
            _decode_canonical_base64(
                permit.ed25519_signature_base64,
                field="ed25519_signature_base64",
            ),
            permit.unsigned_bytes,
        )
    except Exception as error:
        raise FinalTraceAccessDenied("final evaluation permit signature is invalid") from error
    if permit.context != current_context:
        raise FinalTraceAccessDenied("current context differs from the approved freeze")
    if current_context.final_envelope_sha256 != envelope.sha256:
        raise FinalTraceAccessDenied("approved context names a different final envelope")


def _create_open_ledger_exclusively(path: Path, payload: dict[str, object]) -> None:
    if type(path) is not _PATH_TYPE:
        raise TypeError("ledger_path must be an exact pathlib.Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise FinalTraceAlreadyOpened(
            f"final trace open attempt already exists: {path}"
        ) from error
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def acquire_final_trace(
    envelope: FinalTraceEnvelope,
    *,
    purpose: AccessPurpose,
    permit: FinalEvaluationPermit | None,
    ledger_path: Path,
    private_key: X25519PrivateKey | None = None,
    approver_public_key: Ed25519PublicKey | None = None,
    current_context: FinalEvaluationContext | None = None,
    associated_manifest: bytes | None = None,
) -> Generator[SensitiveBytesIO, None, None]:
    """Open a final trace exactly once after validating its signed freeze."""

    if purpose is not AccessPurpose.FINAL_EVALUATION:
        raise FinalTraceAccessDenied(
            f"final trace access is forbidden for purpose {purpose.value!r}"
        )
    if (
        permit is None
        or private_key is None
        or approver_public_key is None
        or current_context is None
        or associated_manifest is None
    ):
        raise FinalTraceAccessDenied("final evaluation requires a complete signed context")

    _verify_permit(
        permit,
        approver_public_key=approver_public_key,
        current_context=current_context,
        envelope=envelope,
    )
    initial_ledger: dict[str, object] = {
        "schema_version": "1.0",
        "status": "opened",
        "exit_status": "pending",
        "purpose": purpose.value,
        "freeze_id": permit.freeze_id,
        "permit_sha256": permit.sha256,
        "envelope_sha256": envelope.sha256,
    }
    _create_open_ledger_exclusively(ledger_path, initial_ledger)

    stream: SensitiveBytesIO | None = None
    try:
        stream = open_final_trace(
            envelope,
            private_key,
            associated_manifest=associated_manifest,
        )
        yield stream
    except BaseException as error:
        write_json_atomic(
            ledger_path,
            {
                **initial_ledger,
                "exit_status": "error",
                "error_class": type(error).__name__,
            },
        )
        os.chmod(ledger_path, 0o600)
        raise
    else:
        write_json_atomic(
            ledger_path,
            {
                **initial_ledger,
                "exit_status": "success",
            },
        )
        os.chmod(ledger_path, 0o600)
    finally:
        if stream is not None:
            stream.close()


__all__ = [
    "AccessPurpose",
    "FinalEvaluationContext",
    "FinalEvaluationPermit",
    "FinalTraceAccessDenied",
    "FinalTraceAlreadyOpened",
    "FinalTraceEnvelope",
    "FinalTracePublicManifest",
    "SensitiveBytesIO",
    "acquire_final_trace",
    "create_final_evaluation_permit",
    "generate_x25519_keypair",
    "open_final_trace",
    "seal_final_trace",
]
