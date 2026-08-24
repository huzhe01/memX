import pytest
from pydantic import ValidationError

from ratemem.artifacts.schema import AttemptManifest


def test_manifest_rejects_secret_shaped_values() -> None:
    try:
        AttemptManifest(
            run_id="cpu-smoke",
            git_revision="f" * 40,
            config_hash="a" * 64,
            status="passed",
            notes="token " + "ak-" + "SYNTHETIC_TEST_VALUE_123456789",
        )
    except ValueError as exc:
        assert "credential-shaped" in str(exc)
    else:
        raise AssertionError("credential-shaped value was accepted")


@pytest.mark.parametrize("field", ["run_id", "git_revision", "config_hash", "notes"])
@pytest.mark.parametrize("prefix", ["ak-", "as-"])
def test_manifest_rejects_every_credential_shape_without_echoing_input(
    field: str, prefix: str
) -> None:
    synthetic_credential = prefix + "SYNTHETIC_TEST_VALUE_123456789"
    values = {
        "run_id": "cpu-smoke",
        "git_revision": "f" * 40,
        "config_hash": "a" * 64,
        "status": "passed",
        "notes": "",
    }
    values[field] = synthetic_credential

    with pytest.raises(ValueError, match="credential-shaped") as raised:
        AttemptManifest.model_validate(values)

    assert synthetic_credential not in str(raised.value)
    assert synthetic_credential not in repr(raised.value)


def test_manifest_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AttemptManifest.model_validate(
            {
                "run_id": "cpu-smoke",
                "git_revision": "f" * 40,
                "config_hash": "a" * 64,
                "status": "passed",
                "unexpected": "value",
            }
        )


def test_manifest_is_frozen() -> None:
    manifest = AttemptManifest(
        run_id="cpu-smoke",
        git_revision="f" * 40,
        config_hash="a" * 64,
        status="passed",
    )

    with pytest.raises(ValidationError):
        setattr(manifest, "notes", "changed")
