from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

SOURCE_PATH = Path("src/ratemem/pilot/modal_app.py")


def _source_and_tree() -> tuple[str, ast.Module]:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source)


class _Builder:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __getattr__(self, name: str) -> object:
        def record(*args: object, **kwargs: object) -> _Builder:
            self.calls.append((name, args, kwargs))
            return self

        return record


class _Volume:
    commits = 0
    on_commit: object = None

    @classmethod
    def from_name(cls, *_args: object, **_kwargs: object) -> _Volume:
        return cls()

    def commit(self) -> None:
        self.commits += 1
        if self.on_commit is not None:
            self.on_commit()


class _App:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def function(self, **_kwargs: object) -> object:
        return lambda function: function

    def local_entrypoint(self) -> object:
        return lambda function: function


def _load_with_fake_modal(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _Volume.commits = 0
    _Volume.on_commit = None
    _Builder.calls = []
    monkeypatch.setenv("MODAL_TASK_ID", "task-internal")
    fake = types.ModuleType("modal")
    fake.Image = types.SimpleNamespace(debian_slim=lambda **_kwargs: _Builder())
    fake.Volume = _Volume
    fake.App = _App
    fake.current_function_call_id = lambda: "fc-test"
    fake.current_input_id = lambda: "in-test"
    monkeypatch.setitem(sys.modules, "modal", fake)
    spec = importlib.util.spec_from_file_location("_ratemem_modal_contract", SOURCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request() -> dict[str, object]:
    commit = "1" * 40
    rates = {
        "gpu_l40s_per_second": "0.000542",
        "cpu_core_per_second": "0.0000131",
        "memory_gib_per_second": "0.00000222",
        "volume_gib_month": "0.09",
    }
    rates_bytes = json.dumps(rates, sort_keys=True, separators=(",", ":")).encode()
    return {
        "attempt_id": "019d0000-0000-7000-8000-000000000001",
        "workspace": "workspace",
        "source_sha256": hashlib.sha256(commit.encode()).hexdigest(),
        "git_commit": commit,
        "git_diff_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "slot_sha256": "4" * 64,
        "permit_sha256": "5" * 64,
        "submission_receipt_sha256": "6" * 64,
        "known_usage_before_usd": "1.25",
        "pending_worst_case_usd": "10.15",
        "phase_bound_usd": "10.15",
        "rates": rates,
        "rates_sha256": hashlib.sha256(rates_bytes).hexdigest(),
    }


def test_modal_job_has_one_synchronous_single_l40s_invocation() -> None:
    source, tree = _source_and_tree()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]

    assert attributes.count("remote") == 1
    assert not {
        "spawn",
        "spawn_map",
        "map",
        "starmap",
        "deploy",
        "detach",
        "serve",
    } & set(attributes)
    for forbidden in (".remote.aio", "rerun", "gpu=[", "A100", "H100", "A10"):
        assert forbidden not in source

    assert 'gpu="L40S"' in source
    assert "retries=0" in source
    assert "max_containers=1" in source
    assert "single_use_containers=True" in source
    assert "timeout=7200" in source
    assert "startup_timeout=1800" in source


def test_volumes_are_precreated_and_image_contains_no_credentials() -> None:
    source, _tree = _source_and_tree()
    assert source.count("create_if_missing=False") == 2
    assert 'volumes={"/cache": cache_volume, "/artifacts": artifact_volume}' in source
    assert "Secret.from_name" not in source
    assert "secrets=" not in source
    assert "HF_HUB_DISABLE_TELEMETRY" in source
    assert "WANDB_MODE" in source


def test_image_places_schema_where_installed_artifact_code_resolves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_with_fake_modal(monkeypatch)
    local_directories = [
        (args, kwargs) for name, args, kwargs in _Builder.calls if name == "add_local_dir"
    ]
    assert (("schemas", "/schemas"), {}) in local_directories


def test_remote_validates_request_device_and_commits_receipt_before_model_work() -> None:
    source, tree = _source_and_tree()
    assert "_validate_request(request)" in source
    required_fields = {
        "attempt_id",
        "workspace",
        "source_sha256",
        "git_commit",
        "git_diff_sha256",
        "config_sha256",
        "slot_sha256",
        "permit_sha256",
        "submission_receipt_sha256",
        "known_usage_before_usd",
        "pending_worst_case_usd",
        "phase_bound_usd",
        "rates",
        "rates_sha256",
    }
    request_keys = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_REQUEST_KEYS" for target in node.targets
        )
    )
    assert ast.literal_eval(request_keys.value) == required_fields
    for field in required_fields:
        assert f'"{field}"' in source
    assert "uuid.UUID" in source
    assert "hashlib.sha256" in source
    assert "Decimal" in source
    assert "torch.cuda.device_count() != 1" in source
    assert '"L40S" not in torch.cuda.get_device_name(0)' in source
    assert "forbidden credential variables are present" in source
    assert "execution-receipts" in source
    assert "lower_bound_may_miss_precommit_reschedule" in source
    assert "fcntl.flock" not in source
    assert "O_APPEND" not in source
    assert "O_EXCL" in source and "O_NOFOLLOW" in source
    assert "restrict_modal_access=" not in source
    assert "Volume.commit requires Modal resource access" in source
    assert source.index("artifact_volume.commit()") < source.index("run_real_pilot(")
    remote = source[source.index("def run_first_pilot(") :]
    assert remote.index("_validate_request(request)") < remote.index(
        "_commit_execution_receipt(checked_request)"
    )
    assert remote.index("_commit_execution_receipt(checked_request)") < remote.index(
        "_forbidden_credentials()"
    )
    assert remote.index("_commit_execution_receipt(checked_request)") < remote.index("import torch")
    assert 'task_id = os.environ["MODAL_TASK_ID"]' in remote
    assert '"task_id": task_id' in remote


def test_local_entrypoint_rechecks_environment_and_consumes_before_remote() -> None:
    source, _tree = _source_and_tree()
    assert "GLOBAL_SLOT_PATH" in source
    assert "GLOBAL_SUBMISSION_RECEIPT_PATH" in source
    assert 'os.environ.get("MODAL_ENVIRONMENT")' in source
    assert "permit_path" not in source
    assert source.index("consume_launch_request(") < source.index("run_first_pilot.remote(")


def test_each_execution_commits_one_create_only_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    monkeypatch.setenv("MODAL_TASK_ID", "task-internal")

    first = module._commit_execution_receipt(_request(), artifact_root=artifact_root)
    first_snapshot = first[0].read_bytes() + b"\n"
    assert first[3] == hashlib.sha256(first_snapshot).hexdigest()
    second = module._commit_execution_receipt(_request(), artifact_root=artifact_root)

    receipt_directory = artifact_root / "execution-receipts" / str(_request()["attempt_id"])
    files = sorted(receipt_directory.iterdir())
    assert len(files) == 2
    assert files[0].suffix == files[1].suffix == ".json"
    assert first[0] != second[0]
    assert first[2] == 1 and second[2] == 2
    second_snapshot = b"".join(path.read_bytes() + b"\n" for path in files)
    assert second[3] == hashlib.sha256(second_snapshot).hexdigest()
    for path in files:
        payload = json.loads(path.read_text())
        assert payload["receipt_id"] == path.stem
        assert payload["task_id"] == "task-internal"
        assert path.stat().st_mode & 0o777 == 0o600


def test_receipt_handles_short_write_and_rejects_unsafe_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    real_write = os.write
    shortened = False

    def short_once(descriptor: int, content: bytes) -> int:
        nonlocal shortened
        if not shortened and len(content) > 1:
            shortened = True
            return real_write(descriptor, content[: len(content) // 2])
        return real_write(descriptor, content)

    monkeypatch.setattr(module.os, "write", short_once)
    path, _directory, count, _snapshot_sha256, _function_id, _input_id = (
        module._commit_execution_receipt(_request(), artifact_root=artifact_root)
    )
    assert shortened and count == 1 and json.loads(path.read_text())["attempt_id"]

    symlink_root = tmp_path / "symlink-artifacts"
    symlink_root.mkdir(mode=0o700)
    (symlink_root / "execution-receipts").symlink_to(artifact_root / "execution-receipts")
    with pytest.raises(PermissionError):
        module._commit_execution_receipt(_request(), artifact_root=symlink_root)

    unsafe_root = tmp_path / "unsafe-artifacts"
    unsafe_root.mkdir(mode=0o700)
    unsafe_receipts = unsafe_root / "execution-receipts"
    unsafe_receipts.mkdir(mode=0o755)
    with pytest.raises(PermissionError):
        module._commit_execution_receipt(_request(), artifact_root=unsafe_root)


def test_credential_guard_rejects_tokens_without_rejecting_modal_task_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    monkeypatch.setenv("MODAL_TASK_SECRET", "internal-resource-access")
    monkeypatch.setenv("MODAL_TOKEN_ID", "forbidden-user-token")
    assert module._forbidden_credentials() == ["MODAL_TOKEN_ID"]


def test_receipt_rejects_final_name_replaced_with_equal_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    real_fsync = os.fsync
    replaced = False

    def replace_before_file_fsync(descriptor: int) -> None:
        nonlocal replaced
        metadata = os.fstat(descriptor)
        if not replaced and metadata.st_mode & 0o170000 == 0o100000:
            paths = list(artifact_root.rglob("*.json"))
            assert len(paths) == 1
            path = paths[0]
            os.lseek(descriptor, 0, os.SEEK_SET)
            content = os.read(descriptor, metadata.st_size)
            path.rename(tmp_path / "moved-original-receipt.json")
            path.write_bytes(content)
            path.chmod(0o600)
            replaced = True
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", replace_before_file_fsync)
    with pytest.raises(RuntimeError, match="inode|identity|changed"):
        module._commit_execution_receipt(_request(), artifact_root=artifact_root)
    assert replaced


def test_receipt_file_descriptor_is_closed_before_volume_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)

    def assert_no_receipt_fd() -> None:
        open_targets: list[str] = []
        for name in os.listdir("/proc/self/fd"):
            try:
                open_targets.append(os.readlink(f"/proc/self/fd/{name}"))
            except FileNotFoundError:
                pass
        assert not any(
            "execution-receipts" in target and target.endswith(".json") for target in open_targets
        )

    module.artifact_volume.on_commit = assert_no_receipt_fd
    module._commit_execution_receipt(_request(), artifact_root=artifact_root)


@pytest.mark.parametrize("rebound", ["attempt", "receipts", "root"])
def test_receipt_commit_rejects_rebound_parent_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rebound: str
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    attempt_id = str(_request()["attempt_id"])

    def replace_directory_binding() -> None:
        receipts = artifact_root / "execution-receipts"
        attempt = receipts / attempt_id
        if rebound == "attempt":
            attempt.rename(tmp_path / "moved-attempt")
            attempt.mkdir(mode=0o700)
        elif rebound == "receipts":
            receipts.rename(tmp_path / "moved-receipts")
            receipts.mkdir(mode=0o700)
        else:
            artifact_root.rename(tmp_path / "moved-root")
            artifact_root.mkdir(mode=0o700)

    module.artifact_volume.on_commit = replace_directory_binding
    with pytest.raises((PermissionError, RuntimeError), match="directory|binding|identity|changed"):
        module._commit_execution_receipt(_request(), artifact_root=artifact_root)


def test_receipt_snapshot_rejects_mixed_requests_for_one_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    module._commit_execution_receipt(_request(), artifact_root=artifact_root)
    changed = _request()
    changed["config_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="request"):
        module._commit_execution_receipt(changed, artifact_root=artifact_root)


def test_receipt_requires_modal_task_identity_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    monkeypatch.delenv("MODAL_TASK_ID")
    with pytest.raises(RuntimeError, match="task"):
        module._commit_execution_receipt(_request(), artifact_root=artifact_root)
    assert not list(artifact_root.rglob("*.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unknown": "field"}),
        lambda payload: payload.update({"function_call_id": ""}),
        lambda payload: payload.update({"task_id": ""}),
        lambda payload: payload.update({"observed_at": "2026-08-24T00:00:00+01:00"}),
        lambda payload: payload.update({"source_sha256": "0" * 64}),
        lambda payload: payload.update({"phase_bound_usd": float("nan")}),
        lambda payload: payload.update({"observed_at": "2026-08-24T00:00:00.000000Z"}),
        lambda payload: payload.update({"observed_at": "2026-08-24T00:00:00.000000-00:00"}),
        lambda payload: payload.update({"observed_at": "2026-W35-1T00:00:00.000000+00:00"}),
    ],
)
def test_snapshot_rejects_semantically_invalid_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: object,
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o700)
    first, _directory, _count, _snapshot_sha256, _function, _input = (
        module._commit_execution_receipt(_request(), artifact_root=artifact_root)
    )
    payload = json.loads(first.read_text())
    mutation(payload)
    first.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    first.chmod(0o600)
    with pytest.raises((TypeError, ValueError)):
        module._commit_execution_receipt(_request(), artifact_root=artifact_root)


@pytest.mark.parametrize("workspace", ["Uppercase", "-leading", "trailing-", "a" * 65])
def test_request_workspace_is_one_canonical_pilot_identity_slug(
    monkeypatch: pytest.MonkeyPatch, workspace: str
) -> None:
    module = _load_with_fake_modal(monkeypatch)
    request = _request()
    request["workspace"] = workspace
    with pytest.raises(ValueError, match="workspace"):
        module._validate_request(request)
