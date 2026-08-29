from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/run_modal_pilot.sh")
RUNBOOK = Path("docs/runbooks/ratemem-sana-modal-pilot.md")


def test_launch_script_is_valid_executable_bash_with_private_umask() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert source.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in source
    assert "umask 077" in source
    assert "/usr/bin/env -i" in source
    assert "MODAL_CONFIG_PATH=/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml" in source
    assert "/home/ubuntu/.local/bin/uv run --extra modal" in source
    assert stat.S_IMODE(SCRIPT.stat().st_mode) == 0o755


def test_launch_script_has_one_synchronous_guarded_modal_run() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    invocations = re.findall(r"^run_guarded_uv modal run .*?$", source, re.MULTILINE)
    assert invocations == ["run_guarded_uv modal run -m ratemem.pilot.modal_app"]
    assert source.count("modal run") == 1
    preflight = source.index("ratemem-pilot preflight")
    config = source.index("ratemem-pilot validate-modal-config configured")
    provision = source.index("ratemem-pilot provision-volumes")
    attempt = source.index('RATEMEM_ATTEMPT_ID="$(')
    destination_guard = source.index("artifact destination already exists")
    remote = source.index("modal run -m ratemem.pilot.modal_app")
    download = source.index("modal volume get")
    validate = source.index("ratemem-pilot validate-artifact")
    scan = source.index("ratemem-pilot security-scan")
    assert (
        config
        < preflight
        < provision
        < attempt
        < destination_guard
        < remote
        < download
        < scan
        < validate
    )
    assert "attempts/${RATEMEM_ATTEMPT_ID}" in source
    assert "artifacts/pilot/${RATEMEM_ATTEMPT_ID}" in source
    volume_get = next(line for line in source.splitlines() if "modal volume get" in line)
    assert volume_get.endswith('"${RATEMEM_DOWNLOAD_ROOT}"')
    assert '"${RATEMEM_DESTINATION}"' not in volume_get
    assert "--force" not in source


def test_guarded_wrapper_discards_malicious_modal_environment(tmp_path: Path) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"run_guarded_uv\(\) \{\n.*?\n\}", source, re.DOTALL)
    assert match is not None
    probe = tmp_path / "uv-probe"
    probe.write_text("#!/bin/sh\n/usr/bin/env\n", encoding="utf-8")
    probe.chmod(0o755)
    function = match.group(0).replace("/home/ubuntu/.local/bin/uv", str(probe))
    token_id_name = "MODAL_TOKEN_" + "ID"
    token_secret_name = "MODAL_TOKEN_" + "SECRET"
    completed = subprocess.run(
        ["bash", "-c", f"{function}\nrun_guarded_uv probe"],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": "/attacker/home",
            "MODAL_CONFIG_PATH": "/attacker/config.toml",
            token_id_name: "synthetic-id",
            token_secret_name: "synthetic-secret",
            "MODAL_SERVER_URL": "https://attacker.invalid",
        },
    )
    environment = dict(
        line.split("=", maxsplit=1) for line in completed.stdout.splitlines() if "=" in line
    )
    assert environment["HOME"] == "/home/ubuntu"
    assert environment["MODAL_CONFIG_PATH"] == (
        "/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml"
    )
    assert environment["MODAL_PROFILE"] == "ratemem-pilot"
    assert environment["MODAL_ENVIRONMENT"] == "main"
    assert token_id_name not in environment
    assert token_secret_name not in environment
    assert "MODAL_SERVER_URL" not in environment


def test_launch_script_exposes_no_retry_auth_or_expansive_modal_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    command_lines = [line for line in source.splitlines() if line and not line.startswith("#")]
    assert all(
        not line.startswith("uv run ") or line.startswith("uv run --extra modal ")
        for line in command_lines
    )
    for forbidden in (
        "modal deploy",
        "modal serve",
        "modal shell",
        "modal token",
        "--detach",
        ".spawn",
        ".map",
        "while ",
        "until ",
        "for ",
        "retry",
        "RATEMEM_ATTEMPT_ID:-",
        "--permit-path",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "||",
    ):
        assert forbidden not in source
    assert source.count("modal volume get") == 1
    assert os.path.basename(str(SCRIPT)) == "run_modal_pilot.sh"


def test_runbook_orders_dashboard_before_hidden_auth_and_documents_recovery() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    dashboard = source.index("Workspace usage budget** to exactly **USD 28.00**")
    authentication = source.index("modal token set --profile ratemem-pilot --no-activate --verify")
    assert dashboard < authentication
    assert "xinming-hu-rd" in source
    assert "ak-" not in source and "as-" not in source
    assert "Never rerun `scripts/run_modal_pilot.sh`, `modal run`" in source
    assert "execution_receipt_count" in source
    assert "lower bound" in source
    assert "up to four days after deletion" in source
    assert "same-UID" in source
    assert "HARD BUDGET VIOLATION" in source
    assert "run_guarded_uv()" in source
    for required in (
        "/usr/bin/env -i",
        "umask 077",
        "HOME=/home/ubuntu",
        "PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "MODAL_CONFIG_PATH=/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml",
        "MODAL_PROFILE=ratemem-pilot",
        "MODAL_ENVIRONMENT=main",
        "/home/ubuntu/.local/bin/uv run --extra modal",
        "run_guarded_uv modal volume get",
        "run_guarded_uv ratemem-pilot security-scan",
        "run_guarded_uv ratemem-pilot validate-artifact",
        "run_guarded_uv ratemem-pilot validate-modal-config empty",
        "run_guarded_uv ratemem-pilot validate-modal-config configured",
        "workspace_spend_limit_usd",
        "Workspace spend limit** to",
        "USD 0.00",
        "run_guarded_uv ratemem-pilot security-scan src tests configs schemas scripts",
        "run_guarded_uv ratemem-pilot attest-volume-absence",
        "run_guarded_uv ratemem-pilot reconcile",
        "four full days after volume deletion",
        "A changed reading resets the four-day window",
        "after provisioning, refresh again and require exact profile, workspace, environment",
        "fresh_usage + phase_bound <= USD 27.00",
        "leaves the permit unsubmitted and issues no remote call",
        "semantically invalid raw execution receipt",
        "RATEMEM_FORENSIC_ROOT=artifacts/pilot/execution-receipts",
        '"execution-receipts/${RATEMEM_ATTEMPT_ID}" "${RATEMEM_FORENSIC_ROOT}"',
        "run_guarded_uv ratemem-pilot validate-forensic-receipts",
        "run_guarded_uv ratemem-pilot record-incident",
        "run_guarded_uv ratemem-pilot attest-incident-volume-absence",
        "run_guarded_uv ratemem-pilot reconcile-incident",
        "incident.json",
        "USD 0.00 delta",
        "restarts the full four-day window",
        "maturity check passes, the ledger is reconciled and closed",
    ):
        assert required in source
    assert "chmod 600" not in source
    assert "/home/ubuntu/.modal.toml" not in source
    assert "server_url" in source
    assert "MODAL_PROFILE=ratemem-pilot uv run" not in source
    recovery_get = source.index('"attempts/${RATEMEM_ATTEMPT_ID}" artifacts/pilot')
    recovery_scan = source.index('ratemem-pilot security-scan "${RATEMEM_DESTINATION}"')
    recovery_validate = source.index(
        'ratemem-pilot validate-artifact "${RATEMEM_DESTINATION}/attempt.pending.json"'
    )
    assert recovery_get < recovery_scan < recovery_validate
    forensic_root = source.index("RATEMEM_FORENSIC_ROOT=artifacts/pilot/execution-receipts")
    forensic_get = source.index(
        '"execution-receipts/${RATEMEM_ATTEMPT_ID}" "${RATEMEM_FORENSIC_ROOT}"'
    )
    forensic_scan = source.index('ratemem-pilot security-scan "${RATEMEM_FORENSIC_DESTINATION}"')
    forensic_validate = source.index("ratemem-pilot validate-forensic-receipts")
    first_delete = source.index("modal volume delete")
    assert forensic_root < forensic_get < forensic_scan < forensic_validate < first_delete
    delete_lines = [line for line in source.splitlines() if "modal volume delete" in line]
    assert len(delete_lines) == 4
    assert all("--allow-missing" in line and "--yes" in line for line in delete_lines)
