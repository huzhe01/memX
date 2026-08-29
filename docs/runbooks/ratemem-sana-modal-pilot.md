# RateMem SANA Modal pilot runbook

This runbook authorizes one engineering-only SANA pilot. It does not authorize a scientific
evaluation, a rerun, account rotation, another GPU type, a detached job, or a deployment. The
selected first account is identified only by its user-facing label: `xinming-hu-rd`. Never paste a
token value into a command, environment variable, file, issue, log, or this repository.

Do not run the paid launch until the free Task 14 gate has passed and the author has reviewed its
output. In particular, never run `scripts/run_modal_pilot.sh` while implementing or reviewing this
code.

## 0. Use one sanitized command wrapper

Run all commands from the repository root. Define this function once in the current shell. It is
the only way this runbook invokes `uv`, Modal, or `ratemem-pilot`: it drops inherited token,
profile, endpoint, and config variables and binds every command to the intended local config,
profile, and environment.

```bash
run_guarded_uv() {
  umask 077
  /usr/bin/env -i \
    HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
    LANG=C.UTF-8 \
    MODAL_CONFIG_PATH=/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml \
    MODAL_PROFILE=ratemem-pilot \
    MODAL_ENVIRONMENT=main \
    /home/ubuntu/.local/bin/uv run --extra modal "$@"
}
```

Do not replace it with a shell alias, export a Modal token, activate a global profile, or use an
inherited `MODAL_CONFIG_PATH`. A command that cannot run through this wrapper is a hard stop.

## 1. Create private local locations

From the repository root, run:

```bash
run_guarded_uv python -c 'from pathlib import Path; from ratemem.pilot.private_io import ensure_private_directory; [ensure_private_directory(Path(value)) for value in ("/home/ubuntu/.local/share/ratemem/modal", "/home/ubuntu/.local/state/ratemem", "artifacts/pilot")]'
```

Expected: exit 0. Each named directory is owned by the current user and has mode `0700`. The helper
does not repair an existing symlink, foreign-owned path, or permissive directory; any such finding
is a hard stop.

Confirm the state without printing file contents:

```bash
stat -c '%a %U %n' /home/ubuntu/.local/share/ratemem/modal /home/ubuntu/.local/state/ratemem artifacts/pilot
```

Expected: three `700` entries owned by the current user.

## 2. Establish the USD 28 hard outer stop before CLI authentication

Do not use the 2026-08-29 dashboard: the calendar-month guard blocks this launch until 2026-09-01
UTC. On or after that date, in a browser already authenticated to the intended `xinming-hu-rd`
account, select the one intended workspace. Open **Usage & Billing**, explicitly set the custom
**Workspace usage budget** to exactly **USD 28.00** and the custom **Workspace spend limit** to
exactly **USD 0.00**, save both settings, and note the current billing-cycle pre-credit metered
usage. Save a new screenshot that shows the same workspace identity and both settings at:

```text
/home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png
```

This screenshot must exist before configuring the CLI token. If either control is absent, defaulted
rather than explicitly custom, cannot be saved, is ambiguous, or cannot be shown in the bound
screenshot, stop. Do not authenticate the CLI, create volumes, build an image, or launch work.

Make the screenshot private without following a symlink:

```bash
run_guarded_uv python -c 'import os, stat; path="/home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png"; fd=os.open(path, os.O_RDONLY | os.O_NOFOLLOW); metadata=os.fstat(fd); assert stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid() and metadata.st_nlink == 1; os.fchmod(fd, 0o600); os.fsync(fd); os.close(fd)'
```

Then inspect metadata only:

```bash
stat -c '%a %U %n' /home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png
```

Expected: mode `600`, owned by the current user. Do not continue with a symlink, hard link,
foreign-owned file, or unexpected parent directory.

The verified USD 28 Workspace usage budget is the hard pre-credit outer stop. The verified USD 0
Workspace spend limit prevents post-credit out-of-pocket spend if credits are exhausted or do not
apply; it does not reduce credit-covered work below the usage budget. The local USD 27 ledger is a
conservative pre-launch admission bound, not a hard cap on realized spend. Free credits never
replace pre-credit metered usage in either usage check.

## 3. Configure exactly one isolated profile through hidden prompts

Only after Step 2, run:

First create or validate the exact config file without following a link. This must complete before
any credential is entered:

```bash
run_guarded_uv python - <<'PY'
import os
import stat

path = "/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml"
flags = os.O_RDWR | os.O_NOFOLLOW
try:
    descriptor = os.open(path, flags)
except FileNotFoundError:
    descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
metadata = os.fstat(descriptor)
assert stat.S_ISREG(metadata.st_mode)
assert metadata.st_uid == os.getuid()
assert metadata.st_nlink == 1
assert stat.S_IMODE(metadata.st_mode) == 0o600
assert metadata.st_size == 0
os.close(descriptor)
PY
run_guarded_uv ratemem-pilot validate-modal-config empty
```

An existing nonempty file, table, profile, symlink, hard link, foreign-owned file, or mode other
than `0600` is a hard stop. This dedicated config must be exactly empty before the hidden prompt,
so no stale `server_url` can receive the submitted token during verification. Do not repair an
existing file with `chmod`, because that could modify an attacker-selected target.

Only then run:

```bash
run_guarded_uv modal token set --profile ratemem-pilot --no-activate --verify
```

At Modal's hidden interactive prompts, enter the token ID and token secret for the selected
`xinming-hu-rd` account. Expected: verification succeeds. On a fresh config Modal 1.5.4 marks this
profile active even with `--no-activate`; that activation exists only inside the dedicated pilot
config and does not affect the normal global Modal config. Do not put credentials in command-line
options and do not redirect output. If verification fails, stop and ask the author; do not try
another supplied account and do not auto-rotate credentials.

An explicitly author-approved browser-login alternative is:

```bash
run_guarded_uv modal token new --profile ratemem-pilot --no-activate --verify
```

It is not an automatic fallback.

Secure and inspect the local configuration without printing it:

```bash
run_guarded_uv ratemem-pilot validate-modal-config configured
run_guarded_uv modal profile current
run_guarded_uv modal profile list --json
```

Expected: config validation prints `PASS modal_config_state=configured`, the selected profile is
`ratemem-pilot`, and its workspace is exactly the workspace in the screenshot. The config must
contain only `[ratemem-pilot]` and exact `token_id`, `token_secret`, and `active=true` fields; a
`server_url`, extra field, or extra profile is a hard stop. Guarded code uses exactly
`/home/ubuntu/.local/share/ratemem/modal/ratemem-pilot.toml`; it rejects a symlink, non-owner file,
link count other than one, or mode other than `0600`.

## 4. Create the structured operator evidence and workspace attestation

Run:

```bash
run_guarded_uv ratemem-pilot attest-workspace --evidence /home/ubuntu/.local/share/ratemem/modal/usage-budget-28.png
```

At the prompts:

1. type the exact lowercase workspace slug shown in the screenshot;
2. type `28.00`;
3. type `0.00`;
4. type exactly `I confirm the Modal dashboard Workspace usage budget is USD 28.00 before credits and the Workspace spend limit is USD 0.00 after credits.`

Expected output:

```text
PASS workspace=<exact-workspace> usage_budget_usd=28.00 spend_limit_usd=0.00
```

The command creates a canonical mode-0600 structured evidence file beside the screenshot named
`usage-budget-28.<screenshot-hash-prefix>.operator-attestation.json`. It binds the workspace, profile, environment, exact
usage budget, exact `workspace_spend_limit_usd="0.00"`, screenshot absolute path, screenshot SHA-256, screenshot UTC
modification time, and exact confirmation statement. That JSON—not the screenshot alone—is passed into the billing snapshot.
The published `artifacts/pilot/workspace-attestation.json` is also mode `0600` and ignored by Git.
Any profile/workspace mismatch, stale evidence, billing API denial, malformed rates, or dashboard
hash change is a hard stop; there is no estimate fallback.

## 5. Run the complete free gate

Run all commands from a clean repository root:

```bash
run_guarded_uv pytest -q -m 'not paid_modal and not real_sana and not cuda'
run_guarded_uv ruff check src tests
run_guarded_uv mypy src/ratemem
run_guarded_uv ratemem-pilot security-scan src tests configs schemas scripts
git status --short
run_guarded_uv python - <<'PY'
from pathlib import Path

for path in (
    Path("/home/ubuntu/.local/state/ratemem/modal-pilot-slot.json"),
    Path("/home/ubuntu/.local/state/ratemem/modal-pilot-submitted.json"),
):
    assert not path.exists() and not path.is_symlink(), path
pilot = Path("artifacts/pilot")
assert pilot.is_dir() and not pilot.is_symlink()
assert {entry.name for entry in pilot.iterdir()} == {"workspace-attestation.json"}
PY
```

Expected: tests and static checks exit 0, the credential gate prints `PASS`, and after Task 13 is
committed `git status --short` is empty. The final Python gate must also exit 0: ignored files are
invisible to ordinary Git status, so only the current workspace attestation may exist under
`artifacts/pilot`; no prior slot, submission receipt, permit, ledger, provision intent/receipt,
reconciliation, incident, UUID attempt, or external-receipt directory may exist.
Preflight rejects every tracked, staged, or untracked source change. Generated pilot state is ignored
and does not alter the source identity.

Review only the key names and permissions of the attestation; do not dump configuration or the
environment. W&B is disabled. The pinned public SANA, DINOv2, and Subjects200K resources need no
Hugging Face token. No unredacted environment or configuration dump is permitted.

## 6. Execute the one authorized paid command once

Run exactly once:

```bash
scripts/run_modal_pilot.sh
```

The script uses `umask 077` and the same sanitized `env -i` wrapper throughout. Its ordered
fail-closed behavior is:

1. revalidate the dedicated config as the exact single profile with no endpoint override, then
   refresh profile, workspace, both dashboard limits, current pre-credit usage, and exact Modal
   rates;
2. require a clean HEAD and hash the exact commit, empty diff, and locked pilot config bundle;
3. conservatively compute one-L40S, CPU, RAM, startup, timeout, and storage cost, at most USD 21;
4. burn one immutable UUIDv7 global slot, append one USD 27 ledger reservation, and create one
   launch permit whose `pending_worst_case_usd` equals its first-pilot phase bound;
5. validate the unsubmitted permit and provision only `ratemem-sana-cache` and
   `ratemem-pilot-artifacts` in environment `main`; the dedicated workspace must initially have no
   volumes at all. A create-only intent records that empty list before mutation, and a second
   create-only receipt binds the verified exact two-volume set to this attempt, permit, slot, and
   workspace. Unknown volumes or mismatched receipts are a hard stop;
6. securely validate the permit again and hold its attempt ID before consumption;
7. after provisioning, refresh again and require exact profile, workspace, environment, rate
   map/hash, monotonic pre-credit usage, and `fresh_usage + phase_bound <= USD 27.00` before creating the
   immutable submission receipt, consuming the permit, or calling `.remote()`; any drift is a hard
   stop that leaves the permit unsubmitted and issues no remote call;
8. refuse an existing or symlinked local UUID destination before paid submission, then create the
   immutable local submission receipt and issue exactly one synchronous
   `modal run -m ratemem.pilot.modal_app`, requesting one L40S;
9. download only remote
   `attempts/<attempt-id>` into the existing `artifacts/pilot` parent, scan it for credential
   material before parsing it, and then validate every artifact/checksum/receipt/launch binding.

Preflight also requires at least ten full days before the next UTC calendar-month boundary, before
burning the slot. This covers the run, delayed charges, and a restarted four-day stability window.
On 2026-08-29 it must print `PENDING` and do nothing irreversible; the earliest eligible fresh
attestation is 2026-09-01 UTC.

Expected local one-shot evidence:

```text
/home/ubuntu/.local/state/ratemem/modal-pilot-slot.json
/home/ubuntu/.local/state/ratemem/modal-pilot-submitted.json
artifacts/pilot/launch-permit.json
artifacts/pilot/cost-ledger.jsonl
```

Each JSON/JSONL file is owner-only and immutable or append-only under its protocol. The slot, permit,
submission receipt, artifact, and every infrastructure receipt bind the same attempt, workspace,
commit-derived source hash, and evidence hashes. A serial or concurrent second launch fails before
`.remote()`. There is no deployment, schedule, detached execution, fan-out, fallback GPU, or retry
loop.

Modal can reschedule infrastructure even with user-code `retries=0` and `max_containers=1`.
Therefore `execution_receipt_count` is a lower bound on physical executions, not proof of exactly
one container. Inspect it and reconcile actual pre-credit metered usage.

After the script returns—whether normally or into recovery—establish these parent-shell variables;
the script's internal variables do not propagate to this shell:

```bash
RATEMEM_ATTEMPT_ID="$(run_guarded_uv ratemem-pilot permit-field attempt_id)"
[[ "${RATEMEM_ATTEMPT_ID}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$ ]]
RATEMEM_DESTINATION="artifacts/pilot/${RATEMEM_ATTEMPT_ID}"
```

## 7. Recover safely after client or infrastructure failure

Never rerun `scripts/run_modal_pilot.sh`, `modal run`, or another account after any failure once the
slot exists. If the 15-minute evidence has expired, save a newly captured screenshot under a new
private filename (for example, one with a UTC timestamp), and repeat Step 4 with that new `--evidence`
path. Do not overwrite or delete the earlier evidence. Recover the bound attempt ID with the
unconditional full slot/permit/receipt validation block at the end of Step 6.

If the remote call completed but the client stopped before download, verify that the UUID destination
does not exist, then run only these free/read-only recovery steps:

```bash
test -d artifacts/pilot
test ! -L artifacts/pilot
test ! -e "${RATEMEM_DESTINATION}"
test ! -L "${RATEMEM_DESTINATION}"
run_guarded_uv modal volume get --env main ratemem-pilot-artifacts "attempts/${RATEMEM_ATTEMPT_ID}" artifacts/pilot
run_guarded_uv ratemem-pilot security-scan "${RATEMEM_DESTINATION}"
run_guarded_uv ratemem-pilot validate-artifact "${RATEMEM_DESTINATION}/attempt.pending.json"
```

These commands contain no paid invocation. The credential scan must pass before any artifact parser,
editor, previewer, or metrics reader.

Before deleting the artifact volume, always preserve the external infrastructure lower-bound
receipt directory. This is mandatory even when the attempt artifact exists: an exception artifact
may contain only a safe semantic-invalid marker whose raw source exists solely in this directory.
Download into an existing, separate mode-0700 parent so Modal preserves every relative path:

```bash
run_guarded_uv python -c 'from pathlib import Path; from ratemem.pilot.private_io import ensure_private_directory; ensure_private_directory(Path("artifacts/pilot/execution-receipts"))'
RATEMEM_FORENSIC_ROOT=artifacts/pilot/execution-receipts
RATEMEM_FORENSIC_DESTINATION="${RATEMEM_FORENSIC_ROOT}/${RATEMEM_ATTEMPT_ID}"
test ! -e "${RATEMEM_FORENSIC_DESTINATION}"
test ! -L "${RATEMEM_FORENSIC_DESTINATION}"
run_guarded_uv modal volume get --env main ratemem-pilot-artifacts "execution-receipts/${RATEMEM_ATTEMPT_ID}" "${RATEMEM_FORENSIC_ROOT}"
run_guarded_uv ratemem-pilot security-scan "${RATEMEM_FORENSIC_DESTINATION}"
run_guarded_uv ratemem-pilot validate-forensic-receipts "${RATEMEM_DESTINATION}/attempt.pending.json" "${RATEMEM_FORENSIC_DESTINATION}"
```

The validator securely sorts the exact raw receipt filenames and reconstructs the runner snapshot
as each file's exact content followed by `\n`. For a normal artifact its aggregate byte count and
SHA-256 must equal local `execution-receipts.jsonl`; for a semantic-invalid exception they must equal
the exact external marker fields. It publishes a create-only private per-file and aggregate manifest.
If a valid attempt artifact exists but the external directory is absent, malformed, symlinked, fails
the credential scan, or fails the aggregate binding, the normal attempt path is invalid. Preserve
whatever local bytes and scan diagnostics were obtained, do not parse or display material that
failed the credential scan, and enter the incident cleanup immediately below. Recording the incident
before deletion preserves the cost identity; indefinite remote storage retention is not evidence.

The normal attempt artifact can represent a semantically invalid raw execution receipt only with a
strict non-secret exception marker and matching metrics diagnostic. The raw receipt remains in the
external `execution-receipts/<attempt-id>` directory; scan that directory before viewing or parsing
it. The marker is accepted only for this exception path and never makes the raw receipt trustworthy.

If the paid command never submitted, or the attempt/external receipt evidence is absent or cannot be
validated, do not retain paid storage indefinitely and do not invent an attempt artifact. Record the
permit-bound incident first, then perform only idempotent cleanup:

```bash
run_guarded_uv ratemem-pilot record-incident
RATEMEM_INCIDENT="artifacts/pilot/incidents/${RATEMEM_ATTEMPT_ID}/incident.pending.json"
run_guarded_uv modal volume delete --env main --allow-missing --yes ratemem-sana-cache
run_guarded_uv modal volume delete --env main --allow-missing --yes ratemem-pilot-artifacts
run_guarded_uv ratemem-pilot attest-incident-volume-absence "${RATEMEM_INCIDENT}"
```

Refresh Step 4 evidence and, only after the same pre-credit reading has remained stable for four
full days after this incident absence, run:

```bash
run_guarded_uv ratemem-pilot reconcile-incident "${RATEMEM_INCIDENT}"
```

This path closes the exact ledger reservation and creates only create-only `incident.json`; it never
creates `attempt.json` or claims a scientific result. A pre-submit failure may mature at an exact
USD 0.00 delta. A receipt remains an infrastructure lower bound and may miss a failure before its
commit.

If either volume appears after a normal or incident absence, the command creates a durable record
bound to the preceding absence. Normal `attempt.json` publication is then permanently invalid.
After a normal attempt, run `record-incident` once; for an already-recorded incident, keep its
existing immutable pending record and do not call `record-incident` again. Then delete both volumes
again and call `attest-incident-volume-absence`; the new absence is bound to the reappearance record
and restarts the full four-day window. An earlier absence/candidate can never be reused. Only the
cost-only incident path can then close the ledger.

If reconciliation appended successfully but the process stopped before publishing `attempt.json`,
rerun only the same free `reconcile` command in Step 9. It reads the durable ledger result and creates
or exact-validates the create-only final file. It never submits work. Fresh metered usage must equal
the durable `known_usage_after`; a larger value could be delayed same-attempt spend and is not
silently folded into a stale final artifact.

The one-shot protocol prevents cooperating concurrent processes from bypassing the marker, even if
the local state directory is renamed while they run. Like every local-only protocol, it cannot prove
that a same-UID actor did not wait for all processes to exit and then delete or replace every local
record. Such deletion never authorizes another launch; escalate to the author.

## 8. Retain local evidence, delete storage, and attest absence

Before deleting remote storage, retain the local validated attempt directory, `attempt.pending.json`,
`checksums.sha256`, execution-receipts JSONL, slot/permit/submission evidence, and any trainable
checkpoint. Confirm the credential scan passed and record `execution_receipt_count`. Do not delete
local evidence while the ledger remains open. The external receipt directory and its local hash
manifest from Step 7 must also be present before deleting `ratemem-pilot-artifacts`.

Only after download, security scan, and validation, delete both named volumes and attest their live
absence:

```bash
run_guarded_uv modal volume delete --env main --allow-missing --yes ratemem-sana-cache
run_guarded_uv modal volume delete --env main --allow-missing --yes ratemem-pilot-artifacts
run_guarded_uv ratemem-pilot attest-volume-absence "${RATEMEM_DESTINATION}/attempt.pending.json"
```

Deleting a volume is irreversible, although the public frozen checkpoint can be restored from its
pinned revision. The pinned Modal 1.5.4 `--allow-missing` behavior makes only these two cleanup
deletes idempotent; it never authorizes a repeated launch. The absence command live-lists `main`, confirms both required volumes are absent,
records the current pre-credit billing reading, and creates the durable settlement candidate. If a
volume remains or the attestation fails, leave the ledger reservation open and ask the author; do
not retry the paid script. Modal storage charges may remain visible for up to four days after deletion.

## 9. Reconcile only after four days of equal billing observations

The first successful absence attestation is not settlement. Refresh dashboard evidence through Step
4 and run this same free command only after the pre-credit billing reading has stayed exactly equal
for at least four full days after volume deletion:

```bash
run_guarded_uv ratemem-pilot reconcile "${RATEMEM_DESTINATION}/attempt.pending.json"
```

Expected outcomes:

- exit code 3 with `PENDING: ... another launch is forbidden` while billing is absent, changing, or
  has not stayed equal for four days. A changed reading resets the four-day window; refresh evidence
  and rerun only this free command later.
- `PASS reconciled_cost_usd=<exact-delta>` and a validated create-only `attempt.json` when the
  mature reading agrees with the durable ledger.
- `HARD BUDGET VIOLATION` if a durable observation exceeds USD 28.00. Before maturity, the ledger
  stays open while the create-only violation observation remains durable. Once the four-day
  maturity check passes, the ledger is reconciled and closed at the stable actual delta, but no
  compliant `attempt.json` is published. Never rotate accounts or launch again.

Reconciliation uses current pre-credit usage minus the permit's pre-launch value. Realized cost may
truthfully exceed USD 21; it is still recorded. A later fresh reading that differs from a durable
reconciliation is an error, not permission to rewrite the result. On OOM or another incomplete
result, retain the pending artifact and reconcile billing. Do not change to an A100 or other GPU.

The unallocated USD 6 safety buffer authorizes no rerun, no account rotation, and no additional
experiment. The selected workspace must have no other volumes, jobs, or concurrent billing activity
throughout this protocol; any unrelated usage invalidates attribution and requires incident review.
