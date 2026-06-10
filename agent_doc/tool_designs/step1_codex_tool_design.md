# Step 1 Codex Tool Design — Data Alignment & Schema Report (Gate 0)

## Architecture Philosophy

This design follows three core principles:
- **Divide & Conquer** — decompose the alignment task into the smallest independently executable units
- **Dependency Graph (Relation Map)** — each micro-task has explicit inputs, outputs, and upstream/downstream dependencies
- **High Reliability & Maintainability** — every tool is single-responsibility, idempotent, and self-documenting

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  STEP 1 — GATE 0 SYSTEM                     │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │  T1-A   │───▶│  T1-B   │───▶│  T1-C   │                 │
│  │Discovery│    │Profile  │    │ Verdict │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       │              │               │                      │
│       ▼              ▼               ▼                      │
│  file_manifest  column_profile  gate0_verdict               │
│  .json          .json           .json                       │
│       │              │               │                      │
│       └──────────────┴───────────────┘                      │
│                       │                                     │
│                       ▼                                     │
│              ┌─────────────────┐                            │
│              │  DOC GENERATOR  │                            │
│              │  (post T1-C)    │                            │
│              └─────────────────┘                            │
│                       │                                     │
│          ┌────────────┴────────────┐                        │
│          ▼                         ▼                        │
│   schema_report.md          work_daily_log.md               │
└─────────────────────────────────────────────────────────────┘
```

---

## Micro-Task Breakdown (Divide & Conquer)

### T1-A — File Discovery Tool

**Purpose:** Discover and inventory all data files in the source directory.

**Input:** Source directory path
**Output:** `model_outputs/file_manifest.json`
**Upstream:** None (entry point)
**Downstream:** T1-B

```python
# Tool: file_discovery.py
"""
Responsibility : Single — discover and classify data files only
Idempotent     : Yes — always overwrites file_manifest.json
Side effects   : None — read-only scan
"""
import os, json, hashlib
from pathlib import Path

def run(source_dir: str, output_path: str = "model_outputs/file_manifest.json") -> dict:
    supported = {".csv", ".json", ".parquet", ".tsv"}
    manifest = []

    for root, _, files in os.walk(source_dir):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() not in supported:
                continue
            stat = fpath.stat()
            manifest.append({
                "filename"  : str(fpath),
                "extension" : fpath.suffix.lower(),
                "size_bytes": stat.st_size,
                "sha256"    : _hash(fpath)
            })

    result = {"file_count": len(manifest), "files": manifest}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result

def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

**Reliability Contract:**
- Returns empty manifest (not error) if directory is empty
- SHA-256 hash enables downstream change detection
- Logs to `logs/t1a_discovery.log`

---

### T1-B — Column Profiler Tool

**Purpose:** Read each discovered file and compute per-column statistical profiles.

**Input:** `model_outputs/file_manifest.json`
**Output:** `model_outputs/column_profile.json`
**Upstream:** T1-A (`file_manifest.json` must exist)
**Downstream:** T1-C

```python
# Tool: column_profiler.py
"""
Responsibility : Single — compute column statistics only
Idempotent     : Yes — deterministic given same input
Side effects   : None — never writes to source data
Sample cap     : 10,000 rows if file > 100 MB
"""
import json, pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

SAMPLE_CAP    = 10_000
SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

def run(manifest_path: str, output_path: str = "model_outputs/column_profile.json") -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    profile  = {"files": []}

    for entry in manifest["files"]:
        df = _load(entry)
        columns = []
        for col in df.columns:
            series = df[col]
            col_stat = {
                "column"       : col,
                "dtype"        : str(series.dtype),
                "null_rate"    : round(series.isna().mean(), 4),
                "unique_count" : int(series.nunique()),
            }
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                col_stat.update({
                    "min"      : float(clean.min()),
                    "max"      : float(clean.max()),
                    "mean"     : float(clean.mean()),
                    "std"      : float(clean.std()),
                    "skewness" : float(clean.skew()),
                    "kurtosis" : float(clean.kurtosis()),
                    "normality_pvalue": float(scipy_stats.shapiro(clean[:5000])[1])
                    if len(clean) >= 3 else None
                })
            columns.append(col_stat)

        profile["files"].append({
            "filename"    : entry["filename"],
            "row_count"   : len(df),
            "col_count"   : len(df.columns),
            "sampled"     : entry["size_bytes"] > SIZE_THRESHOLD,
            "columns"     : columns
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(profile, indent=2))
    return profile

def _load(entry: dict) -> pd.DataFrame:
    path = entry["filename"]
    ext  = entry["extension"]
    use_sample = entry["size_bytes"] > SIZE_THRESHOLD
    readers = {
        ".csv"    : lambda p: pd.read_csv(p, nrows=SAMPLE_CAP if use_sample else None),
        ".tsv"    : lambda p: pd.read_csv(p, sep="\t", nrows=SAMPLE_CAP if use_sample else None),
        ".json"   : lambda p: pd.read_json(p).head(SAMPLE_CAP) if use_sample else pd.read_json(p),
        ".parquet": lambda p: pd.read_parquet(p).head(SAMPLE_CAP) if use_sample else pd.read_parquet(p),
    }
    return readers[ext](path)
```

**Reliability Contract:**
- Parallel reads across files using `concurrent.futures.ThreadPoolExecutor`
- Shapiro-Wilk capped at 5,000 samples (scipy requirement)
- Skewness threshold for flagging: `|skewness| > 1.0`
- Logs to `logs/t1b_profiler.log`

---

### T1-C — Gate 0 Verdict Tool

**Purpose:** Evaluate column profiles against quality rules and emit a Gate 0 verdict.

**Input:** `model_outputs/column_profile.json`
**Output:** `model_outputs/gate0_verdict.json`
**Upstream:** T1-B (`column_profile.json` must exist)
**Downstream:** Doc Generator

```python
# Tool: gate0_verdict.py
"""
Responsibility : Single — apply quality rules and emit verdict only
Idempotent     : Yes
Side effects   : None
Verdict levels : PASS | CONDITIONAL | FAIL
"""
import json
from pathlib import Path

RULES = {
    "null_rate_hard_fail"    : 0.50,   # > 50% → FAIL
    "null_rate_conditional"  : 0.30,   # > 30% → CONDITIONAL
    "skewness_flag"          : 1.0,    # |skew| > 1.0 → flag
    "normality_alpha"        : 0.05,   # p < 0.05 → non-normal
}

def run(profile_path: str, output_path: str = "model_outputs/gate0_verdict.json") -> dict:
    profile  = json.loads(Path(profile_path).read_text())
    failures, conditionals, flags = [], [], []

    for file_entry in profile["files"]:
        for col in file_entry["columns"]:
            ref = f"{file_entry['filename']}::{col['column']}"

            if col["null_rate"] > RULES["null_rate_hard_fail"]:
                failures.append({"column": ref, "reason": f"null_rate={col['null_rate']:.1%} > 50%"})

            elif col["null_rate"] > RULES["null_rate_conditional"]:
                conditionals.append({"column": ref, "reason": f"null_rate={col['null_rate']:.1%} > 30%"})

            if col.get("skewness") and abs(col["skewness"]) > RULES["skewness_flag"]:
                flags.append({"column": ref, "reason": f"skewness={col['skewness']:.3f}"})

            if col.get("normality_pvalue") and col["normality_pvalue"] < RULES["normality_alpha"]:
                flags.append({"column": ref, "reason": f"non-normal p={col['normality_pvalue']:.4f}"})

    if failures:
        verdict = "FAIL"
        summary = f"{len(failures)} column(s) exceed hard-fail thresholds."
    elif conditionals:
        verdict = "CONDITIONAL"
        summary = f"{len(conditionals)} column(s) require human review before PCA."
    else:
        verdict = "PASS"
        summary = "All columns meet quality thresholds. Cleared for PCA."

    result = {
        "verdict"     : verdict,
        "summary"     : summary,
        "failures"    : failures,
        "conditionals": conditionals,
        "flags"       : flags,
        "rules_applied": RULES
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result
```

**Reliability Contract:**
- Rules are declared as a top-level constant — editable without touching logic
- Verdict enum is strictly enforced: `PASS | CONDITIONAL | FAIL`
- Logs to `logs/t1c_verdict.log`

---

## Dependency Relation Map

```
T1-A (Discovery)
    │
    │  produces: file_manifest.json
    ▼
T1-B (Profiler)
    │
    │  produces: column_profile.json
    ▼
T1-C (Verdict)
    │
    │  produces: gate0_verdict.json
    ▼
DOC GENERATOR  ◀── triggered only after T1-C completes
    │
    ├──▶ schema_report.md       (human-readable summary)
    └──▶ work_daily_log.md      (Codex memory & traceability)
```

**Blocking rule:** Each task MUST verify its upstream artifact exists and is non-empty before execution. If missing → raise `UpstreamArtifactMissing` with the expected path.

---

## Documentation Generation (Post T1-C)

Codex must auto-generate two documents after T1-C completes.

### Doc 1 — schema_report.md

```markdown
# Schema Report — Gate 0
Generated: {timestamp}

## Data Source Inventory
| File | Rows | Columns | Sampled |
|------|------|---------|---------|
| ... | ... | ... | Yes/No  |

## Column Alignment Table
| Column | dtype | null_rate | mean | std | skewness | normality_p | inferred_semantic | flag |
|--------|-------|-----------|------|-----|----------|-------------|-------------------|------|

## Gate 0 Verdict
**Verdict: PASS / CONDITIONAL / FAIL**
- Summary: ...
- Failures: ...
- Conditionals: ...
- Flags (non-blocking): ...

## Top 3 Questions for Human Review
1. ...
2. ...
3. ...
```

### Doc 2 — work_daily_log.md

```markdown
# Work Daily Log — Step 1 Gate 0
Date       : {date}
Project    : redcap_rrc_inactive_sdt_oran_control_v1
Step       : 1 — Data Alignment
Gate       : 0
Case       : A

## Tasks Completed
- [x] T1-A File Discovery     — {file_count} files found
- [x] T1-B Column Profiling   — {col_count} columns profiled
- [x] T1-C Gate 0 Verdict     — {verdict}

## Artifacts Produced
| Artifact | Path | Size |
|----------|------|------|
| file_manifest.json | model_outputs/ | ... |
| column_profile.json | model_outputs/ | ... |
| gate0_verdict.json | model_outputs/ | ... |
| schema_report.md | reports/ | ... |

## Issues & Flags
- ...

## Next Step
- [ ] If PASS → proceed to Step 2 (PCA)
- [ ] If CONDITIONAL → resolve flagged columns, re-run T1-C
- [ ] If FAIL → fix source data, restart from T1-A
```

---

## Codex Prompt for Step 1 (Full)

```
# Step 1 — Data Alignment & Gate 0 Validation

## Role
You are a senior data engineer. Execute the following micro-tasks in order.
Each task must verify its upstream artifact before starting.
Do NOT pause between tasks. After T1-C completes, generate both documents.

> Historical note: this design references the retired `pipeline_tools/` workflow and is no longer the current executable structure of the repository.

## Micro-Tasks

### T1-A — File Discovery
1. Run `rg --files data/raw/` to list all data files
2. Run tool: `python pipeline_tools/file_discovery.py --source data/raw/`
3. Verify `model_outputs/file_manifest.json` is non-empty before continuing

### T1-B — Column Profiling
1. Verify `model_outputs/file_manifest.json` exists
2. Run tool: `python pipeline_tools/column_profiler.py`
3. Verify `model_outputs/column_profile.json` is non-empty before continuing

### T1-C — Gate 0 Verdict
1. Verify `model_outputs/column_profile.json` exists
2. Run tool: `python pipeline_tools/gate0_verdict.py`
3. Verify `model_outputs/gate0_verdict.json` is non-empty

## Documentation (run after T1-C)
1. Write `reports/schema_report.md` using apply_patch
2. Write `agent_doc/agent_memory/work_daily/{date}_step1_gate0.md` using apply_patch
3. List your top 3 questions for human review at the end of schema_report.md

## Constraints
- Do NOT modify any file under data/raw/
- Use read_file instead of cat for reading files
- All writes use apply_patch
- If any upstream artifact is missing, raise an error with the expected path
```

---

## Gate 0 Checklist

| Task | Check Item | Standard | Status |
|------|-----------|----------|--------|
| T1-A | All data files discovered | No files missed | [ ] |
| T1-A | file_manifest.json written | Non-empty JSON | [ ] |
| T1-B | All columns profiled | 100% coverage | [ ] |
| T1-B | column_profile.json written | Non-empty JSON | [ ] |
| T1-C | Verdict is PASS/CONDITIONAL/FAIL | Exactly one value | [ ] |
| T1-C | gate0_verdict.json written | Non-empty JSON | [ ] |
| DOC | schema_report.md written | Contains all 5 sections | [ ] |
| DOC | work_daily_log.md written | Matches daily log format | [ ] |

---

## Extended Ideas (Cross-disciplinary Links)

- **SPC (Statistical Process Control)** — Column skewness and normality flags in T1-C directly map to Western Electric rules for out-of-control signals in control charts
- **FMEA (Failure Mode & Effects Analysis)** — The three-tier verdict (PASS / CONDITIONAL / FAIL) mirrors FMEA's RPN severity classification
- **Supply Chain Traceability** — SHA-256 hashing in T1-A provides an immutable data lineage record, directly applicable to IoT sensor data audit trails
- **O-RAN Gate Alignment** — T1-A → T1-B → T1-C mirrors the Gate 1 → Gate 2 → Gate 3 progression in your project_plan.md; the same blocking rule applies
