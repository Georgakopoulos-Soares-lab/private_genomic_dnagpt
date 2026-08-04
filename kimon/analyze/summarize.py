#!/usr/bin/env python3
"""Aggregate kimon run logs and DERIVE speed improvements across configs.

Reads every result JSON under kimon/logs/<config>/<tag>/*.json (any infra --
TACC, a production GPU box, whatever produced them), then prints and writes:
  - a per-run table (latency + accuracy), and
  - derived speedups: diagcache (config1 vs config3), block->e2e scaling, and
    the config2 Q/K/V-projection 2-GPU concurrency benefit.

Pure standard library. Usage:
    python3 kimon/analyze/summarize.py [--logs kimon/logs] [--json]
Writes kimon/logs/summary.md (+ summary.json with --json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---- the timing fields our drivers emit (seconds) ----
ENC = "encrypted_evaluation"  # headline wall of the encrypted eval
SRV = "server_linear_algebra_seconds"  # server-side linear algebra
CLI = "client_boundary_seconds_total"  # client refresh/boundary time


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def classify(config: str, tag: str) -> str:
    """block | e2e | shard-* , from the config folder + tag."""
    t = tag.lower()
    if "qkvshard" in t or config == "config2_sharding":
        if "reader_querykey" in t or "reader_query" in t:
            return "shard-readerA(query,key)"
        if "reader_value" in t:
            return "shard-readerB(value)"
        if "writer" in t:
            return "shard-writer"
        return "shard-other"
    if "e2e" in t or "all_blocks_head" in t:
        return "e2e"
    return "block"


def timings(d: dict) -> dict:
    t = d.get("timings_seconds", {}) or {}
    return {k: t.get(k) for k in (ENC, SRV, CLI)}


def fnum(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def collect(logs: Path) -> list[dict]:
    runs = []
    for jp in sorted(logs.glob("*/*/*.json")):
        if jp.name in ("summary.json",):
            continue
        d = load(jp)
        if not isinstance(d, dict) or "timings_seconds" not in d:
            continue
        config = jp.parts[len(logs.parts)]  # kimon/logs/<config>/...
        tag = jp.stem
        runs.append(
            {
                "config": config,
                "tag": tag,
                "scope": classify(config, tag),
                "path": str(jp),
                "mtime": jp.stat().st_mtime,
                "passed": d.get("passed"),
                "rel_inf": d.get("global_rel_inf"),
                **timings(d),
            }
        )
    return runs


def latest(runs, config, scope, want_pass=True):
    c = [
        r
        for r in runs
        if r["config"] == config
        and r["scope"] == scope
        and (r["passed"] in (True, None) or not want_pass)
    ]
    return max(c, key=lambda r: r["mtime"]) if c else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path(__file__).resolve().parents[1] / "logs"))
    ap.add_argument("--json", action="store_true", help="also write summary.json")
    a = ap.parse_args()
    logs = Path(a.logs)
    if not logs.is_dir():
        print(f"no logs dir: {logs}", file=sys.stderr)
        return 2
    runs = collect(logs)

    lines = ["# kimon run summary", ""]
    if not runs:
        lines += [f"_No result JSONs found yet under_ `{logs}`."]
        out = "\n".join(lines) + "\n"
        (logs / "summary.md").write_text(out)
        print(out)
        return 0

    # --- per-run table ---
    lines += [
        "## Runs",
        "",
        "| config | scope | enc_eval s | server s | client s | pass | rel_inf |",
        "|---|---|--:|--:|--:|:--:|--:|",
    ]
    for r in sorted(runs, key=lambda r: (r["config"], r["scope"], r["mtime"])):
        lines.append(
            "| {config} | {scope} | {e} | {s} | {c} | {p} | {ri} |".format(
                config=r["config"],
                scope=r["scope"],
                e=fnum(r[ENC]),
                s=fnum(r[SRV]),
                c=fnum(r[CLI]),
                p=("✅" if r["passed"] else ("?" if r["passed"] is None else "❌")),
                ri=(
                    f"{r['rel_inf']:.2e}"
                    if isinstance(r["rel_inf"], (int, float))
                    else "-"
                ),
            )
        )
    lines.append("")

    # --- derived speedups ---
    lines += ["## Derived speedups", ""]
    derived = {}

    def ratio(a, b):
        return (a[ENC] / b[ENC]) if (a and b and a.get(ENC) and b.get(ENC)) else None

    # diagcache benefit: config1 (no cache) vs config3 (cache), per scope
    for scope in ("block", "e2e"):
        c1 = latest(runs, "config1_simd", scope)
        c3 = latest(runs, "config3_all_opts", scope)
        r = ratio(c1, c3)
        if r:
            derived[f"diagcache_speedup_{scope}"] = r
            lines.append(
                f"- **diagcache ({scope})**: config1 {fnum(c1[ENC])}s -> "
                f"config3 {fnum(c3[ENC])}s = **{r:.2f}× faster** "
                f"(same accuracy; CPU-side cache)."
            )
    # block -> e2e scaling (how the full 12-block+head compares to one block)
    for cfg in ("config1_simd", "config3_all_opts"):
        b = latest(runs, cfg, "block")
        e = latest(runs, cfg, "e2e")
        if b and e and b.get(ENC) and e.get(ENC):
            f = e[ENC] / b[ENC]
            derived[f"block_to_e2e_factor_{cfg}"] = f
            lines.append(
                f"- **block→e2e ({cfg})**: {fnum(b[ENC])}s → {fnum(e[ENC])}s "
                f"= **{f:.1f}×** (12 blocks + head vs 1 block; ~12× + head + refresh)."
            )
    # config2 Q/K/V 2-GPU concurrency benefit
    ra = latest(runs, "config2_sharding", "shard-readerA(query,key)")
    rb = latest(runs, "config2_sharding", "shard-readerB(value)")
    if ra and rb and ra.get(ENC) and rb.get(ENC):
        concurrent = max(ra[ENC], rb[ENC])
        serial = ra[ENC] + rb[ENC]
        sp = serial / concurrent
        derived["qkv_projection_shard_speedup"] = sp
        lines.append(
            f"- **Q/K/V projection sharding (config2)**: concurrent "
            f"max({fnum(ra[ENC])}, {fnum(rb[ENC])}) = {fnum(concurrent)}s vs "
            f"serial {fnum(serial)}s = **{sp:.2f}× faster** (2-GPU projection step)."
        )
    if not derived:
        lines.append("_Not enough matched runs yet to derive a speedup._")
    lines.append("")
    lines += [
        (
            "> enc_eval = `encrypted_evaluation`; speedups use the latest PASSing "
            "run per (config, scope). Timings are contention-sensitive — compare "
            "runs from comparably-loaded nodes."
        ),
        "",
    ]

    out = "\n".join(lines) + "\n"
    (logs / "summary.md").write_text(out)
    print(out)
    if a.json:
        (logs / "summary.json").write_text(
            json.dumps({"runs": runs, "derived": derived}, indent=2, default=str)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
