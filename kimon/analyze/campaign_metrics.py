#!/usr/bin/env python3
"""Normalize one Scheme-B/T123 run into a single evidence row for the
optimization campaign, across ALL axes: oracle, client/server time split,
per-stage client work, RAM (host + GPU + keygen-phase), CPU phases
(server GPU-active vs client CPU-busy, from telemetry), disk, and thread config.

Usage:
  campaign_metrics.py LABEL RESULT_JSON TELEMETRY_CSV [RUN_LOG]

Appends a row to kimon/logs/optimization_campaign/campaign.csv and reprints
the whole table so evidence survives across sessions (no context needed).
"""
import csv, json, os, sys, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "kimon", "logs", "optimization_campaign")
CSV = os.path.join(CAMP, "campaign.csv")

FIELDS = [
    "label", "job", "passed", "global_rel_inf", "worst_token_rel_inf",
    "wall_s", "keygen_s", "encrypt_input_s", "server_linalg_s",
    "client_boundary_s", "final_decrypt_s", "client_total_s",
    "server_share_pct", "client_share_pct",
    "round_trips", "score_tiles", "weight_tiles", "ln_gelu_rt",
    "gpu_mem_peak_mib", "host_rss_peak_mib", "keygen_rss_mib",
    "gpu_active_wall_s", "client_cpu_wall_s", "peak_client_cpu_cores",
    "omp_threads", "json_bytes", "runlog_bytes", "telemetry_rows",
    # v2+: appended at the end so pre-v2 rows stay valid with these blank.
    # client_boundary_s/client_total_s/*_share_pct above use the WALL
    # estimate when present (true wall-clock under concurrency); this column
    # keeps the raw aggregate-of-per-call-CPU-seconds visible too, since for
    # v0/v1 (fully sequential) the two are identical but for v2+ (parallel
    # boundaries) the aggregate can exceed wall-clock by design.
    "client_boundary_cpu_agg_s",
]


def num(j, *path):
    x = j
    for p in path:
        if isinstance(x, dict) and p in x:
            x = x[p]
        else:
            return None
    return x


def main():
    label, rj, tel = sys.argv[1], sys.argv[2], sys.argv[3]
    runlog = sys.argv[4] if len(sys.argv) > 4 else ""
    j = json.load(open(rj))
    t = j.get("timings_seconds", {})
    cfg = j.get("config", {})
    proto = j.get("protocol", {})

    keygen = t.get("context_keygen_load")
    encin = t.get("encrypt_embedded_inputs")
    srv = t.get("server_linear_algebra_seconds")
    cli_agg = t.get("client_boundary_seconds_total")
    # v2+: client_boundary_wall_seconds_estimate is the true wall-clock
    # client cost when boundary crossings run concurrently (see the source
    # comment on Client::seconds() in the t123_v2 driver); cli_agg alone
    # would overstate client share once boundaries are parallelized. Older
    # (v0/v1, fully sequential) runs lack this field -- fall back to cli_agg,
    # which equals wall-clock time in that case anyway.
    cli = t.get("client_boundary_wall_seconds_estimate", cli_agg)
    dec = t.get("final_decrypt")
    client_total = sum(v for v in (encin, cli, dec) if v is not None)
    online = (srv or 0) + client_total
    st = cfg.get("score_tile_decryptions")
    wt = cfg.get("weight_tile_encryptions")
    rt = proto.get("round_trips") or j.get("declared_client_boundary_crossings")
    ln_gelu = (rt - (st or 0) - (wt or 0)) if rt is not None else None
    env = j.get("environment", "")
    m = re.search(r"omp=(\d+)", env)
    omp = int(m.group(1)) if m else None

    # telemetry: handle 7-col (no cpu) or 8-col (with cpu_cores_busy)
    gpu_peak = host_peak = keygen_rss = 0
    gpu_active = client_cpu = 0
    peak_cores = 0.0
    rows = 0
    tick = 5
    with open(tel) as f:
        r = csv.reader(f)
        header = next(r, None)
        has_cpu = header and "cpu_cores_busy" in header
        for row in r:
            if len(row) < 7:
                continue
            rows += 1
            el = int(float(row[1]))
            gmem = 0 if row[2] == "NA" else int(float(row[2]))
            # multi-GPU nodes (e.g. gpu-a100-dev has 3 A100s) report
            # "util0;util1;util2" from nvidia-smi; the job always runs
            # --gpu 0, so take that GPU's value, not the whole string.
            gutil_raw = row[3].split(";")[0] if row[3] else "NA"
            gutil = 0 if gutil_raw == "NA" else float(gutil_raw)
            rss = 0 if row[5] == "NA" else int(float(row[5]))  # vmhwm col 6? use RSS col5
            rssv = 0 if row[4] == "NA" else int(float(row[4]))
            cb = None
            if has_cpu and len(row) >= 8 and row[7] not in ("", "NA"):
                cb = float(row[7])
            gpu_peak = max(gpu_peak, gmem)
            host_peak = max(host_peak, rssv)
            if el <= 10:
                keygen_rss = max(keygen_rss, rssv)
            if gutil >= 20:
                gpu_active += tick
            elif cb is not None and gutil < 5 and cb >= 1.0:
                client_cpu += tick
            if cb is not None:
                peak_cores = max(peak_cores, cb)

    row = {
        "label": label, "job": j.get("environment", "")[-24:],
        "passed": j.get("passed"), "global_rel_inf": j.get("global_rel_inf"),
        "worst_token_rel_inf": j.get("worst_token_rel_inf"),
        "wall_s": None, "keygen_s": keygen, "encrypt_input_s": encin,
        "server_linalg_s": srv, "client_boundary_s": cli, "final_decrypt_s": dec,
        "client_total_s": round(client_total, 2),
        "server_share_pct": round(100 * (srv or 0) / online, 1) if online else None,
        "client_share_pct": round(100 * client_total / online, 1) if online else None,
        "round_trips": rt, "score_tiles": st, "weight_tiles": wt, "ln_gelu_rt": ln_gelu,
        "gpu_mem_peak_mib": gpu_peak, "host_rss_peak_mib": host_peak,
        "keygen_rss_mib": keygen_rss, "gpu_active_wall_s": gpu_active,
        "client_cpu_wall_s": client_cpu, "peak_client_cpu_cores": peak_cores,
        "omp_threads": omp, "json_bytes": os.path.getsize(rj),
        "runlog_bytes": os.path.getsize(runlog) if runlog and os.path.exists(runlog) else 0,
        "telemetry_rows": rows,
        "client_boundary_cpu_agg_s": cli_agg,
    }
    os.makedirs(CAMP, exist_ok=True)
    exists = os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

    # reprint the whole table (evidence trail)
    print(f"[recorded] {label}")
    with open(CSV) as f:
        for line in f:
            print(line.rstrip())


if __name__ == "__main__":
    main()
