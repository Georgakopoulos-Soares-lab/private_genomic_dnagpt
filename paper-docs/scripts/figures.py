#!/usr/bin/env python3
"""Generate compact IEEE figures from sourced evidence/*.yaml, as vector PDF."""
from __future__ import annotations

import argparse
import pathlib
import sys
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from figstyle import (
    AMBER, BLACK, CLIENT, CLIENT_PALE, COLUMN_WIDTH, DPI, FAILURE, GRAY,
    LIGHT_GRAY, MEASURED_KW, PAGE_WIDTH, PALE_ORANGE, PROJECTED_KW,
    SERVER, SERVER_PALE, arrow, canvas, despine, labelled_box, save, save_png,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "paper-docs" / "evidence"
DEFAULT_OUT = ROOT / "paper-docs" / "manuscript" / "figures"
PNG_DIR: pathlib.Path | None = None


def load(name: str) -> dict:
    with (EVIDENCE / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)


class Ledger:
    def __init__(self) -> None:
        self.rows = {}
        for section in load("measurements").values():
            if isinstance(section, list):
                for row in section:
                    if isinstance(row, dict) and "id" in row:
                        self.rows[row["id"]] = row

    def row(self, rid: str) -> dict:
        if rid not in self.rows:
            raise SystemExit(f"Missing evidence row: {rid}")
        return self.rows[rid]

    def value(self, rid: str):
        return self.row(rid)["value"]

    def require_tag(self, rid: str, tag: str = "[V]") -> None:
        if self.row(rid).get("tag") != tag:
            raise SystemExit(f"Evidence row {rid} must have tag {tag}.")


def sci(value: float, digits: int = 3) -> str:
    coefficient, exponent = f"{value:.{digits - 1}e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def emit(fig, out, name):
    if PNG_DIR is not None:
        save_png(fig, PNG_DIR, name, dpi=220)
    return save(fig, out, name)


def panel_title(fig, text, note=None):
    fig.text(.02, .975, text, ha="left", va="top", fontsize=8.1, weight="bold")
    if note:
        fig.text(.02, .895, note, ha="left", va="top", fontsize=6.8, color=GRAY)


def below(fig, artist, gap_pt=5.0):
    """Figure-fraction y just below `artist`'s actually-rendered extent.

    Positions text relative to real geometry rather than a hand-picked axes-fraction offset,
    which is unreliable across subplots of different physical heights (e.g. unequal
    height_ratios): the same axes-fraction delta covers a different absolute distance in each.
    """
    fig.canvas.draw()
    bbox = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.transFigure.inverted())
    return bbox.y0 - gap_pt / 72.0 / fig.get_figheight()


def fig_graphical_abstract(out):
    L = Ledger()
    for rid in ("prompt.gsr_total", "full.wall", "full.head_margin_relative_error",
                "full.head_label_matches_reference", "client.needs_gpu"):
        L.require_tag(rid)
    fig, ax = canvas((PAGE_WIDTH, 2.05))
    ax.text(50, 96, "Encrypted genomic classification", ha="center", va="center",
            fontsize=10, weight="bold")
    for x, title, note, col, pale in (
        (1, "Data owner", "Genome and secret key\nCPU boundary evaluation", CLIENT, CLIENT_PALE),
        (69, "Compute provider", "Model and evaluation keys\nEncrypted linear algebra", SERVER, SERVER_PALE),
    ):
        labelled_box(ax, x, 49, 30, 34, title, note, facecolor=pale,
                     edgecolor=col, title_size=8.4, sub_size=7.4)
    labelled_box(ax, 36, 49, 28, 34, "Client-assisted CKKS",
                 "Encrypted vectors in\nEncrypted prediction out", facecolor="white",
                 edgecolor=BLACK, title_size=8.2, sub_size=7.2)
    arrow(ax, 31.5, 66, 35.5, 66, color=CLIENT, mutation=9)
    arrow(ax, 68.5, 66, 64.5, 66, color=SERVER, mutation=9)
    ax.text(50, 39, "Provider receives no plaintext activation or secret key.",
            ha="center", va="center", fontsize=7.4)
    values = [
        (f"{L.value('prompt.gsr_total')} tokens", "complete classifier"),
        (f"{L.value('full.wall'):,.0f} s", "measured wall time"),
        (sci(L.value("full.head_margin_relative_error")), "head-margin relative error"),
        ("Matching label", "plaintext reference"),
    ]
    for i, (value, note) in enumerate(values):
        labelled_box(ax, 1+i*25, 3, 23, 27, value, note, title_size=8.5,
                     sub_size=6.9, facecolor=LIGHT_GRAY, edgecolor=GRAY)
    return emit(fig, out, "fig_graphical_abstract")


def fig_architecture(out):
    fig, ax = canvas((PAGE_WIDTH, 2.25))
    ax.text(1, 96, "One transformer block: evaluation order and executor",
            va="center", fontsize=9, weight="bold")
    ax.text(1, 86, "Data owner: genome + secret key", color=CLIENT, fontsize=7.3)
    ax.text(56, 86, "Compute provider: model + evaluation keys", color=SERVER, fontsize=7.3)
    steps = [
        ("server", "LayerNorm\nstatistics"),
        ("client", "Inverse\nsquare root"),
        ("server", "Normalize;\nproject queries,\nkeys, values"),
        ("server", "Causal\nattention scores"),
        ("client", "Stable softmax\non complete\nrows"),
        ("server", "Attention context\nprojection\n+ residual"),
        ("server", "Second\nLayerNorm\nstatistics"),
        ("client", "Inverse\nsquare root"),
        ("server", "Normalize; MLP\nup-projection"),
        ("client", "GELU\nactivation"),
        ("server", "MLP\ndown-projection\n+ residual"),
        ("server", "Encrypted\nblock output"),
    ]
    locs, label_boxes = [], []
    box_width, box_height = 15, 26
    for i, (who, label) in enumerate(steps):
        col = i if i < 6 else 11-i
        x, y = 1+col*16.65, 50 if i < 6 else 13
        colour, pale = (CLIENT, CLIENT_PALE) if who == "client" else (SERVER, SERVER_PALE)
        labelled_box(ax, x, y, box_width, box_height, label, facecolor=pale,
                     edgecolor=colour, title_size=6.6)
        label_boxes.append((ax.texts[-1], ax.patches[-1]))
        locs.append((x, y))
    for i in range(len(locs)-1):
        x, y = locs[i]
        nx, ny = locs[i+1]
        if ny != y:
            arrow(ax, x+7.5, y-.6, nx+7.5, ny+box_height+.6, mutation=8, lw=1)
        elif nx > x:
            arrow(ax, x+15.2, y+box_height/2, nx-.2, ny+box_height/2, mutation=7, lw=1)
        else:
            arrow(ax, x-.2, y+box_height/2, nx+15.2, ny+box_height/2, mutation=7, lw=1)
    ax.text(50, 4, "Client functions decrypt, evaluate in double precision, and re-encrypt; "
            "score collection and softmax emission are separate.", fontsize=6.7,
            ha="center", va="center", color=GRAY)
    # Check actual rendered type, not character counts: every label must clear
    # all four borders by at least 3 pt at the final two-column print width.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_padding = 3 * fig.dpi / 72
    for text, patch in label_boxes:
        tb, pb = text.get_window_extent(renderer), patch.get_window_extent(renderer)
        clearance = min(tb.x0-pb.x0, pb.x1-tb.x1, tb.y0-pb.y0, pb.y1-tb.y1)
        if clearance < min_padding:
            raise ValueError(f"Architecture label has insufficient padding: {text.get_text()!r}")
    return emit(fig, out, "fig_architecture")


def fig_packing(out):
    L = Ledger()
    copies, width, lanes, slots, groups, tokens, active = [L.value(rid) for rid in (
        "layout.copies", "layout.pack_width", "layout.token_lanes", "param.slots",
        "layout.token_groups", "prompt.gsr_total", "layout.embedding_width")]
    fig, ax = canvas((COLUMN_WIDTH, 2.2))
    ax.text(1, 96, f"{slots:,} slots per ciphertext", fontsize=8.4, weight="bold", va="center")
    ax.text(1, 85, f"{copies} copies × {width:,} features × {lanes} token lanes",
            fontsize=7.0, va="center")
    for c in range(copies):
        x = 1+c*24.75
        labelled_box(ax, x, 54, 23, 22, f"Copy {c+1}", f"{lanes} lanes / feature",
                     facecolor=SERVER_PALE, edgecolor=SERVER, title_size=7.2, sub_size=6.0)
    ax.text(1, 44, f"{active} active features; padding includes attention staging.",
            fontsize=6.6, color=GRAY, va="center")
    ax.text(1, 31, f"{tokens} tokens → {groups} groups", fontsize=8.1, weight="bold", va="center")
    for g in range(groups):
        live = min(lanes, tokens-g*lanes)
        ax.add_patch(plt.Rectangle((1+g*7.5, 12), 6.6, 10,
                     facecolor=SERVER_PALE if live == lanes else PALE_ORANGE,
                     edgecolor=SERVER if live == lanes else AMBER, linewidth=.8))
    ax.text(1, 4, f"Final group: {tokens % lanes} live lanes + {(-tokens) % lanes} padded lane.",
            fontsize=6.8, va="center", color=GRAY)
    return emit(fig, out, "fig_packing")


def fig_cost_decomposition(out):
    """Merges the former fig_waterfall (by phase) and fig_cost_split (by party) panels.

    Both read the same `full.encrypted_evaluation` ledger row, so they are two cuts of one
    measured interval rather than two independent measurements.
    """
    L = Ledger()
    wf_ids = ("full.wall", "full.encrypted_evaluation", "full.blocks_encrypted_evaluation",
              "full.refreshes", "full.head_encrypted_evaluation")
    for rid in wf_ids:
        L.require_tag(rid)
    wall, encrypted, blocks, refresh, head = [L.value(rid) for rid in wf_ids]

    cs_ids = ("full.encrypted_evaluation", "full.server_linear_algebra", "full.client_boundaries",
              "full.server_share_pct", "full.client_share_pct")
    for rid in cs_ids:
        L.require_tag(rid)
    total, server, client, server_pct, client_pct = [L.value(rid) for rid in cs_ids]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.55), dpi=DPI,
                                    gridspec_kw={"height_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.24, right=.97, bottom=.24, top=.76, hspace=1.15)
    panel_title(fig, "Where the measured time goes",
                "The same measured interval, viewed by phase and by party.")

    # By phase: process wall vs. blocks/refreshes/head
    ax1.barh(1, wall, height=.38, facecolor=LIGHT_GRAY, **MEASURED_KW)
    ax1.text(wall*.5, 1, f"{wall:,.0f} s", ha="center", va="center", weight="bold")
    ax1.barh(0, blocks, height=.38, facecolor=SERVER, **MEASURED_KW)
    ax1.barh(0, refresh, left=blocks, height=.38, facecolor=CLIENT, edgecolor=BLACK, linewidth=.8)
    ax1.barh(0, head, left=blocks+refresh, height=.38, facecolor=PALE_ORANGE, edgecolor=BLACK, linewidth=.8)
    ax1.text(blocks*.5, 0, f"{encrypted:,.1f} s", color="white", ha="center", va="center", weight="bold")
    ax1.set_yticks([0, 1], ["Encrypted\nevaluation", "Process wall"])
    ax1.set_xlim(0, wall*1.025)
    ax1.set_ylim(-.65, 1.5)
    ax1.set_xlabel("Time (s)", labelpad=8)
    ax1.grid(axis="x", color=LIGHT_GRAY)
    ax1.set_axisbelow(True)
    despine(ax1, keep=("bottom",))
    x0 = ax1.get_position().x0
    fig.text(x0, below(fig, ax1.xaxis.label, gap_pt=7),
             f"Blocks {blocks:,.1f} s · refreshes {refresh:.1f} s · head {head:.2f} s",
             ha="left", va="top", fontsize=6.5, color=GRAY)

    # By party: server vs. client share of the same encrypted-evaluation total
    for value, left, colour in ((server, 0, SERVER), (client, server, CLIENT)):
        ax2.barh(0, value, left=left, height=.6, facecolor=colour, **MEASURED_KW)
    ax2.text(server/2, 0, f"{server_pct:.1f}%", color="white", weight="bold", ha="center", va="center")
    ax2.text(server+client/2, 0, f"{client_pct:.1f}%", color="white", weight="bold", ha="center", va="center", fontsize=6.6)
    ax2.set_xlim(0, total)
    ax2.set_yticks([])
    ax2.set_xlabel("Time (s)", labelpad=8)
    despine(ax2, keep=("bottom",))
    x0 = ax2.get_position().x0
    provider_note = fig.text(x0, below(fig, ax2.xaxis.label, gap_pt=7),
                              f"Provider: {server:,.0f} s · encrypted linear algebra",
                              ha="left", va="top", color=SERVER, fontsize=6.9)
    fig.text(x0, below(fig, provider_note, gap_pt=6),
             f"Data owner: {client:,.0f} s · CPU boundaries · no GPU",
             ha="left", va="top", color=CLIENT, fontsize=6.9)

    return emit(fig, out, "fig_cost_decomposition")


def fig_timeline(out):
    tel = load("telemetry")["complete_run"]
    if tel["tag"] != "[V]":
        raise SystemExit("Complete-run trajectory must be measured.")
    rows = tel["block_trajectory"]
    block = np.array([r["block"] for r in rows])
    seconds = np.array([r["evaluation_s"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(COLUMN_WIDTH, 2.5), dpi=DPI)
    fig.subplots_adjust(left=.19, right=.98, bottom=.16, top=.82, hspace=.13)
    panel_title(fig, "Measured trajectory across the complete run")
    ax1.plot(block, seconds, color=SERVER, marker="o", ms=3, lw=1.2)
    ax1.set_ylabel("Block time (s)")
    ax1.set_ylim(seconds.min()*.96, seconds.max()*1.03)
    ax1.text(.02, .93, f"{seconds.min():.1f}–{seconds.max():.1f} s", transform=ax1.transAxes,
             va="top", fontsize=6.6)
    ax2.semilogy(block, [r["global_rel_inf"] for r in rows], color=SERVER, marker="o",
                 ms=3, lw=1.2, label="Global")
    ax2.semilogy(block, [r["worst_token_rel_inf"] for r in rows], color=CLIENT,
                 marker="s", ms=2.6, ls="--", lw=1.1, label="Worst token")
    ax2.set_ylabel("Relative error")
    ax2.set_xlabel("Transformer block")
    ax2.set_xticks(block)
    ax2.legend(frameon=False, fontsize=6.4, loc="upper left", ncol=2,
                bbox_to_anchor=(0, 1.13), handlelength=1.1, columnspacing=.8)
    for ax in (ax1, ax2):
        despine(ax)
        ax.grid(axis="y", color=LIGHT_GRAY)
    return emit(fig, out, "fig_timeline")


def fig_memory(out):
    L = Ledger()
    ids = ("mem.stage_qkv", "mem.stage_attention", "mem.stage_mlp", "mem.unbounded_failure")
    for rid in ids:
        L.require_tag(rid)
    values = [L.value(rid) for rid in ids[:3]]
    fail = L.value(ids[-1])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3), dpi=DPI)
    fig.subplots_adjust(left=.18, right=.98, bottom=.21, top=.75)
    panel_title(fig, "Stage flushing bounds host memory", "Standalone-block samples; target-process resident set.")
    ax.bar(range(len(values)), values, width=.55, facecolor=SERVER, **MEASURED_KW)
    for i, value in enumerate(values):
        ax.text(i, value+1.6, f"{value:.1f}", ha="center", fontsize=7, weight="bold")
    ax.axhline(fail, color=FAILURE, lw=1.1, ls="--")
    ax.text(.02, .95, f"Unbounded cache: failure at {fail:.1f} GiB", transform=ax.transAxes,
             color=FAILURE, va="top", fontsize=6.7)
    ax.set_xticks(range(len(values)), ["Query/key/value", "Attention", "MLP"])
    ax.set_ylabel("Host memory (GiB)")
    ax.set_ylim(0, fail*1.27)
    ax.grid(axis="y", color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    despine(ax)
    fig.text(.02, .035, "Stage peaks are sampled, not allocation-by-allocation maxima.", fontsize=6.5, color=GRAY)
    return emit(fig, out, "fig_memory")


def fig_scaling(out):
    rows = load("scaling")["tasks"]
    labels = ["Core promoter", "300 bp promoter", "Splice site", "Genomic signal"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4), dpi=DPI)
    fig.subplots_adjust(left=.31, right=.98, bottom=.21, top=.75)
    panel_title(fig, "Complete run and shorter-task projections", "Solid: measured; hatched: group-linear wall-time model.")
    largest = max(r["full_pass_seconds"] for r in rows)
    for i, row in enumerate(rows):
        measured = row["full_pass_tag"] == "[V]"
        value = row["full_pass_seconds"]
        ax.barh(i, value, height=.58, facecolor=SERVER if measured else SERVER_PALE,
                 **(MEASURED_KW if measured else PROJECTED_KW))
        ax.text(value+largest*.025, i, f"{value:,.0f}", ha="left", va="center",
                 fontsize=6.8, weight="bold" if measured else "normal")
    ax.set_yticks(range(len(rows)), [f"{label}\n{row['tokens']} tokens" for label, row in zip(labels, rows)])
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, largest*1.29)
    ax.grid(axis="x", color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    fig.text(.02, .035, "Fixed circuit/head; attention and fixed costs not fitted separately.", fontsize=6.5, color=GRAY)
    return emit(fig, out, "fig_scaling")


def fig_baseline(out):
    L = Ledger()
    rows = [("Signal (accuracy)", "base.gsr_accuracy", 4),
            ("Core promoter (MCC)", "base.gue_prom_core_mcc", 3),
            ("300 bp promoter (MCC)", "base.gue_prom_300_mcc", 3),
            ("Splice site (MCC)", "base.gue_splice_mcc", 3),
            ("mRNA ($r^2$)", "base.mrna_r2", 3)]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5), dpi=DPI)
    fig.subplots_adjust(left=.43, right=.98, bottom=.16, top=.78)
    panel_title(fig, "Plaintext model reproduction", "Filled: this work; open: published reference.")
    for i, (_, rid, precision) in enumerate(rows):
        own, ref = L.value(rid), L.row(rid)["reference"]
        ax.plot([own, ref], [i, i], color=GRAY, lw=.8)
        ax.plot(own, i, "o", color=SERVER, ms=4.4)
        ax.plot(ref, i, "s", mfc="white", mec=GRAY, ms=4)
        ax.text(.995, i+.36, f"{own:.{precision}f} / {ref:.{precision}f}",
                 ha="right", va="center", fontsize=6.3, color=GRAY)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(.48, 1.0)
    ax.set_ylim(len(rows)-.45, -.65)
    ax.set_xlabel("Task-specific metric value")
    ax.grid(axis="x", color=LIGHT_GRAY)
    despine(ax, keep=("bottom",))
    return emit(fig, out, "fig_baseline")


FIGURES = {
    "graphical_abstract": fig_graphical_abstract,
    "architecture": fig_architecture,
    "packing": fig_packing,
    "cost_decomposition": fig_cost_decomposition,
    "timeline": fig_timeline,
    "memory": fig_memory,
    "scaling": fig_scaling,
    "baseline": fig_baseline,
}


def main() -> int:
    global PNG_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), metavar="NAME")
    ap.add_argument("--png", type=pathlib.Path)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(FIGURES))
        return 0
    PNG_DIR = args.png
    for name in args.only or list(FIGURES):
        print(f"  {FIGURES[name](args.output_dir).relative_to(ROOT)}")
    print(f"{len(args.only or FIGURES)} figure(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
