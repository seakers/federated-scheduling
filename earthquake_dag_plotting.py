#!/usr/bin/env python3
"""
Plot the earthquake damage-assessment workflow.

Two panels:
  (a) the per-settlement DAG, with the gate expression on every node and the
      AND / OR / NOT structure marked explicitly;
  (b) the phase windows on a time axis, so the sequencing and the 72 h horizon
      are visible at a glance.

Timing constants are imported from earthquake_utils when it is importable, so the
figure cannot drift out of sync with the workflow it documents. If the import
fails (no FAME on the path) it falls back to the values below and says so.

    python plot_earthquake_workflow.py                  # -> earthquake_workflow.pdf/.png
    python plot_earthquake_workflow.py --settlements 4  # add the multi-settlement inset
    python plot_earthquake_workflow.py -o fig3          # choose the basename
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# --------------------------------------------------------------------------
# Constants -- imported so the figure tracks the code
# --------------------------------------------------------------------------
FALLBACK = dict(
    EXTENT_WINDOW_H=8.0,
    URBAN_OPEN_H=1.0,  URBAN_CLOSE_H=12.0,
    TRIAGE_OPEN_H=6.0, TRIAGE_CLOSE_H=24.0,
    FINAL_OPEN_H=12.0, FINAL_CLOSE_H=48.0,
    VALUE_EXTENT=60.0, VALUE_URBAN_OPT=90.0, VALUE_URBAN_SAR=70.0,
    VALUE_TRIAGE=130.0, VALUE_HIRES=100.0, VALUE_ACCESS=80.0,
)

try:
    import earthquake_utils as eq
    C = {k: getattr(eq, k) for k in FALLBACK}
    SOURCE = "earthquake_utils"
except Exception as exc:                                    # noqa: BLE001
    C = dict(FALLBACK)
    SOURCE = f"fallback constants ({type(exc).__name__})"

# --------------------------------------------------------------------------
# Palette: instrument drives colour, so the SAR/optical substitution at the OR
# gate reads without consulting the legend.
# --------------------------------------------------------------------------
SAR_FACE, SAR_EDGE = "#dbe7f3", "#2c5f8a"
OPT_FACE, OPT_EDGE = "#fae3d0", "#b5651d"
LOGIC_FACE, LOGIC_EDGE = "#e8e4f0", "#5b4b8a"
NOT_COLOUR = "#a4243b"
INK = "#222222"

# name, x, y, instrument, label, gate text, mission value
NODES = [
    ("extent",    0.0, 1.50, "sar", "EXTENT\nregion",      None,                  "VALUE_EXTENT"),
    ("urban_opt", 1.5, 2.55, "opt", "URBAN_opt\nurban area", "Lit(extent)",        "VALUE_URBAN_OPT"),
    ("urban_sar", 1.5, 0.45, "sar", "URBAN_sar\nurban area", "Lit(extent)",        "VALUE_URBAN_SAR"),
    ("char",      2.7, 1.50, "logic", "characterised",     "Or(opt, sar)",         None),
    ("triage",    3.9, 1.50, "opt", "TRIAGE\ndistrict",    "Lit(char)",            "VALUE_TRIAGE"),
    ("hires",     5.4, 2.55, "opt", "HIRES\ndistrict",     "And(triage, ¬opt)",    "VALUE_HIRES"),
    ("access",    5.4, 0.45, "sar", "ACCESS\ndistrict",    "Lit(triage)",          "VALUE_ACCESS"),
]
POS = {n[0]: (n[1], n[2]) for n in NODES}

EDGES = [
    ("extent", "urban_opt", "solid"),
    ("extent", "urban_sar", "solid"),
    ("urban_opt", "char", "solid"),
    ("urban_sar", "char", "solid"),
    ("char", "triage", "solid"),
    ("triage", "hires", "solid"),
    ("triage", "access", "solid"),
    ("urban_opt", "hires", "not"),
]

BOX_W, BOX_H = 1.02, 0.62


def _draw_node(ax, name, x, y, kind, label, gate, value_key):
    if kind == "logic":
        d = 0.44
        ax.add_patch(Polygon([(x, y + d), (x + d * 1.5, y), (x, y - d), (x - d * 1.5, y)],
                             closed=True, facecolor=LOGIC_FACE, edgecolor=LOGIC_EDGE,
                             lw=1.6, zorder=3))
        ax.text(x, y + 0.05, label, ha="center", va="center", fontsize=8.5,
                color=LOGIC_EDGE, style="italic", zorder=4)
        ax.text(x, y - 0.17, "OR", ha="center", va="center", fontsize=9,
                color=LOGIC_EDGE, fontweight="bold", zorder=4)
        return

    face, edge = (SAR_FACE, SAR_EDGE) if kind == "sar" else (OPT_FACE, OPT_EDGE)
    mandatory = name == "extent"
    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.045,rounding_size=0.09",
        facecolor=face, edgecolor=edge, lw=2.4 if mandatory else 1.4, zorder=3))
    ax.text(x, y + 0.10, label.split("\n")[0], ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=INK, zorder=4)
    ax.text(x, y - 0.13, label.split("\n")[1], ha="center", va="center",
            fontsize=7.6, color="#555555", style="italic", zorder=4)
    if mandatory:
        ax.text(x, y + BOX_H / 2 + 0.15, "MANDATORY", ha="center", va="bottom",
                fontsize=7.2, fontweight="bold", color=edge, zorder=4)
    if value_key:
        ax.text(x, y - BOX_H / 2 - 0.13, f"value {C[value_key]:.0f}", ha="center",
                va="top", fontsize=7.0, color="#777777", zorder=4)
    if gate:
        ax.text(x, y - BOX_H / 2 - 0.31, gate, ha="center", va="top",
                fontsize=7.2, color=NOT_COLOUR if "¬" in gate else "#666666",
                family="monospace",
                fontweight="bold" if "¬" in gate else "normal", zorder=4)


def _draw_edge(ax, src, dst, style):
    x0, y0 = POS[src]
    x1, y1 = POS[dst]
    if style == "not":
        # Route the anti-complementarity edge over the top so it cannot be
        # mistaken for a prerequisite arrow. Keep the arc shallow: a deep one
        # clips against the axes and reads as two disconnected strokes.
        ax.add_patch(FancyArrowPatch(
            (x0, y0 + BOX_H / 2), (x1, y1 + BOX_H / 2),
            connectionstyle="arc3,rad=-0.20", arrowstyle="-[,widthB=0.45,lengthB=0.16",
            color=NOT_COLOUR, lw=1.8, ls=(0, (4, 2)), zorder=2,
            shrinkA=2, shrinkB=2))
        ax.text((x0 + x1) / 2, max(y0, y1) + 1.28, "NOT  (anti-complementary)",
                ha="center", va="bottom", fontsize=8, color=NOT_COLOUR,
                fontweight="bold")
        return
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
        color="#8a8a8a", lw=1.3, zorder=2,
        shrinkA=34, shrinkB=34))


def panel_dag(ax):
    for src, dst, style in EDGES:
        _draw_edge(ax, src, dst, style)
    for name, x, y, kind, label, gate, vk in NODES:
        _draw_node(ax, name, x, y, kind, label, gate, vk)

    # Bracket the AND cascade: EXTENT gates everything to its right.
    ax.annotate("", xy=(6.05, -0.55), xytext=(-0.55, -0.55),
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=1.0))
    ax.text(2.75, -0.72, "AND cascade: EXTENT gates all five downstream tasks, "
                         "so its marginal value is the whole subtree",
            ha="center", va="top", fontsize=8, color="#666666")

    ax.set_xlim(-1.0, 6.5)
    ax.set_ylim(-1.15, 4.25)
    ax.axis("off")
    ax.set_title("(a)  Per-settlement workflow, with prerequisite gates",
                 fontsize=11, fontweight="bold", loc="left", color=INK)


def panel_timing(ax):
    e = C["EXTENT_WINDOW_H"]
    bars = [
        ("ACCESS",    C["FINAL_OPEN_H"],  e + C["FINAL_CLOSE_H"],  "sar"),
        ("HIRES",     C["FINAL_OPEN_H"],  e + C["FINAL_CLOSE_H"],  "opt"),
        ("TRIAGE",    C["TRIAGE_OPEN_H"], e + C["TRIAGE_CLOSE_H"], "opt"),
        ("URBAN_sar", C["URBAN_OPEN_H"],  e + C["URBAN_CLOSE_H"],  "sar"),
        ("URBAN_opt", C["URBAN_OPEN_H"],  e + C["URBAN_CLOSE_H"],  "opt"),
        ("EXTENT",    0.0,                e,                       "sar"),
    ]
    for i, (label, lo, hi, kind) in enumerate(bars):
        face, edge = (SAR_FACE, SAR_EDGE) if kind == "sar" else (OPT_FACE, OPT_EDGE)
        ax.barh(i, hi - lo, left=lo, height=0.56, color=face, edgecolor=edge,
                lw=1.3, zorder=3)
        ax.text(lo + 0.5, i, label, va="center", ha="left", fontsize=8.5,
                color=INK, zorder=4)

    for h, txt, col in ((24, "24 h\nCharter target", "#888888"),
                        (72, "72 h\nSAR survival", NOT_COLOUR)):
        ax.axvline(h, color=col, ls="--", lw=1.2, zorder=2)
        ax.text(h + 0.8, len(bars) - 0.35, txt, fontsize=7.6, color=col,
                va="top", ha="left")

    ax.set_yticks([])
    ax.set_ylim(-0.7, len(bars) - 0.1)
    ax.set_xlim(-1, 80)
    ax.set_xlabel("hours since the event  (windows are relative to the REALISED "
                  "parent acquisition)", fontsize=8.5)
    ax.set_title("(b)  Phase windows: mission value decays with acquisition age",
                 fontsize=11, fontweight="bold", loc="left", color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#eeeeee", zorder=0)


def panel_scale(ax, k):
    """Compact view of K independent settlement subtrees."""
    for s in range(k):
        y = k - 1 - s
        ax.text(-0.45, y, f"settlement {s + 1}", fontsize=8, ha="right",
                va="center", color="#555555")
        for j, (kind, lbl) in enumerate([("sar", "E"), ("opt", "Uo"), ("sar", "Us"),
                                         ("opt", "T"), ("opt", "H"), ("sar", "A")]):
            face, edge = (SAR_FACE, SAR_EDGE) if kind == "sar" else (OPT_FACE, OPT_EDGE)
            ax.add_patch(FancyBboxPatch(
                (j * 0.75, y - 0.19), 0.55, 0.38,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=face, edgecolor=edge, lw=2.0 if j == 0 else 1.0, zorder=3))
            ax.text(j * 0.75 + 0.275, y, lbl, ha="center", va="center",
                    fontsize=7.5, color=INK, zorder=4)
        if s < k - 1:
            ax.plot([-0.35, 4.6], [y - 0.42, y - 0.42], color="#eeeeee", lw=0.8)
    ax.set_xlim(-1.9, 4.8)
    ax.set_ylim(-0.6, k - 0.3)
    ax.axis("off")
    ax.set_title(f"(c)  {k} settlements x 6 tasks = {6 * k} tasks; subtrees are "
                 f"independent (no shared root)",
                 fontsize=11, fontweight="bold", loc="left", color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="earthquake_workflow",
                    help="output basename (default: earthquake_workflow)")
    ap.add_argument("--settlements", type=int, default=0,
                    help="if >0, add panel (c) showing K settlement subtrees")
    args = ap.parse_args()

    k = args.settlements
    if k > 0:
        fig, axes = plt.subplots(
            3, 1, figsize=(10.5, 10.4),
            gridspec_kw=dict(height_ratios=[3.1, 1.7, max(1.0, 0.32 * k)]))
        panel_dag(axes[0]); panel_timing(axes[1]); panel_scale(axes[2], k)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4),
                                 gridspec_kw=dict(height_ratios=[3.1, 1.7]))
        panel_dag(axes[0]); panel_timing(axes[1])

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=10, mfc=SAR_FACE, mec=SAR_EDGE,
                   label="SAR (cloud- and night-independent)"),
        plt.Line2D([], [], marker="s", ls="", ms=10, mfc=OPT_FACE, mec=OPT_EDGE,
                   label="optical (resolves structures)"),
        plt.Line2D([], [], marker="D", ls="", ms=9, mfc=LOGIC_FACE, mec=LOGIC_EDGE,
                   label="logic node (OR over substitutable parents)"),
        plt.Line2D([], [], color=NOT_COLOUR, ls=(0, (4, 2)), lw=1.8,
                   label="NOT edge: more URBAN_opt lowers the value of HIRES"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Post-earthquake damage assessment: AND / OR / NOT workflow",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.055, 1, 0.975])

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"[plot] constants from {SOURCE}")
    print(f"[plot] wrote {args.out}.pdf and {args.out}.png")
    if SOURCE.startswith("fallback"):
        print("[plot] WARNING: earthquake_utils not importable -- the timing panel "
              "may not match the workflow actually being run.", file=sys.stderr)


if __name__ == "__main__":
    main()