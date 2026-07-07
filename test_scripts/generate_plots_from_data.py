import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from the benchmark run
RESULTS = {
    "Ship Tracking": {
        "scip": {
            "time_mean": 74.74,
            "time_std": 0.76,
            "obj_mean": 527.79,
            "obj_std": 0.00
        },
        "gurobi": [
            {"gap": 0.0, "time_mean": 1.45, "time_std": 0.22, "obj_mean": 527.79, "obj_std": 0.00},
            {"gap": 1.0, "time_mean": 1.59, "time_std": 0.11, "obj_mean": 527.79, "obj_std": 0.00},
            {"gap": 2.0, "time_mean": 1.72, "time_std": 0.16, "obj_mean": 527.79, "obj_std": 0.00},
            {"gap": 5.0, "time_mean": 1.57, "time_std": 0.13, "obj_mean": 527.79, "obj_std": 0.00},
            {"gap": 10.0, "time_mean": 1.27, "time_std": 0.03, "obj_mean": 527.79, "obj_std": 0.00},
            {"gap": 20.0, "time_mean": 0.89, "time_std": 0.07, "obj_mean": 527.79, "obj_std": 0.00}
        ]
    },
    "Volcano Scheduling": {
        "scip": {
            "time_mean": 1031.74,
            "time_std": 0.55,
            "obj_mean": np.nan,
            "obj_std": np.nan
        },
        "gurobi": [
            {"gap": 0.0, "time_mean": 179.52, "time_std": 3.47, "obj_mean": 6012.08, "obj_std": 0.00},
            {"gap": 1.0, "time_mean": 181.12, "time_std": 0.90, "obj_mean": 6011.35, "obj_std": 0.00},
            {"gap": 2.0, "time_mean": 180.33, "time_std": 0.72, "obj_mean": 5906.60, "obj_std": 0.00},
            {"gap": 5.0, "time_mean": 151.93, "time_std": 1.13, "obj_mean": 5906.51, "obj_std": 0.00},
            {"gap": 10.0, "time_mean": 150.30, "time_std": 0.54, "obj_mean": 5906.45, "obj_std": 0.00},
            {"gap": 20.0, "time_mean": 149.48, "time_std": 1.23, "obj_mean": 5693.62, "obj_std": 0.00}
        ]
    }
}

output_dir = "plots"
Path(output_dir).mkdir(exist_ok=True)

prob_names = list(RESULTS.keys())

# Generate color palette dynamically
def get_colors(n):
    cmap = plt.cm.viridis
    return [cmap(i / (n-1)) for i in range(n)]

# Plot 1: Time vs Gap for each problem
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (prob_name, data) in enumerate(RESULTS.items()):
    ax = axes[idx]

    if data["gurobi"]:
        gaps = [r["gap"] for r in data["gurobi"]]
        times_mean = [r["time_mean"] for r in data["gurobi"]]
        times_std = [r["time_std"] for r in data["gurobi"]]

        ax.errorbar(gaps, times_mean, yerr=times_std, marker='o', linewidth=2,
                    markersize=8, capsize=5, capthick=2, label='Gurobi', color='#3498db')

    if data["scip"]:
        scip_mean = data["scip"]["time_mean"]
        scip_std = data["scip"]["time_std"]
        ax.axhline(scip_mean, color='red', linestyle='--', linewidth=2, label='SCIP')
        if data["gurobi"]:
            gaps = [r["gap"] for r in data["gurobi"]]
            ax.fill_between(gaps, scip_mean - scip_std, scip_mean + scip_std,
                            color='red', alpha=0.2)

    ax.set_xlabel('MIP Gap (%)', fontsize=12)
    ax.set_ylabel('Solve Time (s)', fontsize=12)
    ax.set_title(f'{prob_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/time_vs_gap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/time_vs_gap.png")

# Plot 2: Solver comparison (grouped bar chart)
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(prob_names))
num_gaps = len(RESULTS[prob_names[0]]["gurobi"])
width = 0.8 / (num_gaps + 1)  # Dynamic width

# SCIP bars
for i, pname in enumerate(prob_names):
    if RESULTS[pname]["scip"]:
        scip_mean = RESULTS[pname]["scip"]["time_mean"]
        scip_std = RESULTS[pname]["scip"]["time_std"]
        ax.bar(x[i] - (num_gaps/2)*width, scip_mean, width, yerr=scip_std,
               label='SCIP' if i == 0 else '', color='#e74c3c', capsize=3)

# Gurobi bars with dynamic colors
colors = get_colors(num_gaps)
for gap_idx in range(num_gaps):
    for prob_idx, pname in enumerate(prob_names):
        if RESULTS[pname]["gurobi"] and gap_idx < len(RESULTS[pname]["gurobi"]):
            r = RESULTS[pname]["gurobi"][gap_idx]
            ax.bar(x[prob_idx] + (gap_idx - num_gaps/2 + 1)*width, r["time_mean"],
                   width, yerr=r["time_std"],
                   label=f'Gurobi {r["gap"]:.0f}%' if prob_idx == 0 else '',
                   color=colors[gap_idx], capsize=3)

ax.set_xlabel('Problem', fontsize=12)
ax.set_ylabel('Solve Time (s)', fontsize=12)
ax.set_title('Solver Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(prob_names)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/solver_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/solver_comparison.png")

# Plot 3: Speedup comparison
fig, ax = plt.subplots(figsize=(10, 6))

for prob_name in prob_names:
    if RESULTS[prob_name]["scip"] and RESULTS[prob_name]["gurobi"]:
        scip_mean = RESULTS[prob_name]["scip"]["time_mean"]
        speedup_means = [scip_mean / r["time_mean"] for r in RESULTS[prob_name]["gurobi"]]
        gaps = [r["gap"] for r in RESULTS[prob_name]["gurobi"]]

        speedup_stds = []
        for r in RESULTS[prob_name]["gurobi"]:
            rel_err = np.sqrt((RESULTS[prob_name]["scip"]["time_std"]/scip_mean)**2 +
                             (r["time_std"]/r["time_mean"])**2)
            speedup_stds.append((scip_mean / r["time_mean"]) * rel_err)

        ax.errorbar(gaps, speedup_means, yerr=speedup_stds, marker='o', linewidth=2,
                   markersize=8, capsize=5, capthick=2, label=prob_name)

ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('MIP Gap (%)', fontsize=12)
ax.set_ylabel('Speedup vs SCIP', fontsize=12)
ax.set_title('Gurobi Speedup Relative to SCIP', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/speedup_comparison.png")

# Plot 4: Objective vs Gap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (prob_name, data) in enumerate(RESULTS.items()):
    ax = axes[idx]

    if data["gurobi"]:
        gaps = [r["gap"] for r in data["gurobi"]]
        obj_means = [r["obj_mean"] for r in data["gurobi"]]
        obj_stds = [r["obj_std"] for r in data["gurobi"]]

        ax.errorbar(gaps, obj_means, yerr=obj_stds, marker='o', linewidth=2,
                    markersize=8, capsize=5, capthick=2, label='Gurobi', color='#3498db')

    if data["scip"] and not np.isnan(data["scip"]["obj_mean"]):
        scip_obj_mean = data["scip"]["obj_mean"]
        scip_obj_std = data["scip"]["obj_std"]
        ax.axhline(scip_obj_mean, color='red', linestyle='--', linewidth=2, label='SCIP')
        if data["gurobi"]:
            gaps = [r["gap"] for r in data["gurobi"]]
            ax.fill_between(gaps, scip_obj_mean - scip_obj_std, scip_obj_mean + scip_obj_std,
                            color='red', alpha=0.2)

    ax.set_xlabel('MIP Gap (%)', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(f'{prob_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/objective_vs_gap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/objective_vs_gap.png")

# Plot 5: Objective comparison (grouped bar chart)
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(prob_names))
num_gaps = len(RESULTS[prob_names[0]]["gurobi"])
width = 0.8 / (num_gaps + 1)

# SCIP bars (skip if NaN)
for i, pname in enumerate(prob_names):
    if RESULTS[pname]["scip"] and not np.isnan(RESULTS[pname]["scip"]["obj_mean"]):
        scip_obj_mean = RESULTS[pname]["scip"]["obj_mean"]
        scip_obj_std = RESULTS[pname]["scip"]["obj_std"]
        ax.bar(x[i] - (num_gaps/2)*width, scip_obj_mean, width, yerr=scip_obj_std,
               label='SCIP' if i == 0 else '', color='#e74c3c', capsize=3)

# Gurobi bars
colors = get_colors(num_gaps)
for gap_idx in range(num_gaps):
    for prob_idx, pname in enumerate(prob_names):
        if RESULTS[pname]["gurobi"] and gap_idx < len(RESULTS[pname]["gurobi"]):
            r = RESULTS[pname]["gurobi"][gap_idx]
            ax.bar(x[prob_idx] + (gap_idx - num_gaps/2 + 1)*width, r["obj_mean"],
                   width, yerr=r["obj_std"],
                   label=f'Gurobi {r["gap"]:.0f}%' if prob_idx == 0 else '',
                   color=colors[gap_idx], capsize=3)

ax.set_xlabel('Problem', fontsize=12)
ax.set_ylabel('Objective Value', fontsize=12)
ax.set_title('Objective Value Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(prob_names)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/objective_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/objective_comparison.png")

# Plot 6: Time vs Objective tradeoff
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_tradeoff = get_colors(num_gaps)

for idx, prob_name in enumerate(prob_names):
    ax = axes[idx]
    data = RESULTS[prob_name]

    if data["gurobi"]:
        for i, r in enumerate(data["gurobi"]):
            ax.errorbar(r["time_mean"], r["obj_mean"],
                       xerr=r["time_std"], yerr=r["obj_std"],
                       marker='o', markersize=8, capsize=4,
                       label=f'Gurobi {r["gap"]:.0f}%',
                       color=colors_tradeoff[i])

    if data["scip"] and not np.isnan(data["scip"]["obj_mean"]):
        ax.errorbar(data["scip"]["time_mean"], data["scip"]["obj_mean"],
                   xerr=data["scip"]["time_std"], yerr=data["scip"]["obj_std"],
                   marker='s', markersize=10, capsize=4, color='red',
                   label='SCIP')

    ax.set_xlabel('Solve Time (s)', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(f'{prob_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/time_vs_objective_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/time_vs_objective_tradeoff.png")

print("\nAll plots generated successfully!")
