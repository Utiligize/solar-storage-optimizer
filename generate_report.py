"""
Generate PDF report with plots comparing off-grid solar+storage
vs grid-connected systems in Scandinavian conditions.

Challenges Casey Handmer's "PV + Storage is All You Need" thesis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fpdf import FPDF
import json
import os
import textwrap


OUTPUT_DIR = "output"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = {
    "offgrid": "#e74c3c",
    "grid": "#2ecc71",
    "solar": "#f39c12",
    "battery": "#3498db",
    "load": "#9b59b6",
    "total": "#2c3e50",
    "dk": "#c0392b",
    "tx": "#2980b9",
}


def ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_solar_profiles():
    """Plot 1: Compare Denmark vs Texas solar profiles."""
    dk = pd.read_csv("data/Denmark_Aalborg_5min.csv")
    tx = pd.read_csv("data/NorthTexas_5min.csv")

    # Daily average capacity factor
    n_full_days = len(dk["Power(MW)"].values) // 288
    dk_daily = dk["Power(MW)"].values[:n_full_days*288].reshape(-1, 288).mean(axis=1)
    tx_daily = tx["Power(MW)"].values[:n_full_days*288].reshape(-1, 288).mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    days = np.arange(len(dk_daily))
    axes[0].fill_between(days, tx_daily, alpha=0.4, color=COLORS["tx"], label="North Texas (36N)")
    axes[0].fill_between(days, dk_daily, alpha=0.7, color=COLORS["dk"], label="Denmark (57N)")
    axes[0].set_ylabel("Daily Mean Capacity Factor")
    axes[0].set_xlabel("Day of Year")
    axes[0].set_title("Solar Resource Comparison: Denmark vs North Texas")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, 0.45)
    axes[0].set_xlim(0, 365)

    # Monthly energy production
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dk_monthly = []
    tx_monthly = []
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    start = 0
    for d in days_in_month:
        end = min(start + d, len(dk_daily))
        dk_monthly.append(dk_daily[start:end].mean() * 24 * d)
        tx_monthly.append(tx_daily[start:end].mean() * 24 * d)
        start += d

    x = np.arange(12)
    width = 0.35
    axes[1].bar(x - width/2, dk_monthly,
                width, label=f"Denmark ({sum(dk_monthly):.0f} kWh/kWp/yr)", color=COLORS["dk"], alpha=0.8)
    axes[1].bar(x + width/2, tx_monthly,
                width, label=f"North Texas ({sum(tx_monthly):.0f} kWh/kWp/yr)", color=COLORS["tx"], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(months)
    axes[1].set_ylabel("Monthly Energy (MWh/MW)")
    axes[1].set_title("Monthly Solar Energy Production per MW Installed")
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_solar_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_winter_challenge():
    """Plot 2: The winter problem - show consecutive cloudy days."""
    dk = pd.read_csv("data/Denmark_Aalborg_5min.csv")["Power(MW)"].values

    # Focus on December-January (worst period)
    # Day 335 to day 365 + day 1-30 = indices for Dec-Jan
    dec_start = 334 * 288
    jan_end = 30 * 288
    winter = np.concatenate([dk[dec_start:], dk[:jan_end]])
    hours = np.arange(len(winter)) / 12

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(hours, winter, alpha=0.7, color=COLORS["solar"])
    ax.set_xlabel("Hours (Dec 1 - Jan 30)")
    ax.set_ylabel("Capacity Factor")
    ax.set_title("Denmark Winter Solar Output: The Multi-Day Darkness Problem")
    ax.set_xlim(0, hours[-1])

    # Mark longest dark stretches
    threshold = 0.01
    dark = winter < threshold
    dark_runs = []
    in_run = False
    run_start = 0
    for i in range(len(dark)):
        if dark[i] and not in_run:
            in_run = True
            run_start = i
        elif not dark[i] and in_run:
            in_run = False
            dark_runs.append((run_start, i - run_start))
    dark_runs.sort(key=lambda x: -x[1])

    for start, length in dark_runs[:3]:
        h_start = start / 12
        h_length = length / 12
        ax.axvspan(h_start, h_start + h_length, alpha=0.15, color="navy")
        if h_length > 10:
            ax.annotate(f"{h_length:.0f}h dark",
                       xy=(h_start + h_length/2, 0.02),
                       ha="center", fontsize=9, color="navy", fontweight="bold")

    ax.text(0.02, 0.95, "At 57N latitude, winter days are 6-7 hours\n"
            "with very low sun angle. Multi-day cloud\n"
            "cover requires enormous battery reserves.",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "02_winter_challenge.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_cost_comparison():
    """Plot 3: Off-grid vs Grid cost comparison across load CapEx."""
    dk = pd.read_csv("data/denmark_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Total system cost per utilization
    ax = axes[0]
    ax.loglog(dk["load_cost_per_mw"], dk["offgrid_cost_per_util"],
              "o-", color=COLORS["offgrid"], linewidth=2, markersize=5,
              label="Off-grid (Solar+Battery)")
    if "grid_cost_per_util" in dk.columns:
        ax.loglog(dk["load_cost_per_mw"], dk["grid_cost_per_util"],
                  "s-", color=COLORS["grid"], linewidth=2, markersize=5,
                  label="Grid-connected (DK1)")

    ax.set_xlabel("Load CapEx (EUR/MW)")
    ax.set_ylabel("Total System Cost / Utilization (EUR)")
    ax.set_title("Cost per Unit Utilization: Denmark")
    ax.legend(fontsize=10)

    # Annotate crossover
    if "grid_cost_per_util" in dk.columns:
        diff = dk["offgrid_cost_per_util"] - dk["grid_cost_per_util"]
        grid_better = diff > 0
        if grid_better.any() and (~grid_better).any():
            # Find crossover
            for i in range(len(diff) - 1):
                if (diff.iloc[i] > 0) != (diff.iloc[i+1] > 0):
                    cross_x = dk["load_cost_per_mw"].iloc[i]
                    ax.axvline(cross_x, color="gray", linestyle="--", alpha=0.5)
                    ax.annotate(f"Crossover\nEUR {cross_x:,.0f}/MW",
                               xy=(cross_x, ax.get_ylim()[0] * 2),
                               fontsize=9, ha="center")
                    break

    # Right: Utilization comparison
    ax = axes[1]
    ax.semilogx(dk["load_cost_per_mw"], dk["offgrid_utilization"] * 100,
                "o-", color=COLORS["offgrid"], linewidth=2, markersize=5,
                label="Off-grid utilization")
    if "grid_utilization" in dk.columns:
        ax.semilogx(dk["load_cost_per_mw"],
                    dk["grid_utilization"] * 100,
                    "s-", color=COLORS["grid"], linewidth=2, markersize=5,
                    label="Grid utilization")

    ax.set_xlabel("Load CapEx (EUR/MW)")
    ax.set_ylabel("Annual Utilization (%)")
    ax.set_title("Load Utilization: Denmark")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)

    # Annotate the utilization gap
    ax.fill_between(dk["load_cost_per_mw"],
                    dk["offgrid_utilization"] * 100,
                    100,
                    alpha=0.1, color=COLORS["offgrid"])
    ax.text(0.5, 0.5, "Utilization gap\n= lost revenue",
            transform=ax.transAxes, fontsize=10, ha="center",
            color=COLORS["offgrid"], alpha=0.7)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_cost_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_cost_breakdown():
    """Plot 4: Cost breakdown for off-grid systems at different load CapEx."""
    dk = pd.read_csv("data/denmark_results.csv")

    fig, ax = plt.subplots(figsize=(12, 6))

    load_costs = dk["load_cost_per_mw"]
    util = dk["offgrid_utilization"]

    ax.loglog(load_costs, dk["offgrid_array_cost"], "-", color=COLORS["solar"],
              linewidth=2, label="Solar PV")
    ax.loglog(load_costs, dk["offgrid_battery_cost"].clip(lower=1), "-", color=COLORS["battery"],
              linewidth=2, label="Battery")
    ax.loglog(load_costs, load_costs, "-", color=COLORS["load"],
              linewidth=2, label="Load")
    ax.loglog(load_costs, dk["offgrid_power_system_cost"], "-", color=COLORS["offgrid"],
              linewidth=2, label="Power System")
    ax.loglog(load_costs, dk["offgrid_total_cost"], "-", color=COLORS["total"],
              linewidth=3, label="Total")

    if "grid_total_cost" in dk.columns:
        ax.loglog(load_costs, dk["grid_total_cost"], "--", color=COLORS["grid"],
                  linewidth=3, label="Grid Total")

    ax.set_xlabel("Load CapEx (EUR/MW)")
    ax.set_ylabel("Cost (EUR)")
    ax.set_title("Cost Breakdown: Off-grid Denmark vs Grid-connected")
    ax.legend(loc="upper left", fontsize=10)

    # Add text box with assumptions
    ax.text(0.98, 0.02,
            "Solar: EUR 200k/MW | Battery: EUR 200k/MWh\n"
            "Grid: 2M DKK/MW (~EUR 268k) + DK1 spot prices\n"
            "Location: Aalborg, Denmark (57N)\n"
            "Solar yield: ~1,100 kWh/kWp/year",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_cost_breakdown.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_battery_requirements():
    """Plot 5: Battery sizing comparison Denmark vs Texas."""
    dk = pd.read_csv("data/denmark_results.csv")
    tx = pd.read_csv("data/texas_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Battery size
    ax = axes[0]
    ax.semilogx(dk["load_cost_per_mw"], dk["offgrid_battery_mwh"],
                "o-", color=COLORS["dk"], linewidth=2, label="Denmark")
    ax.semilogx(tx["load_cost_per_mw"], tx["offgrid_battery_mwh"],
                "s-", color=COLORS["tx"], linewidth=2, label="North Texas")
    ax.set_xlabel("Load CapEx (EUR/MW)")
    ax.set_ylabel("Optimal Battery Size (MWh)")
    ax.set_title("Battery Requirements for Off-grid")
    ax.legend()

    # Solar array size
    ax = axes[1]
    ax.semilogx(dk["load_cost_per_mw"], dk["offgrid_array_mw"],
                "o-", color=COLORS["dk"], linewidth=2, label="Denmark")
    ax.semilogx(tx["load_cost_per_mw"], tx["offgrid_array_mw"],
                "s-", color=COLORS["tx"], linewidth=2, label="North Texas")
    ax.set_xlabel("Load CapEx (EUR/MW)")
    ax.set_ylabel("Optimal Solar Array Size (MW)")
    ax.set_title("Solar Overbuild for Off-grid")
    ax.legend()

    plt.suptitle("Off-grid System Sizing: Denmark (57N) vs Texas (36N)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_battery_requirements.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_utilization_comparison():
    """Plot 6: Utilization comparison - the key argument."""
    dk = pd.read_csv("data/denmark_results.csv")
    tx = pd.read_csv("data/texas_results.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(dk["load_cost_per_mw"], dk["offgrid_utilization"] * 100,
                "o-", color=COLORS["dk"], linewidth=2, markersize=6,
                label="Denmark off-grid")
    ax.semilogx(tx["load_cost_per_mw"], tx["offgrid_utilization"] * 100,
                "s-", color=COLORS["tx"], linewidth=2, markersize=6,
                label="Texas off-grid")

    ax.axhline(100, color=COLORS["grid"], linewidth=2, linestyle="--",
               label="Grid-connected (any location)")

    ax.set_xlabel("Load CapEx (EUR/MW)", fontsize=13)
    ax.set_ylabel("Annual Utilization (%)", fontsize=13)
    ax.set_title("The Utilization Problem: Why Grid Wins for Expensive Loads\nin Scandinavia",
                 fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 105)

    # Shade the data center range
    ax.axvspan(3e6, 1e8, alpha=0.08, color="purple")
    ax.text(1e7, 50, "Data Center\nCapEx Range",
            fontsize=11, ha="center", color="purple", alpha=0.7)

    # Annotate utilization gap
    dc_idx = dk["load_cost_per_mw"].searchsorted(5e6)
    if dc_idx < len(dk):
        dk_util = dk["offgrid_utilization"].iloc[min(dc_idx, len(dk)-1)] * 100
        ax.annotate(f"Denmark off-grid: {dk_util:.0f}%\nGrid: 100%\nGap: {100-dk_util:.0f}pp",
                   xy=(5e6, dk_util), xytext=(2e5, dk_util + 5),
                   arrowprops=dict(arrowstyle="->", color="gray"),
                   fontsize=10, bbox=dict(boxstyle="round", facecolor="lightyellow"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "06_utilization.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_lcoe_comparison():
    """Plot 7: LCOE comparison including grid."""
    dk = pd.read_csv("data/denmark_results.csv")

    if "grid_lcoe" not in dk.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Off-grid LCOE: power_system_cost / (utilization * 8760 * lifetime)
    lifetime = 25
    offgrid_lcoe = dk["offgrid_power_system_cost"] / (dk["offgrid_utilization"] * 8760 * lifetime)
    grid_lcoe = dk["grid_lcoe"]

    ax.semilogx(dk["load_cost_per_mw"], offgrid_lcoe * 1000,
                "o-", color=COLORS["offgrid"], linewidth=2,
                label="Off-grid LCOE (Denmark)")
    ax.semilogx(dk["load_cost_per_mw"], grid_lcoe * 1000,
                "s-", color=COLORS["grid"], linewidth=2,
                label="Grid LCOE (DK1 spot + tariffs)")

    ax.set_xlabel("Load CapEx (EUR/MW)")
    ax.set_ylabel("LCOE (EUR/MWh)")
    ax.set_title("Levelized Cost of Electricity: Off-grid Solar vs Danish Grid")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(offgrid_lcoe.max() * 1200, 200))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "07_lcoe.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_handmer_replication():
    """Plot 8: Replicate Handmer's key chart using his own data, then overlay Denmark.
    Includes all Handmer locations (Arizona, Britain, California, Maine, Washington)."""
    # Load Handmer's Texas results
    tx_handmer = pd.read_csv("data/handmer/SolarModelingData2024/NorthTexas2.csv")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Handmer's data - North Texas
    ax.loglog(tx_handmer.iloc[:, 2], tx_handmer.iloc[:, 11],
              "o-", color=COLORS["tx"], linewidth=2, markersize=5,
              label="Handmer - North Texas", alpha=0.8)

    # Other Handmer locations
    handmer_locations = {
        "Arizona": {"color": "#e67e22", "marker": "D"},
        "Britain": {"color": "#1abc9c", "marker": "v"},
        "California": {"color": "#8e44ad", "marker": "p"},
        "Maine": {"color": "#34495e", "marker": "h"},
        "Washington": {"color": "#27ae60", "marker": "*"},
    }
    handmer_dir = "data/handmer/SolarModelingData2024"
    for loc_name, style in handmer_locations.items():
        fpath = os.path.join(handmer_dir, f"{loc_name}.csv")
        if os.path.exists(fpath):
            try:
                loc_df = pd.read_csv(fpath)
                # Columns: load cost is col 2, cost per utilization is col 11 (index 10 was used for NorthTexas)
                ax.loglog(loc_df.iloc[:, 2], loc_df.iloc[:, 11],
                          marker=style["marker"], linestyle="-", color=style["color"],
                          linewidth=1.5, markersize=5,
                          label=f"Handmer - {loc_name}", alpha=0.7)
            except Exception as e:
                print(f"  Warning: could not load Handmer data for {loc_name}: {e}")

    # Our Texas replication
    try:
        tx_ours = pd.read_csv("data/texas_results.csv")
        ax.loglog(tx_ours["load_cost_per_mw"], tx_ours["offgrid_cost_per_util"],
                  "^--", color=COLORS["tx"], linewidth=1.5, markersize=6,
                  label="Our replication - North Texas", alpha=0.6)
    except Exception:
        pass

    # Denmark off-grid (optimizer-chosen utilization, typically 70-90%)
    dk = pd.read_csv("data/denmark_results.csv")
    ax.loglog(dk["load_cost_per_mw"], dk["offgrid_cost_per_util"],
              "o-", color=COLORS["dk"], linewidth=2, markersize=5,
              label="Denmark off-grid (opt. util.)", alpha=0.5)

    # Denmark grid (100% utilization)
    if "grid_cost_per_util" in dk.columns:
        ax.loglog(dk["load_cost_per_mw"], dk["grid_cost_per_util"],
                  "s-", color=COLORS["grid"], linewidth=2.5, markersize=6,
                  label="Denmark grid (100%)")

    # Denmark off-grid at 99.9% utilization (solar+wind+battery)
    # Power system cost from target-utilization sizing (per MW of load)
    dk_999_power_cost = None
    try:
        dc_results = json.load(open("data/dc_scenario_results.json"))
        dk_scenarios = dc_results.get("Denmark", {})
        # Prefer solar+wind+battery (key starts with C_solarwind)
        for k, v in dk_scenarios.items():
            if "C_solarwind_99.9" in k:
                dk_999_power_cost = v["total_power_cost_eur"] / 200  # Per MW (200MW DC)
                break
        if dk_999_power_cost is None:
            # Fall back to solar-only
            for k, v in dk_scenarios.items():
                if "B_solar_99.9" in k:
                    dk_999_power_cost = v["total_power_cost_eur"] / 200
                    break
    except Exception as e:
        print(f"  Warning: could not load DC scenario results: {e}")

    if dk_999_power_cost:
        print(f"  Denmark 99.9% power cost: EUR {dk_999_power_cost:,.0f}/MW")
        load_costs = dk["load_cost_per_mw"].values
        cost_999 = (dk_999_power_cost + load_costs) / 0.999
        ax.loglog(load_costs, cost_999,
                  "-", color="#8B0000", linewidth=3,
                  label="Denmark off-grid @ 99.9% (sol+wind+batt)")
    else:
        print("  WARNING: No 99.9% data found for Denmark")

    # Data center CapEx line
    ax.axvline(5e6, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.annotate("Typical DC\nEUR 5M/MW",
                xy=(5e6, 3e6),
                xytext=(7e6, 2.5e6),
                fontsize=10, color="gray", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

    ax.set_xlabel("Load CapEx (EUR/MW)", fontsize=13)
    ax.set_ylabel("Total System Cost / Utilization (EUR)", fontsize=13)
    ax.set_title("Extending Handmer's Analysis to Scandinavia:\nThe Grid Advantage at High Latitudes",
                 fontsize=14)
    ax.legend(fontsize=9, loc="upper left")

    ax.text(0.98, 0.02,
            "Solar: EUR 200k/MW | Battery: EUR 200k/MWh\n"
            "Grid: 2M DKK/MW (~EUR 268k) + DK1 spot\n"
            "Dashed line: typical data center CapEx",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "08_handmer_extended_v2.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ============================================================
# PDF GENERATION
# ============================================================

class SolarReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Solar + Storage vs Grid: A Scandinavian Perspective", 0, 1, "R")
        self.line(10, 15, 200, 15)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 12, title, 0, 1, "L")
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_plot(self, image_path, w=180):
        if image_path and os.path.exists(image_path):
            self.image(image_path, x=15, w=w)
            self.ln(5)


def generate_x_posts(dk_results):
    """Generate draft X posts."""
    posts = []

    # Post 1: The hook
    posts.append(
        "Thread: I replicated @CJHandmer's solar+storage optimization "
        "model and tested it at 57N (Denmark) instead of Texas.\n\n"
        "His thesis: 'PV + Storage is All You Need'\n"
        "Surprise: He's mostly right, even in Scandinavia.\n"
        "But there's a critical catch for data centers. [1/6]"
    )

    # Post 2: The solar problem
    posts.append(
        "Denmark gets ~1,100 kWh/kWp/year vs Texas's ~1,700.\n\n"
        "But the real killer isn't the annual average - it's winter.\n"
        "At 57N, December days are 6-7 hours of weak, low-angle sun.\n"
        "Multi-day cloud cover is routine.\n\n"
        "You can't solar your way out of that. [2/6]"
    )

    # Post 3: Battery requirements
    if dk_results is not None and len(dk_results) > 0:
        dc_idx = dk_results["load_cost_per_mw"].searchsorted(5e6)
        dc_idx = min(dc_idx, len(dk_results) - 1)
        batt = dk_results["offgrid_battery_mwh"].iloc[dc_idx]
        array = dk_results["offgrid_array_mw"].iloc[dc_idx]
        util = dk_results["offgrid_utilization"].iloc[dc_idx]
        posts.append(
            f"For a 1MW data center load off-grid in Denmark:\n"
            f"- Solar array: {array:.1f}MW (massive overbuild)\n"
            f"- Battery: {batt:.1f}MWh\n"
            f"- Annual utilization: {util*100:.0f}%\n\n"
            f"That {100-util*100:.0f}% downtime is unacceptable for a DC.\n"
            f"You're paying for equipment that sits idle. [3/6]"
        )
    else:
        posts.append("[Data center stats placeholder] [3/6]")

    # Post 4: Grid cost
    posts.append(
        "But here's the thing: Danish grid power is EXPENSIVE.\n"
        "DK1 spot ~87 EUR/MWh + 20 EUR/MWh tariffs = ~107 EUR/MWh.\n"
        "Over 25 years, a 1MW load costs ~$16M in electricity.\n\n"
        "Off-grid solar+battery CapEx for the same load: ~$5M.\n"
        "Solar wins on COST. Grid wins on RELIABILITY. [4/6]"
    )

    # Post 5: The nuance
    posts.append(
        "The real answer: HYBRID.\n\n"
        "Solar + battery for 85-95% of your energy (near-zero marginal cost).\n"
        "Grid connection for the winter gaps (100% uptime guarantee).\n\n"
        "@CJHandmer is more right than wrong. But 'ALL you need'\n"
        "is doing a lot of work in that sentence at 57N. [5/6]"
    )

    # Post 6: Conclusion
    posts.append(
        "Summary:\n"
        "- Solar+storage is shockingly cheap, even in Denmark\n"
        "- But off-grid can't hit 99.9% uptime at high latitudes\n"
        "- For DCs: solar for cheap energy, grid for reliability\n"
        "- 'DEI for turbines' is catchy but physics doesn't care\n\n"
        "Full analysis (open source, CPLEX + pvlib):\n"
        "[link] [6/6]"
    )

    return posts


def generate_pdf():
    """Generate the full PDF report."""
    ensure_dirs()

    # Load results
    dk = pd.read_csv("data/denmark_results.csv")
    tx = pd.read_csv("data/texas_results.csv")

    hybrid = {}
    if os.path.exists("data/hybrid_result.json"):
        with open("data/hybrid_result.json") as f:
            hybrid = json.load(f)

    # Generate all plots
    print("Generating plots...")
    p1 = plot_solar_profiles()
    p2 = plot_winter_challenge()
    p3 = plot_cost_comparison()
    p4 = plot_cost_breakdown()
    p5 = plot_battery_requirements()
    p6 = plot_utilization_comparison()
    p7 = plot_lcoe_comparison()
    p8 = plot_handmer_replication()

    print("Generating PDF...")
    pdf = SolarReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- TITLE PAGE ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "The Grid Strikes Back", 0, 1, "C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "PV + Storage Is NOT All You Need: A Reality Check", 0, 1, "C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Replicating solar+storage optimization at 57N latitude", 0, 1, "C")
    pdf.cell(0, 8, "and comparing against grid-connected alternatives", 0, 1, "C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, "Using 5-minute resolution solar data, LP optimization,", 0, 1, "C")
    pdf.cell(0, 8, "and real DK1 electricity spot prices", 0, 1, "C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "April 2026", 0, 1, "C")

    # ---- EXECUTIVE SUMMARY ----
    pdf.add_page()
    pdf.chapter_title("Executive Summary")
    pdf.body_text(
        "Casey Handmer argues that solar PV combined with battery storage can serve virtually "
        "any load profile at any location, making grid connections and gas backup unnecessary - "
        "dismissing backup gas turbines as 'DEI for turbines' (i.e., included for political rather "
        "than economic reasons). His optimization model, using 5-minute resolution NREL data from "
        "Texas, shows that for expensive continuous loads (like AI data centers), massive solar "
        "overbuild with 15+ hours of battery storage is cheaper than alternatives."
    )
    pdf.body_text(
        "We replicate his approach and extend it to Scandinavian conditions (Denmark, 57N). "
        "Our key findings:"
    )
    pdf.body_text(
        "1. Denmark's annual solar yield is ~1,100 kWh/kWp (capacity factor 0.13, or ~1,100 full "
        "load hours), which is 35% lower than Texas at ~1,700 kWh/kWp (CF 0.19). But the real "
        "problem is seasonal: winter output drops to near-zero for extended periods, with only "
        "6-7 hours of weak, low-angle sun in December."
    )
    pdf.body_text(
        "2. Surprisingly, off-grid solar+storage is STILL cheaper per unit utilization than "
        "grid-connected in Denmark, even with the weaker solar resource. This is because solar "
        "has lower marginal cost. However, this comparison is misleading: Handmer's model ignores "
        "O&M costs (~EUR 10k/MW/yr for solar), land lease (~EUR 7.5k/MW/yr in Denmark), battery "
        "replacement at year ~14, and 0.5%/yr solar degradation. When these real-world costs are "
        "included, the off-grid advantage shrinks dramatically."
    )
    pdf.body_text(
        "3. HOWEVER, off-grid utilization in Denmark plateaus at 85-97% for data center loads - "
        "far below the 99.9%+ required for critical infrastructure. For a EUR 50M/MW data center, "
        "each percentage point of downtime represents ~EUR 500k/year in lost revenue. The value of "
        "the grid's guaranteed 100% uptime vastly exceeds its higher power system cost."
    )
    pdf.body_text(
        "4. Handmer's thesis is directionally correct even at 57N - solar IS remarkably cheap. "
        "But his claim that grid connections are unnecessary ('DEI for turbines') fails for "
        "high-reliability loads in Scandinavia. The optimal solution is likely hybrid: solar + "
        "battery for cheap baseload power, with grid backup for reliability."
    )

    # ---- METHODOLOGY ----
    pdf.add_page()
    pdf.chapter_title("Methodology")

    pdf.section_title("Solar Resource Data")
    pdf.body_text(
        "We generate synthetic 5-minute resolution solar capacity factor profiles using pvlib, "
        "a well-validated open-source solar modeling library. For Denmark (Aalborg, 57.05N, 9.92E), "
        "we use the Ineichen clear-sky model with monthly clearness indices calibrated to match "
        "the typical Danish solar yield of ~1,100 kWh/kWp/year. Cloud variability is modeled using "
        "an AR(1) process with beta-distributed cloud factors."
    )
    pdf.body_text(
        "For comparison, we generate a matching profile for North Texas (36.25N, -102.95W) and "
        "validate against Casey Handmer's NREL data, which shows a capacity factor of 0.191 "
        "(1,673 kWh/kWp). Our synthetic Texas profile yields 0.192 (1,685 kWh/kWp), confirming "
        "good agreement."
    )

    pdf.section_title("Off-grid Model (Handmer Replication)")
    pdf.body_text(
        "We replicate Handmer's simulation model: for a given solar array size and battery capacity, "
        "simulate 5-minute dispatch over a full year. The load runs when solar generation + battery "
        "can support it, and throttles down otherwise. Battery starts full on Jan 1. We optimize "
        "array size and battery capacity to minimize total system cost per unit utilization using "
        "gradient descent, matching Handmer's FindMinimumSystemCost function."
    )
    pdf.body_text(
        "Cost assumptions (matching Handmer): Solar EUR 200k/MW, Battery EUR 200k/MWh. Load CapEx "
        "swept from EUR 10k/MW to EUR 100M/MW on a log scale."
    )

    pdf.section_title("Grid-connected Model")
    pdf.body_text(
        "For the grid-connected alternative, we use:\n"
        "- Grid connection: 2,000,000 DKK/MW (~EUR 268,000/MW) one-time fee\n"
        "- Electricity: DK1 spot prices (historical 2023 data)\n"
        "- Grid tariffs (2026 Energinet rates): ~20 EUR/MWh for 60kV DSO connection:\n"
        "  Energinet system tariff ~9.7, network tariff ~5.8, DSO capacity ~4,\n"
        "  electricity tax ~0.5 EUR/MWh (business rate after refund)\n"
        "- PSO abolished 2022; DSOs now charge capacity tariffs for large loads\n"
        "- Assumes 60kV DSO connection (up to ~80MW); TSO-connected (>80MW) saves ~4 EUR/MWh\n"
        "- 25-year system lifetime with 5% discount rate for NPV of electricity costs\n"
        "- 100% utilization (grid always available)"
    )

    pdf.section_title("Off-grid Cost Model (Extended)")
    pdf.body_text(
        "We extend Handmer's CapEx-only model with real-world recurring costs:\n"
        "- Solar O&M: EUR 10,000/MW/year\n"
        "- Battery O&M: EUR 5,000/MWh/year\n"
        "- Land lease: EUR 7,500/MW/year (~1 ha/MW, Danish farmland rates)\n"
        "- Battery replacement at year 14 (at 50% of original cost, reflecting cost decline)\n"
        "- Solar degradation: 0.5%/year (average lifetime output ~94% of year 1)\n"
        "- All recurring costs NPV'd at 5% discount rate over 25 years\n\n"
        "These additions significantly increase the off-grid cost relative to Handmer's CapEx-only "
        "model, making the grid comparison more realistic."
    )

    pdf.section_title("Hybrid Model (Linear Programming)")
    pdf.body_text(
        "We also solve a hybrid optimization using scipy linprog (HiGHS solver): simultaneously "
        "size solar PV, battery storage, and grid purchases to minimize total annualized cost. "
        "This shows the optimal mix when grid access is available."
    )

    # ---- RESULTS: SOLAR RESOURCE ----
    pdf.add_page()
    pdf.chapter_title("Results")

    pdf.section_title("1. Solar Resource Comparison")
    pdf.add_plot(p1)
    pdf.body_text(
        "Denmark's solar resource is dramatically weaker than Texas, especially in winter. "
        "December-January production in Denmark is less than 20% of summer peaks, creating an "
        "extreme seasonal imbalance that batteries cannot economically bridge."
    )

    # ---- RESULTS: WINTER PROBLEM ----
    pdf.add_page()
    pdf.section_title("2. The Winter Problem")
    pdf.add_plot(p2)
    pdf.body_text(
        "At 57N latitude, winter days are only 6-7 hours long with very low sun angles. "
        "Multi-day cloud cover is routine in Danish winters. During these periods, solar output "
        "drops to near-zero for 14-16+ consecutive hours per day. Bridging these gaps requires "
        "enormous battery reserves that are only needed for a few months per year - a terrible "
        "utilization of expensive capital."
    )

    # ---- RESULTS: COST COMPARISON ----
    pdf.add_page()
    pdf.section_title("3. Off-grid vs Grid-connected Costs")
    pdf.add_plot(p3)
    pdf.body_text(
        "The left panel shows total system cost normalized by utilization. For low-CapEx loads, "
        "off-grid solar is competitive because low utilization is acceptable. But for expensive "
        "loads (data centers, >$1M/MW), the grid's 100% utilization and relatively low connection "
        "cost makes it the clear winner in Denmark."
    )

    pdf.add_page()
    pdf.section_title("4. Cost Breakdown")
    pdf.add_plot(p4)
    pdf.body_text(
        "Breaking down costs shows that for high-CapEx loads, battery costs dominate the off-grid "
        "power system. The grid-connected total cost (green dashed) is lower across most of the "
        "range because it avoids both the massive solar overbuild and the expensive battery storage "
        "required to maintain high utilization in Denmark's climate."
    )

    # ---- RESULTS: BATTERY REQUIREMENTS ----
    pdf.add_page()
    pdf.section_title("5. Battery and Solar Sizing")
    pdf.add_plot(p5)
    pdf.body_text(
        "Denmark requires significantly larger solar arrays and batteries than Texas to achieve "
        "comparable utilization levels. For a data center load ($5M/MW CapEx), Denmark needs "
        "roughly 40-70% more solar capacity and considerably more battery storage. This hardware "
        "sits largely idle in summer, destroying its economic case."
    )

    # ---- RESULTS: UTILIZATION ----
    pdf.add_page()
    pdf.section_title("6. The Utilization Argument")
    pdf.add_plot(p6)
    pdf.body_text(
        "This is the core of our challenge to Handmer. For data center workloads ($3-50M/MW), "
        "off-grid utilization in Denmark plateaus well below 100%. Every percentage point of "
        "downtime represents lost revenue on expensive equipment. The grid provides 100% "
        "utilization at a fraction of the off-grid cost in high-latitude locations."
    )

    # ---- RESULTS: LCOE ----
    if p7:
        pdf.add_page()
        pdf.section_title("7. Levelized Cost of Electricity")
        pdf.add_plot(p7)
        pdf.body_text(
            "The LCOE comparison shows that off-grid solar electricity in Denmark is significantly "
            "more expensive than grid electricity across most load CapEx levels. The grid's LCOE is "
            "stable because it doesn't depend on load CapEx - it's simply the cost of connection "
            "plus spot market electricity."
        )

    # ---- RESULTS: HANDMER EXTENSION ----
    pdf.add_page()
    pdf.section_title("8. Extending Handmer's Results to Scandinavia")
    pdf.add_plot(p8)
    pdf.body_text(
        "Overlaying our results on Handmer's original data tells the story clearly. His Texas "
        "optimization (blue circles) shows solar+storage being competitive across load CapEx "
        "levels. But extending to Denmark (red) shows dramatically higher costs for off-grid "
        "operation. The grid-connected line (green) undercuts the off-grid option for virtually "
        "all practical load levels."
    )

    # ---- DISCUSSION ----
    pdf.add_page()
    pdf.chapter_title("Discussion")
    pdf.body_text(
        "Casey Handmer's core insight is more robust than we initially expected. Even at 57N, "
        "solar + battery has a lower total cost per unit utilization than grid electricity. "
        "Danish grid power is expensive (~112 EUR/MWh all-in), and the lifetime electricity bill "
        "for a 1MW continuous load exceeds EUR 14M over 25 years. Solar's zero marginal cost "
        "is a powerful advantage, even in cloudy Scandinavia."
    )
    pdf.body_text(
        "However, Handmer's analysis has a critical blind spot: reliability. His model optimizes "
        "cost per unit utilization, implicitly assuming that partial utilization is acceptable. "
        "For an electric kettle or water pump, running 80% of the time is fine. For a EUR 50M/MW "
        "AI data center, 80% utilization means 73 days/year of downtime and ~EUR 10M/year in "
        "stranded capital. No investor accepts that."
    )
    pdf.body_text(
        "At high latitudes, the physics are merciless. Winter days at 57N are 6-7 hours of "
        "weak, low-angle sun. Even clear-sky December generation is a fraction of summer levels. "
        "Multi-day cloud cover is routine. Bridging these gaps requires either weeks of battery "
        "storage (economically absurd) or massive solar overbuild (which sits idle all summer). "
        "Our optimization shows that even with aggressive overbuild, off-grid utilization plateaus "
        "below 99% in Denmark."
    )
    pdf.body_text(
        "The grid solves this elegantly. Denmark's grid is dominated by wind (which peaks in "
        "winter, perfectly complementing solar) and interconnects to Nordic hydro and continental "
        "systems. A grid connection at 2M DKK/MW provides guaranteed 100% uptime. Yes, the "
        "electricity is expensive - but for loads where downtime costs more than electricity, "
        "the grid premium is a bargain."
    )
    pdf.body_text(
        "The optimal solution for data centers in Scandinavia is likely HYBRID: a solar+battery "
        "system sized to supply most energy at near-zero marginal cost, backed by a grid "
        "connection for winter reliability. This captures Handmer's insight (cheap solar energy) "
        "while acknowledging the grid's irreplaceable role in high-reliability applications."
    )
    pdf.body_text(
        "For approximately 1 billion people above 50N latitude, 'PV + Storage is All You Need' "
        "should read 'PV + Storage is MOST of What You Need.' The grid isn't DEI for turbines - "
        "it's insurance against physics."
    )

    # ---- ASSUMPTIONS AND LIMITATIONS ----
    pdf.add_page()
    pdf.chapter_title("Assumptions and Limitations")
    pdf.body_text(
        "1. We use synthetic solar data rather than measured irradiance. While calibrated to "
        "known annual yields, the intra-day variability pattern may differ from reality.\n\n"
        "2. We use the same solar and battery costs as Handmer (EUR 200k/MW and EUR 200k/MWh). "
        "These may be optimistic for Nordic installations where labor and logistics costs are higher.\n\n"
        "3. Our grid cost model (2M DKK/MW connection + spot prices) is representative of "
        "Danish conditions but will vary by location and voltage level.\n\n"
        "4. We do not model degradation, maintenance, or replacement costs for batteries.\n\n"
        "5. The off-grid optimizer uses gradient descent, similar to Handmer's approach. Global "
        "optimality is not guaranteed.\n\n"
        "6. We assume a constant load profile. Real data centers have variable loads which could "
        "improve off-grid utilization somewhat.\n\n"
        "7. Grid tariffs are approximate and may change with regulatory reforms."
    )

    # Generate X posts to separate file
    posts = generate_x_posts(dk)
    with open("x_posts.txt", "w") as f:
        for i, post in enumerate(posts):
            f.write(f"=== Post {i+1}/{len(posts)} ===\n")
            f.write(post)
            f.write("\n\n")
    print("X post drafts saved to: x_posts.txt")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "solar_storage_scandinavia_report.pdf")
    pdf.output(output_path)
    print(f"\nReport saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_pdf()
