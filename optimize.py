"""
Solar + Storage Optimization Model
Replicating and extending Casey Handmer's analysis.

Two scenarios compared:
1. Off-grid (Handmer): Solar + Battery only, no grid
2. Grid-connected: Grid connection + spot prices (Denmark DK1)

Uses scipy.optimize.linprog for the grid-connected optimization,
and Casey's simulation approach for the off-grid case.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import time
import os
import json


# ============================================================
# CONSTANTS AND COST ASSUMPTIONS (all EUR)
# ============================================================

SOLAR_COST_PER_MW = 200_000       # EUR/MW CapEx (Handmer's assumption)
BATTERY_COST_PER_MWH = 200_000    # EUR/MWh CapEx (Handmer's assumption)
BATTERY_EFFICIENCY = 0.90         # Round-trip efficiency
BATTERY_CYCLES = 5000             # Cycle life (~14 years at 1 cycle/day)
SYSTEM_LIFETIME_YEARS = 25        # Years

# Off-grid recurring costs (missing from Handmer's model)
SOLAR_OM_EUR_MW_YEAR = 10_000     # O&M: ~10 EUR/kW/yr
BATTERY_OM_EUR_MWH_YEAR = 5_000   # Battery O&M: ~5 EUR/kWh/yr
BATTERY_REPLACEMENT_YEAR = 14     # Replace battery once during 25-yr lifetime
BATTERY_REPLACEMENT_COST_FACTOR = 0.5  # Future batteries cost 50% of today
SOLAR_LAND_EUR_MW_YEAR = 7_500    # Land lease: ~1 ha/MW * ~7,500 EUR/ha/yr (DK farmland)
SOLAR_DEGRADATION_RATE = 0.005    # 0.5%/year output degradation

# Grid connection costs (Denmark)
GRID_CONNECTION_DKK_PER_MW = 2_000_000  # 2M DKK/MW
DKK_TO_EUR = 0.134
GRID_CONNECTION_EUR_PER_MW = GRID_CONNECTION_DKK_PER_MW * DKK_TO_EUR  # ~268k EUR/MW

# Grid tariffs by region (EUR/MWh, on top of wholesale spot price)
# Sources: Energinet 2026, ERCOT 4CP, NESO TNUoS/BSUoS, EIA state data
GRID_TARIFFS = {
    "Denmark_TSO":  16,    # Energinet system+network 15.4 + elec tax 0.5 (2026)
    "Denmark_DSO":  20,    # + DSO capacity ~4 EUR/MWh at baseload (60kV)
    "Texas":         9,    # ERCOT 4CP transmission ~8 + admin ~0.5
    "Britain":     115,    # TNUoS ~25 + BSUoS ~15 + RO/CfD/FiT ~55 + CM ~12 + CCL ~8
    "Arizona":      20,    # APS bundled delivery component
    "California":   75,    # PG&E/SCE non-generation surcharges
    "Maine":        40,    # ISO-NE transmission + delivery
    "Washington":   15,    # BPA territory, low
}

# Wind cost assumptions (onshore, 2025)
WIND_COST_PER_MW = 1_200_000      # EUR/MW CapEx (fully installed)
WIND_OM_EUR_MW_YEAR = 30_000      # O&M: ~30 EUR/kW/yr
WIND_LAND_EUR_MW_YEAR = 3_000     # Land lease (~15 ha/MW but mostly usable for agriculture)
WIND_LIFETIME_YEARS = 25          # Years

# Reference data center size
DC_LOAD_MW = 200                  # 200 MW (typical new hyperscaler facility, 2025-2026)
DC_CAPEX_EUR_PER_MW = 5_000_000   # EUR/MW (~EUR 1B for a 200MW DC)

TIMESTEP_HOURS = 5 / 60  # 5 minutes = 1/12 hour
TIMESTEPS_PER_YEAR = 365 * 24 * 12  # 105120


# ============================================================
# CASEY'S SIMULATION MODEL (Off-grid)
# ============================================================

def simulate_offgrid(solar_cf, array_size_mw, battery_size_mwh, load_mw=1.0):
    """
    Simulate off-grid solar+battery system for a full year at 5-min resolution.
    Replicates Casey Handmer's Uptime[] function.

    Args:
        solar_cf: array of capacity factors (0-1) for a 1MW array, 5-min steps
        array_size_mw: solar array capacity in MW
        battery_size_mwh: battery storage capacity in MWh
        load_mw: constant load in MW (default 1.0)

    Returns:
        dict with utilization, battery_utilization, etc.
    """
    n = len(solar_cf)
    ts = TIMESTEP_HOURS

    # Solar generation at each timestep (MW)
    solar_gen = solar_cf * array_size_mw

    # Battery state tracking
    batt = np.zeros(n + 1)
    batt[0] = battery_size_mwh  # Start full (Handmer's assumption)
    util = np.zeros(n)

    for i in range(n):
        gen = solar_gen[i]
        net = gen - load_mw  # Net power after serving load

        if net >= 0:
            # Surplus: serve load fully, charge battery with remainder
            util[i] = 1.0
            charge = min(net * ts * np.sqrt(BATTERY_EFFICIENCY),
                        battery_size_mwh - batt[i])
            batt[i + 1] = batt[i] + charge
        else:
            # Deficit: try to serve from battery
            deficit_energy = -net * ts  # Energy needed from battery (MWh)
            available = batt[i] / np.sqrt(BATTERY_EFFICIENCY) if BATTERY_EFFICIENCY > 0 else batt[i]

            if available >= deficit_energy:
                # Battery can cover full deficit
                util[i] = 1.0
                batt[i + 1] = batt[i] - deficit_energy * np.sqrt(BATTERY_EFFICIENCY)
            elif batt[i] > 0:
                # Partial load from battery + solar
                partial_load = gen + (batt[i] / np.sqrt(BATTERY_EFFICIENCY)) / ts
                util[i] = min(partial_load / load_mw, 1.0)
                batt[i + 1] = 0
            else:
                # No battery, serve what solar provides
                util[i] = min(gen / load_mw, 1.0) if load_mw > 0 else 0

    annual_utilization = np.mean(util)
    battery_utilization = 1.0 - np.mean(np.sign(batt[:n]) == 0) if battery_size_mwh > 0 else 0

    return {
        "utilization": annual_utilization,
        "battery_utilization": battery_utilization,
        "battery_trace": batt[:n],
        "util_trace": util,
    }


def offgrid_system_cost(solar_cf, array_size_mw, battery_size_mwh, load_cost_per_mw,
                         solar_cost=SOLAR_COST_PER_MW, battery_cost=BATTERY_COST_PER_MWH):
    """
    Calculate total system cost for off-grid solar+battery over 25-year lifetime.
    Extends Handmer's AllInSystemCost with real-world costs:
    - O&M for solar and battery
    - Land lease for solar array
    - Battery replacement at ~year 14
    - Solar degradation (reduces effective utilization over time)
    All costs NPV'd at 5% discount rate.
    """
    result = simulate_offgrid(solar_cf, array_size_mw, battery_size_mwh)

    discount_rate = 0.05
    npv_factor = sum(1 / (1 + discount_rate) ** y for y in range(SYSTEM_LIFETIME_YEARS))

    # CapEx (year 0)
    array_capex = array_size_mw * solar_cost
    batt_capex = battery_size_mwh * battery_cost
    load_cost = load_cost_per_mw  # For 1 MW load

    # Annual O&M (NPV over lifetime)
    solar_om_annual = array_size_mw * SOLAR_OM_EUR_MW_YEAR
    batt_om_annual = battery_size_mwh * BATTERY_OM_EUR_MWH_YEAR
    land_annual = array_size_mw * SOLAR_LAND_EUR_MW_YEAR
    annual_opex = solar_om_annual + batt_om_annual + land_annual
    lifetime_opex = annual_opex * npv_factor

    # Battery replacement at year 14 (discounted)
    batt_replacement = (battery_size_mwh * battery_cost * BATTERY_REPLACEMENT_COST_FACTOR
                        / (1 + discount_rate) ** BATTERY_REPLACEMENT_YEAR)

    # Solar degradation: panels lose 0.5%/yr. We model this as requiring
    # slightly more panels upfront (size for mid-life output).
    # The utilization itself is NOT reduced - the system still runs,
    # just with slightly less headroom in later years.
    # We add the cost of overbuilding by 6% to compensate for avg degradation.
    degradation_overbuild = 1 / (1 - SOLAR_DEGRADATION_RATE * (SYSTEM_LIFETIME_YEARS / 2))
    array_capex *= degradation_overbuild  # Need ~6% more panels
    solar_om_annual *= degradation_overbuild
    land_annual *= degradation_overbuild
    effective_utilization = result["utilization"]  # Utilization is maintained

    power_system_cost = array_capex + batt_capex + lifetime_opex + batt_replacement
    total_cost = power_system_cost + load_cost

    if effective_utilization > 0:
        cost_per_util = total_cost / effective_utilization
    else:
        cost_per_util = float("inf")

    return {
        "array_size_mw": array_size_mw,
        "battery_size_mwh": battery_size_mwh,
        "array_cost": array_capex,
        "battery_cost": batt_capex,
        "opex_lifetime": lifetime_opex,
        "battery_replacement": batt_replacement,
        "load_cost": load_cost,
        "power_system_cost": power_system_cost,
        "total_cost": total_cost,
        "cost_per_utilization": cost_per_util,
        "utilization": effective_utilization,
        "utilization_year1": result["utilization"],
        "battery_utilization": result["battery_utilization"],
    }


def optimize_offgrid(solar_cf, load_cost_per_mw,
                      solar_cost=SOLAR_COST_PER_MW, battery_cost=BATTERY_COST_PER_MWH,
                      verbose=False):
    """
    Find optimal array and battery size for off-grid system.
    Uses gradient descent similar to Handmer's FindMinimumSystemCost.
    """
    # Initial guesses based on load cost
    ai = max(0.01, min(10.0, 1.0 + 9.0 * load_cost_per_mw / 5_000_000))
    bi = max(0.0, min(10.0, 10.0 * load_cost_per_mw / 5_000_000))

    amp = 100 + 700 * (load_cost_per_mw / 5_000_000) ** 1
    if 700_000 < load_cost_per_mw < 1_300_000:
        amp *= 3
    if load_cost_per_mw > 80_000_000:
        amp *= 0.5

    steps = 15
    best_cost = float("inf")
    best_ai, best_bi = ai, bi

    rng = np.random.RandomState(42)

    for step in range(steps):
        cost_info = offgrid_system_cost(solar_cf, ai, bi, load_cost_per_mw,
                                         solar_cost, battery_cost)
        cost = cost_info["cost_per_utilization"]

        if cost < best_cost:
            best_cost = cost
            best_ai = ai
            best_bi = bi
            if verbose:
                print(f"  Step {step}: array={ai:.3f} MW, batt={bi:.3f} MWh, "
                      f"cost/util=EUR {cost:,.0f}, util={cost_info['utilization']:.3f}")

        # Compute gradients via finite differences
        eps_a, eps_b = 0.01, 0.01
        cost_da = offgrid_system_cost(solar_cf, ai * 1.01, bi, load_cost_per_mw,
                                       solar_cost, battery_cost)["cost_per_utilization"]
        cost_db = offgrid_system_cost(solar_cf, ai, max(0, bi + eps_b), load_cost_per_mw,
                                       solar_cost, battery_cost)["cost_per_utilization"]

        grad_a = (cost_da - cost) / (ai * 0.01) if ai > 0 else 0
        grad_b = (cost_db - cost) / eps_b

        # Update with random perturbation
        bi = max(0.0, bi + amp * (0.1 + 0.9 * rng.random()) * (-1 if grad_b > 0 else 1) * 0.01)
        ai = max(0.01, ai + amp * (0.1 + 0.9 * rng.random()) * (-1 if grad_a > 0 else 1) * 0.01)

    # Final evaluation at best point
    result = offgrid_system_cost(solar_cf, best_ai, best_bi, load_cost_per_mw,
                                  solar_cost, battery_cost)
    return result


# ============================================================
# GRID-CONNECTED MODEL
# ============================================================

def calculate_grid_cost(load_mw, spot_prices_eur_mwh, lifetime_years=SYSTEM_LIFETIME_YEARS):
    """
    Calculate total cost of grid-connected data center (all values in EUR).

    Args:
        load_mw: constant load in MW
        spot_prices_eur_mwh: hourly spot prices for one year (EUR/MWh)
        lifetime_years: system lifetime

    Returns:
        dict with cost breakdown (EUR)
    """
    # Grid connection cost (one-time)
    grid_conn_cost_eur = load_mw * GRID_CONNECTION_EUR_PER_MW

    # Annual electricity cost
    hours_per_year = len(spot_prices_eur_mwh)
    annual_energy_mwh = load_mw * hours_per_year  # MWh consumed per year

    # Weighted average price (constant load)
    avg_price_eur = np.mean(spot_prices_eur_mwh)
    annual_elec_cost_eur = annual_energy_mwh * avg_price_eur

    # Total lifetime cost
    # Apply discount rate for NPV of electricity costs
    discount_rate = 0.05
    npv_factor = sum(1 / (1 + discount_rate) ** y for y in range(lifetime_years))
    lifetime_elec_cost_eur = annual_elec_cost_eur * npv_factor

    total_cost_eur = grid_conn_cost_eur + lifetime_elec_cost_eur

    # Danish grid tariffs for large loads (2026 rates, Energinet)
    # Energinet system tariff: ~9.7 EUR/MWh (7.2 ore/kWh)
    # Energinet network tariff: ~5.8 EUR/MWh (4.3 ore/kWh) + capacity fee
    # DSO capacity tariff (60kV): ~3-5 EUR/MWh at baseload
    # Electricity tax (business): ~0.5 EUR/MWh (0.4 ore/kWh after refund)
    # PSO abolished 2022
    # TSO-connected (>80MW): ~16 EUR/MWh; 60kV DSO: ~20 EUR/MWh
    grid_tariffs_eur_mwh = GRID_TARIFFS.get("Denmark_DSO", 20)
    annual_tariff_cost_eur = annual_energy_mwh * grid_tariffs_eur_mwh
    lifetime_tariff_cost_eur = annual_tariff_cost_eur * npv_factor

    total_with_tariffs = total_cost_eur + lifetime_tariff_cost_eur

    return {
        "grid_connection_cost_eur": grid_conn_cost_eur,
        "annual_elec_cost_eur": annual_elec_cost_eur,
        "annual_tariff_cost_eur": annual_tariff_cost_eur,
        "lifetime_elec_cost_eur": lifetime_elec_cost_eur,
        "lifetime_tariff_cost_eur": lifetime_tariff_cost_eur,
        "total_cost_eur": total_with_tariffs,
        "avg_spot_price_eur_mwh": avg_price_eur,
        "utilization": 1.0,  # Grid = 100% utilization always
        "lcoe_eur_mwh": total_with_tariffs / (annual_energy_mwh * lifetime_years),
    }


def grid_connected_system_cost(load_cost_per_mw, spot_prices_eur_mwh,
                                 lifetime_years=SYSTEM_LIFETIME_YEARS):
    """
    Total system cost for grid-connected operation (load + grid).
    Comparable to Handmer's total system cost metric. All values in EUR.
    """
    grid = calculate_grid_cost(1.0, spot_prices_eur_mwh, lifetime_years)

    total_cost = grid["total_cost_eur"] + load_cost_per_mw
    cost_per_util = total_cost  # utilization = 1.0 for grid

    return {
        "grid_connection_cost": grid["grid_connection_cost_eur"],
        "lifetime_electricity_cost": grid["lifetime_elec_cost_eur"] + grid["lifetime_tariff_cost_eur"],
        "load_cost": load_cost_per_mw,
        "power_system_cost": grid["total_cost_eur"],
        "total_cost": total_cost,
        "cost_per_utilization": cost_per_util,
        "utilization": 1.0,
        "lcoe_eur_mwh": grid["lcoe_eur_mwh"],
        "avg_spot_price": grid["avg_spot_price_eur_mwh"],
    }


# ============================================================
# TARGET-UTILIZATION SIZING
# ============================================================

def size_for_target_utilization(solar_cf, target_util=0.999,
                                 solar_cost=SOLAR_COST_PER_MW,
                                 battery_cost=BATTERY_COST_PER_MWH,
                                 wind_cf=None, wind_cost=WIND_COST_PER_MW,
                                 max_iter=200):
    """
    Size solar (+ optional wind) + battery to hit a target utilization.
    This is the fair comparison: what does 99.9% uptime COST off-grid?

    Uses binary search on array size and battery size to find the minimum-cost
    system that achieves the target utilization.
    """
    discount_rate = 0.05
    npv_factor = sum(1 / (1 + discount_rate) ** y for y in range(SYSTEM_LIFETIME_YEARS))
    # Degradation: size for mid-life output (year 12.5 of 25) to ensure target
    # is met on average over lifetime. Solar CF is derated to mid-life value.
    midlife_degradation = 1 - SOLAR_DEGRADATION_RATE * (SYSTEM_LIFETIME_YEARS / 2)
    has_wind = wind_cf is not None

    best_cost = float("inf")
    best_result = None

    if has_wind:
        wind_ratios = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    else:
        wind_ratios = [0]

    # Coarse array sizes, then binary search on battery
    array_sizes = np.concatenate([
        np.arange(2, 15, 1),
        np.arange(15, 40, 2.5),
        np.arange(40, 80, 5),
    ])

    for wind_ratio in wind_ratios:
        if has_wind and wind_ratio > 0:
            combined_gen = solar_cf + wind_cf * wind_ratio
        else:
            combined_gen = solar_cf

        for ai in array_sizes:
            # Binary search for minimum battery that achieves target
            batt_lo, batt_hi = 0, 500
            found = False

            # First check if even max battery is enough
            res_max = simulate_offgrid(combined_gen, ai, batt_hi)
            if res_max["utilization"] < target_util:
                continue  # This array size can't hit target even with 500 MWh

            # Check if zero battery suffices
            res_zero = simulate_offgrid(combined_gen, ai, 0)
            if res_zero["utilization"] >= target_util:
                batt_opt = 0
                result = res_zero
                found = True
            else:
                # Binary search
                for _ in range(25):
                    batt_mid = (batt_lo + batt_hi) / 2
                    res = simulate_offgrid(combined_gen, ai, batt_mid)
                    if res["utilization"] >= target_util:
                        batt_hi = batt_mid
                    else:
                        batt_lo = batt_mid
                batt_opt = batt_hi
                result = simulate_offgrid(combined_gen, ai, batt_opt)
                found = True

            if found:
                effective_util = result["utilization"]
                if has_wind and wind_ratio > 0:
                    effective_solar = ai / (1 + wind_ratio)
                    effective_wind = ai * wind_ratio / (1 + wind_ratio)
                else:
                    effective_solar = ai
                    effective_wind = 0

                # Add ~6% solar overbuild for degradation over lifetime
                deg_overbuild = 1 / (1 - SOLAR_DEGRADATION_RATE * (SYSTEM_LIFETIME_YEARS / 2))
                solar_capex = effective_solar * solar_cost * deg_overbuild
                wind_capex = effective_wind * wind_cost
                batt_capex = batt_opt * battery_cost
                solar_om = effective_solar * deg_overbuild * (SOLAR_OM_EUR_MW_YEAR + SOLAR_LAND_EUR_MW_YEAR) * npv_factor
                wind_om = effective_wind * (WIND_OM_EUR_MW_YEAR + WIND_LAND_EUR_MW_YEAR) * npv_factor
                batt_om = batt_opt * BATTERY_OM_EUR_MWH_YEAR * npv_factor
                batt_repl = batt_opt * battery_cost * BATTERY_REPLACEMENT_COST_FACTOR / (1 + discount_rate) ** BATTERY_REPLACEMENT_YEAR

                total = (solar_capex + wind_capex + batt_capex +
                         solar_om + wind_om + batt_om + batt_repl)

                if total < best_cost:
                    best_cost = total
                    best_result = {
                        "solar_mw": effective_solar,
                        "wind_mw": effective_wind,
                        "battery_mwh": batt_opt,
                        "wind_ratio": wind_ratio,
                        "solar_capex": solar_capex,
                        "wind_capex": wind_capex,
                        "battery_capex": batt_capex,
                        "opex_lifetime": solar_om + wind_om + batt_om,
                        "battery_replacement": batt_repl,
                        "total_power_cost": total,
                        "utilization": effective_util,
                        "utilization_year1": result["utilization"],
                        "target_util": target_util,
                    }
                    print(f"  -> array={ai:.1f}MW (sol={effective_solar:.1f} wind={effective_wind:.1f}), "
                          f"batt={batt_opt:.1f}MWh, util={effective_util:.4f}, cost=EUR {total:,.0f}")

    if best_result is None:
        print(f"  WARNING: Could not achieve {target_util*100:.1f}% utilization!")
        return None
    print(f"  BEST: sol={best_result['solar_mw']:.1f}MW wind={best_result['wind_mw']:.1f}MW "
          f"batt={best_result['battery_mwh']:.1f}MWh, cost=EUR {best_cost:,.0f}/MW")
    return best_result


# ============================================================
# SWEEP ACROSS LOAD CAPEX LEVELS
# ============================================================

def run_load_capex_sweep(solar_cf, spot_prices_eur_mwh=None, location_name="Unknown",
                          n_points=30, verbose=True):
    """
    Run optimization across load CapEx levels from $10k/MW to $100M/MW.
    Compare off-grid vs grid-connected.
    """
    load_costs = np.logspace(4, 8, n_points)  # EUR 10k to 100M per MW

    results = []

    for i, lc in enumerate(load_costs):
        if verbose:
            print(f"\n[{i+1}/{n_points}] Load CapEx: EUR {lc:,.0f}/MW")

        # Off-grid optimization
        t0 = time.time()
        offgrid = optimize_offgrid(solar_cf, lc, verbose=False)
        t_offgrid = time.time() - t0

        row = {
            "load_cost_per_mw": lc,
            "location": location_name,
            # Off-grid results
            "offgrid_array_mw": offgrid["array_size_mw"],
            "offgrid_battery_mwh": offgrid["battery_size_mwh"],
            "offgrid_array_cost": offgrid["array_cost"],
            "offgrid_battery_cost": offgrid["battery_cost"],
            "offgrid_opex_lifetime": offgrid["opex_lifetime"],
            "offgrid_battery_replacement": offgrid["battery_replacement"],
            "offgrid_power_system_cost": offgrid["power_system_cost"],
            "offgrid_total_cost": offgrid["total_cost"],
            "offgrid_cost_per_util": offgrid["cost_per_utilization"],
            "offgrid_utilization": offgrid["utilization"],
            "offgrid_utilization_year1": offgrid["utilization_year1"],
            "offgrid_time_s": t_offgrid,
        }

        # Grid-connected (if spot prices available)
        if spot_prices_eur_mwh is not None:
            grid = grid_connected_system_cost(lc, spot_prices_eur_mwh)
            row.update({
                "grid_connection_cost": grid["grid_connection_cost"],
                "grid_elec_cost": grid["lifetime_electricity_cost"],
                "grid_power_system_cost": grid["power_system_cost"],
                "grid_total_cost": grid["total_cost"],
                "grid_cost_per_util": grid["cost_per_utilization"],
                "grid_utilization": grid["utilization"],
                "grid_lcoe": grid["lcoe_eur_mwh"],
            })

        results.append(row)

        if verbose:
            print(f"  Off-grid: array={offgrid['array_size_mw']:.2f}MW, "
                  f"batt={offgrid['battery_size_mwh']:.2f}MWh, "
                  f"util={offgrid['utilization']:.3f}, "
                  f"cost/util=EUR {offgrid['cost_per_utilization']:,.0f}")
            if spot_prices_eur_mwh is not None:
                print(f"  Grid:     cost/util=EUR {grid['cost_per_utilization']:,.0f}, "
                      f"util=1.000")

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Solar + Storage Optimization: Challenging Handmer's Thesis")
    print("Focus: Scandinavian conditions (Denmark, DK1)")
    print("=" * 70)

    # Load solar data
    dk_solar = pd.read_csv("data/Denmark_Aalborg_5min.csv")["Power(MW)"].values
    tx_solar = pd.read_csv("data/NorthTexas_5min.csv")["Power(MW)"].values

    # Load spot prices
    spot_file = "data/dk1_spotprices_2023.csv"
    if os.path.exists(spot_file):
        spot_df = pd.read_csv(spot_file)
        if "price_eur_mwh" in spot_df.columns:
            spot_prices = spot_df["price_eur_mwh"].values
        elif "SpotPriceEUR" in spot_df.columns:
            spot_prices = spot_df["SpotPriceEUR"].values
        else:
            print(f"Columns in spot file: {spot_df.columns.tolist()}")
            spot_prices = None
        if spot_prices is not None:
            # Remove NaN/negative
            spot_prices = np.nan_to_num(spot_prices, nan=50.0)
            spot_prices = np.clip(spot_prices, 0, 500)
            print(f"\nDK1 spot prices loaded: mean={np.mean(spot_prices):.1f} EUR/MWh, "
                  f"min={np.min(spot_prices):.1f}, max={np.max(spot_prices):.1f}")
    else:
        print(f"\nWARNING: No spot price data found at {spot_file}")
        print("Using synthetic DK1 prices (mean ~80 EUR/MWh)")
        # Synthetic DK1 prices based on historical averages
        rng = np.random.RandomState(42)
        spot_prices = 80 + 30 * rng.randn(8760)
        spot_prices = np.clip(spot_prices, 5, 300)

    print(f"\nDenmark solar CF: {np.mean(dk_solar):.3f} ({np.mean(dk_solar)*8760:.0f} kWh/kWp)")
    print(f"Texas solar CF:   {np.mean(tx_solar):.3f} ({np.mean(tx_solar)*8760:.0f} kWh/kWp)")

    # Run sweeps
    print("\n" + "=" * 70)
    print("Running Denmark off-grid vs grid-connected sweep...")
    print("=" * 70)
    dk_results = run_load_capex_sweep(dk_solar, spot_prices, "Denmark", n_points=25)
    dk_results.to_csv("data/denmark_results.csv", index=False)

    print("\n" + "=" * 70)
    print("Running Texas off-grid sweep (for comparison)...")
    print("=" * 70)
    tx_results = run_load_capex_sweep(tx_solar, None, "NorthTexas", n_points=25)
    tx_results.to_csv("data/texas_results.csv", index=False)

    # ---- Concrete 200MW data center scenario ----
    def run_dc_scenario(location_name, solar_cf, spot_prices_eur=None, wind_cf=None):
        """Run the 200MW DC comparison for a given location."""
        print(f"\n{'='*80}")
        print(f"  {DC_LOAD_MW}MW DATA CENTER: {location_name}")
        print(f"{'='*80}")

        dc_load = DC_LOAD_MW
        dc_capex = DC_CAPEX_EUR_PER_MW
        targets = [0.95, 0.99, 0.999]

        print(f"Solar CF: {np.mean(solar_cf):.3f} ({np.mean(solar_cf)*8760:.0f} full load hours)")
        if wind_cf is not None:
            print(f"Wind  CF: {np.mean(wind_cf):.3f} ({np.mean(wind_cf)*8760:.0f} full load hours)")

        scenarios = {}

        # Scenario A: Grid-only (only if spot prices available)
        if spot_prices_eur is not None:
            grid = calculate_grid_cost(dc_load, spot_prices_eur)
            sc_a = {
                "name": "Grid-only (100%)",
                "solar_mw": 0, "wind_mw": 0, "battery_mwh": 0,
                "total_power_cost_eur": grid["total_cost_eur"],
                "dc_capex_eur": dc_load * dc_capex,
                "utilization": 1.0,
            }
            sc_a["total_cost_eur"] = sc_a["total_power_cost_eur"] + sc_a["dc_capex_eur"]
            scenarios["A_grid"] = sc_a

        # Off-grid scenarios at different target utilizations
        for target in targets:
            label = f"{target*100:.1f}%"

            # Solar + battery only
            print(f"\nSizing solar+battery for {label} utilization...")
            solar_result = size_for_target_utilization(solar_cf, target_util=target)
            if solar_result:
                key = f"B_solar_{label}"
                sc = {
                    "name": f"Solar+Batt @ {label}",
                    "solar_mw": solar_result["solar_mw"] * dc_load,
                    "wind_mw": 0,
                    "battery_mwh": solar_result["battery_mwh"] * dc_load,
                    "total_power_cost_eur": solar_result["total_power_cost"] * dc_load,
                    "dc_capex_eur": dc_load * dc_capex,
                    "utilization": solar_result["utilization"],
                }
                sc["total_cost_eur"] = sc["total_power_cost_eur"] + sc["dc_capex_eur"]
                scenarios[key] = sc
            else:
                print(f"  IMPOSSIBLE with solar+battery (array up to 75MW, batt up to 500MWh)")

            # Solar + wind + battery
            if wind_cf is not None:
                print(f"Sizing solar+wind+battery for {label} utilization...")
                sw_result = size_for_target_utilization(solar_cf, target_util=target,
                                                          wind_cf=wind_cf)
                if sw_result:
                    key = f"C_solarwind_{label}"
                    sc = {
                        "name": f"Solar+Wind+Batt @ {label}",
                        "solar_mw": sw_result["solar_mw"] * dc_load,
                        "wind_mw": sw_result["wind_mw"] * dc_load,
                        "battery_mwh": sw_result["battery_mwh"] * dc_load,
                        "total_power_cost_eur": sw_result["total_power_cost"] * dc_load,
                        "dc_capex_eur": dc_load * dc_capex,
                        "utilization": sw_result["utilization"],
                        "wind_ratio": sw_result["wind_ratio"],
                    }
                    sc["total_cost_eur"] = sc["total_power_cost_eur"] + sc["dc_capex_eur"]
                    scenarios[key] = sc
                else:
                    print(f"  IMPOSSIBLE even with wind")

        # Print comparison table
        print(f"\n{'Scenario':<40} {'Power Cost':>12} {'Util':>6} {'Total':>12}")
        print(f"{'':<40} {'(EUR M)':>12} {'(%)':>6} {'(EUR M)':>12}")
        print("-" * 80)
        for key, sc in scenarios.items():
            power_m = sc["total_power_cost_eur"] / 1e6
            total_m = sc["total_cost_eur"] / 1e6
            util = sc["utilization"] * 100
            extra = ""
            if sc.get("solar_mw", 0) > 0:
                extra += f" Sol {sc['solar_mw']:.0f}MW"
            if sc.get("wind_mw", 0) > 0:
                extra += f" Wnd {sc['wind_mw']:.0f}MW"
            if sc.get("battery_mwh", 0) > 0:
                extra += f" Bat {sc['battery_mwh']:.0f}MWh"
            print(f"{sc['name']:<40} {power_m:>9,.0f}   {util:>5.1f} {total_m:>9,.0f}  {extra}")
        print("-" * 80)
        print(f"DC CapEx: EUR {dc_load * dc_capex / 1e6:,.0f}M ({dc_load}MW * EUR {dc_capex/1e6:.0f}M/MW)")

        return scenarios

    # Generate synthetic wind profiles
    from scipy.stats import norm as norm_dist, beta as beta_dist
    def make_wind_profile(n_steps, mean_cf=0.30, seed=123):
        rng = np.random.RandomState(seed)
        day_of_year = np.arange(n_steps) / 288
        seasonal = mean_cf + 0.10 * np.cos(2 * np.pi * day_of_year / 365)
        noise = np.zeros(n_steps)
        noise[0] = rng.normal(0, 1)
        for i in range(1, n_steps):
            noise[i] = 0.995 * noise[i-1] + np.sqrt(1 - 0.995**2) * rng.normal(0, 1)
        u = norm_dist.cdf(noise)
        wind = np.zeros(n_steps)
        for month in range(1, 13):
            mask = (day_of_year >= (month - 1) * 30.44) & (day_of_year < month * 30.44)
            kt = np.clip(seasonal[mask].mean(), 0.05, 0.95)
            a, b = kt * 8, (1 - kt) * 8
            wind[mask] = beta_dist.ppf(u[mask], a, b)
        return np.clip(wind, 0, 1)

    dk_wind = make_wind_profile(len(dk_solar), mean_cf=0.30, seed=123)
    tx_wind = make_wind_profile(len(tx_solar), mean_cf=0.40, seed=456)

    # Run Denmark scenario
    dk_scenarios = run_dc_scenario("Denmark", dk_solar, spot_prices, dk_wind)

    # Run Texas scenario (use synthetic spot prices for Texas ~40 EUR/MWh)
    rng_tx = np.random.RandomState(99)
    tx_spot = 40 + 20 * rng_tx.randn(8760)
    tx_spot = np.clip(tx_spot, 5, 200)
    tx_scenarios = run_dc_scenario("North Texas", tx_solar, tx_spot, tx_wind)

    # Save all scenarios
    all_scenarios = {"Denmark": dk_scenarios, "NorthTexas": tx_scenarios}
    with open("data/dc_scenario_results.json", "w") as f:
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        json.dump({loc: {k: {kk: convert(vv) for kk, vv in v.items()}
                         for k, v in scenarios.items()}
                   for loc, scenarios in all_scenarios.items()}, f, indent=2)

    print("\nAll results saved to data/")
    print("Run generate_report.py to create the PDF report.")
