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

SOLAR_COST_PER_MW = 200_000       # EUR/MW (Handmer's assumption)
BATTERY_COST_PER_MWH = 200_000    # EUR/MWh (Handmer's assumption)
BATTERY_EFFICIENCY = 0.90         # Round-trip efficiency
BATTERY_CYCLES = 5000             # Cycle life
SYSTEM_LIFETIME_YEARS = 25        # Years

# Grid connection costs (Denmark)
GRID_CONNECTION_DKK_PER_MW = 2_000_000  # 2M DKK/MW
DKK_TO_EUR = 0.134
GRID_CONNECTION_EUR_PER_MW = GRID_CONNECTION_DKK_PER_MW * DKK_TO_EUR  # ~268k EUR/MW

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
    Calculate total system cost for off-grid solar+battery.
    Replicates Handmer's AllInSystemCost function.
    """
    result = simulate_offgrid(solar_cf, array_size_mw, battery_size_mwh)

    array_cost = array_size_mw * solar_cost
    batt_cost = battery_size_mwh * battery_cost
    load_cost = load_cost_per_mw  # For 1 MW load
    power_system_cost = array_cost + batt_cost
    total_cost = power_system_cost + load_cost

    if result["utilization"] > 0:
        cost_per_util = total_cost / result["utilization"]
    else:
        cost_per_util = float("inf")

    return {
        "array_size_mw": array_size_mw,
        "battery_size_mwh": battery_size_mwh,
        "array_cost": array_cost,
        "battery_cost": batt_cost,
        "load_cost": load_cost,
        "power_system_cost": power_system_cost,
        "total_cost": total_cost,
        "cost_per_utilization": cost_per_util,
        "utilization": result["utilization"],
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

    # Danish grid fees and tariffs for large loads (60kV DSO connection)
    # Energinet transmission + system tariff: ~6 EUR/MWh
    # DSO capacity tariff (N1/Norlys, 60kV): ~15-25 kEUR/MW/yr → ~2-3 EUR/MWh at baseload
    # Electricity tax (reduced rate for data centers): ~5 EUR/MWh (0.4 DKK/kWh)
    # PSO abolished in 2022
    # For TSO-connected loads (>80MW): no DSO fee, total ~11 EUR/MWh
    grid_tariffs_eur_mwh = 13  # Total grid fees EUR/MWh (60kV DSO connection)
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
# HYBRID: SOLAR + BATTERY + GRID (CPLEX optimization)
# ============================================================

def optimize_hybrid_scipy(solar_cf, spot_prices_eur_mwh, load_mw=1.0,
                           solar_cost=SOLAR_COST_PER_MW,
                           battery_cost=BATTERY_COST_PER_MWH,
                           max_solar_mw=20.0, max_battery_mwh=50.0,
                           sample_hours=None):
    """
    Scipy linprog optimization for hybrid solar+battery+grid system.
    Minimizes total cost = CapEx(solar+battery) + grid connection + electricity purchases.
    All costs in EUR.

    Uses hourly resolution to keep the LP tractable.
    """
    discount_rate = 0.05
    npv_factor = sum(1 / (1 + discount_rate) ** y for y in range(SYSTEM_LIFETIME_YEARS))

    # Resample solar CF to hourly if needed
    if len(solar_cf) > 9000:
        # 5-min data -> hourly
        n_hours = len(solar_cf) // 12
        solar_hourly = np.array([solar_cf[i*12:(i+1)*12].mean() for i in range(n_hours)])
    else:
        solar_hourly = np.array(solar_cf)

    # Match spot prices length
    n = min(len(solar_hourly), len(spot_prices_eur_mwh))
    if sample_hours and sample_hours < n:
        # Sample representative hours for speed
        idx = np.linspace(0, n - 1, sample_hours, dtype=int)
        solar_hourly = solar_hourly[idx]
        spot = spot_prices_eur_mwh[idx]
        n = sample_hours
        scale = 8760 / sample_hours
    else:
        solar_hourly = solar_hourly[:n]
        spot = spot_prices_eur_mwh[:n]
        scale = 1.0

    # Limit time steps for tractability
    if n > 2000:
        idx = np.linspace(0, n - 1, 2000, dtype=int)
        solar_hourly = solar_hourly[idx]
        spot = spot[idx]
        scale = scale * n / 2000
        n = 2000

    print(f"  Scipy hybrid optimization with {n} time steps...")

    sqrt_eff = np.sqrt(BATTERY_EFFICIENCY)

    # Decision variables layout:
    # [solar_cap, batt_cap, grid_buy(0..n-1), batt_charge(0..n-1),
    #  batt_discharge(0..n-1), batt_soc(0..n-1), solar_curtail(0..n-1)]
    # Total: 2 + 5*n variables

    n_vars = 2 + 5 * n
    # Indices
    IDX_SOLAR = 0
    IDX_BATT = 1
    IDX_GRID = 2
    IDX_CHARGE = 2 + n
    IDX_DISCHARGE = 2 + 2 * n
    IDX_SOC = 2 + 3 * n
    IDX_CURTAIL = 2 + 4 * n

    # Objective: minimize annualized cost
    # capex_solar/lifetime + capex_batt/lifetime + capex_grid/lifetime + grid_elec_cost
    c = np.zeros(n_vars)
    c[IDX_SOLAR] = solar_cost / SYSTEM_LIFETIME_YEARS
    c[IDX_BATT] = battery_cost / SYSTEM_LIFETIME_YEARS
    # Grid connection cost is constant (depends on load_mw, not a decision variable)
    # We add it after optimization

    # Grid tariffs (Danish 60kV connection)
    grid_tariffs_eur = 13  # EUR/MWh (Energinet + DSO + electricity tax)
    for t in range(n):
        c[IDX_GRID + t] = (spot[t] + grid_tariffs_eur) * scale / n * 8760

    # Bounds
    bounds = []
    bounds.append((0, max_solar_mw))    # solar_cap
    bounds.append((0, max_battery_mwh))  # batt_cap
    for t in range(n):  # grid_buy
        bounds.append((0, None))
    for t in range(n):  # batt_charge
        bounds.append((0, None))
    for t in range(n):  # batt_discharge
        bounds.append((0, None))
    for t in range(n):  # batt_soc
        bounds.append((0, None))
    for t in range(n):  # solar_curtail
        bounds.append((0, None))

    # Equality constraints: A_eq @ x = b_eq
    # 1) Power balance for each t:
    #    solar_hourly[t] * solar_cap + grid_buy[t] + batt_discharge[t]
    #    == load_mw + batt_charge[t] + solar_curtail[t]
    # 2) SOC dynamics for each t:
    #    t==0: batt_soc[0] == 0.5*batt_cap + sqrt_eff*batt_charge[0] - batt_discharge[0]/sqrt_eff
    #    t>0:  batt_soc[t] == batt_soc[t-1] + sqrt_eff*batt_charge[t] - batt_discharge[t]/sqrt_eff

    n_eq = 2 * n
    A_eq = np.zeros((n_eq, n_vars))
    b_eq = np.zeros(n_eq)

    for t in range(n):
        # Power balance row
        row_pb = t
        A_eq[row_pb, IDX_SOLAR] = solar_hourly[t]      # solar_cap coefficient
        A_eq[row_pb, IDX_GRID + t] = 1.0                # grid_buy[t]
        A_eq[row_pb, IDX_DISCHARGE + t] = 1.0           # batt_discharge[t]
        A_eq[row_pb, IDX_CHARGE + t] = -1.0             # -batt_charge[t]
        A_eq[row_pb, IDX_CURTAIL + t] = -1.0            # -solar_curtail[t]
        b_eq[row_pb] = load_mw

        # SOC dynamics row
        row_soc = n + t
        A_eq[row_soc, IDX_SOC + t] = 1.0                # batt_soc[t]
        A_eq[row_soc, IDX_CHARGE + t] = -sqrt_eff       # -sqrt_eff * batt_charge[t]
        A_eq[row_soc, IDX_DISCHARGE + t] = 1.0 / sqrt_eff  # +batt_discharge[t]/sqrt_eff
        if t == 0:
            A_eq[row_soc, IDX_BATT] = -0.5              # -0.5 * batt_cap
            b_eq[row_soc] = 0.0
        else:
            A_eq[row_soc, IDX_SOC + t - 1] = -1.0       # -batt_soc[t-1]
            b_eq[row_soc] = 0.0

    # Inequality constraints: A_ub @ x <= b_ub
    # For each t:
    #   batt_soc[t] <= batt_cap  =>  batt_soc[t] - batt_cap <= 0
    #   batt_charge[t] <= batt_cap  =>  batt_charge[t] - batt_cap <= 0
    #   batt_discharge[t] <= batt_cap  =>  batt_discharge[t] - batt_cap <= 0
    n_ineq = 3 * n
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    for t in range(n):
        # SOC <= batt_cap
        row = t
        A_ub[row, IDX_SOC + t] = 1.0
        A_ub[row, IDX_BATT] = -1.0

        # charge <= batt_cap
        row = n + t
        A_ub[row, IDX_CHARGE + t] = 1.0
        A_ub[row, IDX_BATT] = -1.0

        # discharge <= batt_cap
        row = 2 * n + t
        A_ub[row, IDX_DISCHARGE + t] = 1.0
        A_ub[row, IDX_BATT] = -1.0

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if result.success:
        x = result.x
        solar_opt = x[IDX_SOLAR]
        batt_opt = x[IDX_BATT]
        grid_purchases = x[IDX_GRID:IDX_GRID + n]
        annual_grid_mwh = np.sum(grid_purchases) * scale / n * 8760

        capex_grid = GRID_CONNECTION_EUR_PER_MW * load_mw / SYSTEM_LIFETIME_YEARS
        total_annual = result.fun + capex_grid

        return {
            "solar_mw": solar_opt,
            "battery_mwh": batt_opt,
            "annual_grid_purchase_mwh": annual_grid_mwh,
            "solar_cost": solar_opt * solar_cost,
            "battery_cost": batt_opt * battery_cost,
            "grid_connection_cost": GRID_CONNECTION_EUR_PER_MW * load_mw,
            "annual_electricity_cost": annual_grid_mwh * (np.mean(spot) + 25),
            "total_annual_cost": total_annual,
            "status": "optimal",
        }
    else:
        return {"status": "infeasible"}


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
            "offgrid_power_system_cost": offgrid["power_system_cost"],
            "offgrid_total_cost": offgrid["total_cost"],
            "offgrid_cost_per_util": offgrid["cost_per_utilization"],
            "offgrid_utilization": offgrid["utilization"],
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

    # Also run hybrid optimization for a data center case
    print("\n" + "=" * 70)
    print("Running scipy hybrid optimization (Denmark, DC load)...")
    print("=" * 70)
    hybrid = optimize_hybrid_scipy(dk_solar, spot_prices, load_mw=1.0,
                                    sample_hours=2000)
    print(f"\nHybrid optimal: Solar={hybrid.get('solar_mw', 0):.2f}MW, "
          f"Battery={hybrid.get('battery_mwh', 0):.2f}MWh")
    print(f"Annual grid purchase: {hybrid.get('annual_grid_purchase_mwh', 0):.0f} MWh")
    print(f"Total annual cost: EUR {hybrid.get('total_annual_cost', 0):,.0f}")

    with open("data/hybrid_result.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in hybrid.items()}, f, indent=2)

    print("\nAll results saved to data/")
    print("Run generate_report.py to create the PDF report.")
