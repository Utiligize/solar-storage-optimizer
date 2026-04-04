"""
Generate synthetic solar irradiance data for Denmark (and other locations)
using pvlib, producing 5-minute resolution capacity factor time series.
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import os

LOCATIONS = {
    "Denmark_Aalborg": {"lat": 57.05, "lon": 9.92, "alt": 10, "tz": "Europe/Copenhagen"},
    "Denmark_Aarhus": {"lat": 56.16, "lon": 10.20, "alt": 50, "tz": "Europe/Copenhagen"},
    "NorthTexas": {"lat": 36.25, "lon": -102.95, "alt": 1100, "tz": "America/Chicago"},
}


def generate_clear_sky_5min(location_name, year=2023):
    """Generate 5-minute clear-sky solar profile using pvlib."""
    loc_params = LOCATIONS[location_name]
    location = Location(
        latitude=loc_params["lat"],
        longitude=loc_params["lon"],
        altitude=loc_params["alt"],
        tz=loc_params["tz"],
    )

    # 5-minute time index for a full year
    times = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31 23:55",
        freq="5min",
        tz=loc_params["tz"],
    )

    # Get solar position
    solpos = location.get_solarposition(times)

    # Clear sky model (Ineichen)
    cs = location.get_clearsky(times, model="ineichen")

    # PV system: fixed tilt = latitude, south-facing (azimuth=180)
    tilt = loc_params["lat"]
    azimuth = 180

    # Get POA irradiance
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
    )

    # Simple PV model: capacity factor = POA / 1000, capped at 1
    # Apply temperature derating and inverter efficiency (~96%)
    cf = (poa["poa_global"].fillna(0) / 1000.0).clip(0, 1) * 0.96

    return cf, times


def add_cloud_variability(cf_clearsky, location_name, seed=42):
    """
    Add realistic cloud cover variability based on monthly clearness indices.
    Denmark has much more cloud cover than Texas.
    """
    rng = np.random.RandomState(seed)

    # Monthly clearness index (Kt) - ratio of actual to clear-sky irradiance
    # These are approximate values from literature
    monthly_kt = {
        "Denmark_Aalborg": [0.40, 0.44, 0.50, 0.55, 0.60, 0.60, 0.58, 0.55, 0.50, 0.44, 0.40, 0.34],
        "Denmark_Aarhus": [0.40, 0.44, 0.50, 0.55, 0.60, 0.60, 0.58, 0.55, 0.50, 0.44, 0.40, 0.34],
        "NorthTexas": [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.73, 0.70, 0.68, 0.63, 0.58],
    }

    kt = monthly_kt.get(location_name, monthly_kt["Denmark_Aalborg"])
    cf_values = cf_clearsky.values.copy()
    times = cf_clearsky.index
    n = len(cf_values)

    # Generate correlated cloud patterns
    # Use AR(1) process for temporal correlation (clouds persist)
    cloud_state = np.zeros(n)
    cloud_state[0] = rng.normal(0, 1)
    rho = 0.98  # High autocorrelation for 5-min steps
    for i in range(1, n):
        cloud_state[i] = rho * cloud_state[i - 1] + np.sqrt(1 - rho**2) * rng.normal(0, 1)

    # Convert to cloud factor using monthly Kt
    for month in range(1, 13):
        mask = times.month == month
        kt_month = kt[month - 1]

        # Map cloud_state to a beta-distributed cloud factor
        # Mean = kt_month, with some variance
        cloud_sub = cloud_state[mask]
        # Normalize to [0, 1] using CDF of normal
        from scipy.stats import norm
        u = norm.cdf(cloud_sub)

        # Use beta distribution to get cloud factor with correct mean
        a = kt_month * 5
        b = (1 - kt_month) * 5
        from scipy.stats import beta as beta_dist
        cloud_factor = beta_dist.ppf(u, a, b)
        cloud_factor = np.clip(cloud_factor, 0.05, 1.0)

        cf_values[mask] *= cloud_factor

    return pd.Series(cf_values, index=times)


def generate_solar_profile(location_name, year=2023):
    """Generate a full-year 5-minute solar capacity factor profile."""
    print(f"Generating solar profile for {location_name}, {year}...")
    cf_clear, times = generate_clear_sky_5min(location_name, year)
    cf_cloudy = add_cloud_variability(cf_clear, location_name)

    # Verify capacity factor is reasonable
    annual_cf = cf_cloudy.mean()
    annual_yield = annual_cf * 8760  # kWh/kWp equivalent
    print(f"  Annual capacity factor: {annual_cf:.3f}")
    print(f"  Annual yield: {annual_yield:.0f} kWh/kWp")

    return cf_cloudy


def save_profile(cf, location_name, output_dir="data"):
    """Save profile in same format as Casey's data."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{location_name}_5min.csv")

    df = pd.DataFrame({
        "LocalTime": cf.index.strftime("%m/%d/%y %H:%M"),
        "Power(MW)": cf.values,  # Normalized to 1 MW peak
    })
    df.to_csv(filepath, index=False)
    print(f"  Saved to {filepath}")
    return filepath


if __name__ == "__main__":
    for loc in ["Denmark_Aalborg", "NorthTexas"]:
        cf = generate_solar_profile(loc)
        save_profile(cf, loc)
