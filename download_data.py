"""
Download real solar and wind resource data for all study regions.

Solar: NSRDB PSM3 5-min (US), CAMS 1-min→5-min (Europe)
Wind: NREL WIND Toolkit 5-min (US), ERA5 hourly→interpolated (Europe)

Requires:
- NREL API key (free): https://developer.nrel.gov/signup/
- CDS API key (free): https://cds.climate.copernicus.eu/
"""

import os
import sys
import numpy as np
import pandas as pd
import requests
import time as time_mod

DATA_DIR = "data/resource"
os.makedirs(DATA_DIR, exist_ok=True)

# Study locations (matching Casey Handmer's + Denmark)
LOCATIONS = {
    "NorthTexas":  {"lat": 36.25, "lon": -102.95, "country": "US", "tz": "America/Chicago"},
    "Arizona":     {"lat": 33.45, "lon": -112.07, "country": "US", "tz": "America/Phoenix"},
    "California":  {"lat": 34.05, "lon": -118.25, "country": "US", "tz": "America/Los_Angeles"},
    "Maine":       {"lat": 44.80, "lon": -69.00, "country": "US", "tz": "America/New_York"},
    "Washington":  {"lat": 47.25, "lon": -120.50, "country": "US", "tz": "America/Los_Angeles"},
    "Britain":     {"lat": 52.00, "lon": -1.00,   "country": "EU", "tz": "Europe/London"},
    "Denmark":     {"lat": 57.05, "lon": 9.92,    "country": "EU", "tz": "Europe/Copenhagen"},
}

# ---------- NREL API (US solar + wind) ----------

def get_nrel_api_key():
    """Get NREL API key from environment or prompt."""
    key = os.environ.get("NREL_API_KEY")
    if not key:
        key_file = os.path.expanduser("~/.nrel_api_key")
        if os.path.exists(key_file):
            with open(key_file) as f:
                key = f.read().strip()
    if not key:
        print("NREL API key not found.")
        print("Get a free key at: https://developer.nrel.gov/signup/")
        print("Then: export NREL_API_KEY=your_key")
        print("  or: echo your_key > ~/.nrel_api_key")
        return None
    return key


def download_nsrdb_solar(location_name, lat, lon, year=2020, api_key=None):
    """Download 5-minute solar data from NSRDB PSM3."""
    outfile = os.path.join(DATA_DIR, f"{location_name}_solar_5min.csv")
    if os.path.exists(outfile):
        print(f"  {outfile} already exists, skipping.")
        return outfile

    if not api_key:
        print(f"  Skipping NSRDB download for {location_name} (no API key)")
        return None

    print(f"  Downloading NSRDB solar for {location_name} ({lat}, {lon})...")
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-5min-download.csv"
    params = {
        "api_key": api_key,
        "wkt": f"POINT({lon} {lat})",
        "names": str(year),
        "interval": "5",
        "attributes": "ghi,dni,dhi,air_temperature,wind_speed",
        "utc": "false",
        "email": "solar.storage.optimizer@example.com",
    }

    try:
        resp = requests.get(url, params=params, timeout=300)
        if resp.status_code == 200:
            with open(outfile, "w") as f:
                f.write(resp.text)
            print(f"  Saved to {outfile}")
            return outfile
        else:
            print(f"  NSRDB error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  NSRDB download failed: {e}")
        return None


def download_wind_toolkit(location_name, lat, lon, year=2012, api_key=None):
    """Download 5-minute wind data from NREL WIND Toolkit."""
    outfile = os.path.join(DATA_DIR, f"{location_name}_wind_5min.csv")
    if os.path.exists(outfile):
        print(f"  {outfile} already exists, skipping.")
        return outfile

    if not api_key:
        print(f"  Skipping WIND Toolkit download for {location_name} (no API key)")
        return None

    print(f"  Downloading WIND Toolkit for {location_name} ({lat}, {lon})...")
    url = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download"
    params = {
        "api_key": api_key,
        "lat": lat,
        "lon": lon,
        "year": year,
        "hubheight": 100,  # 100m hub height
        "email": "solar.storage.optimizer@example.com",
    }

    try:
        resp = requests.get(url, params=params, timeout=300)
        if resp.status_code == 200:
            with open(outfile, "w") as f:
                f.write(resp.text)
            print(f"  Saved to {outfile}")
            return outfile
        else:
            print(f"  WIND Toolkit error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  WIND Toolkit download failed: {e}")
        return None


# ---------- CAMS (European solar) ----------

def download_cams_solar(location_name, lat, lon, year=2023):
    """Download solar irradiance from CAMS Radiation Service via pvlib."""
    outfile = os.path.join(DATA_DIR, f"{location_name}_solar_cams.csv")
    if os.path.exists(outfile):
        print(f"  {outfile} already exists, skipping.")
        return outfile

    print(f"  Downloading CAMS solar for {location_name} ({lat}, {lon})...")
    try:
        import pvlib
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year}-12-31 23:59", tz="UTC")

        # pvlib.iotools.get_cams returns 1-min data
        data, meta = pvlib.iotools.get_cams(
            latitude=lat,
            longitude=lon,
            start=start,
            end=end,
            email="solar.storage.optimizer@example.com",
            identifier="mcclear",  # Clear-sky; use "radiation" for all-sky
        )

        # Try all-sky (actual irradiance with clouds)
        try:
            data_allsky, _ = pvlib.iotools.get_cams(
                latitude=lat,
                longitude=lon,
                start=start,
                end=end,
                email="solar.storage.optimizer@example.com",
                identifier="radiation",
            )
            data = data_allsky
        except Exception:
            print(f"  CAMS all-sky failed, using clear-sky for {location_name}")

        # Resample to 5-min
        data_5min = data.resample("5min").mean()
        data_5min.to_csv(outfile)
        print(f"  Saved to {outfile} ({len(data_5min)} rows)")
        return outfile
    except Exception as e:
        print(f"  CAMS download failed: {e}")
        return None


# ---------- ERA5 (European wind) ----------

def download_era5_wind(location_name, lat, lon, year=2023):
    """Download ERA5 hourly wind data via CDS API."""
    outfile = os.path.join(DATA_DIR, f"{location_name}_wind_era5.csv")
    if os.path.exists(outfile):
        print(f"  {outfile} already exists, skipping.")
        return outfile

    print(f"  Downloading ERA5 wind for {location_name} ({lat}, {lon})...")
    try:
        import cdsapi
        c = cdsapi.Client()

        nc_file = os.path.join(DATA_DIR, f"{location_name}_wind_era5.nc")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],
                "format": "netcdf",
            },
            nc_file,
        )

        import xarray as xr
        ds = xr.open_dataset(nc_file)
        u100 = ds["u100"].values.flatten()
        v100 = ds["v100"].values.flatten()
        ws100 = np.sqrt(u100**2 + v100**2)

        times = pd.date_range(f"{year}-01-01", periods=len(ws100), freq="h", tz="UTC")
        df = pd.DataFrame({"wind_speed_100m": ws100}, index=times)
        df.to_csv(outfile)
        os.remove(nc_file)
        print(f"  Saved to {outfile} ({len(df)} rows)")
        return outfile
    except ImportError:
        print("  ERA5 requires cdsapi + xarray. Install: pip install cdsapi xarray netCDF4")
        print("  Also configure ~/.cdsapirc with your CDS API key.")
        return None
    except Exception as e:
        print(f"  ERA5 download failed: {e}")
        return None


# ---------- Conversion to capacity factors ----------

def solar_to_capacity_factor(ghi_series, tilt=None, lat=None):
    """Convert GHI (W/m2) to PV capacity factor (0-1)."""
    # Simple model: CF = GHI / 1000 * system_efficiency
    system_efficiency = 0.85  # Inverter + wiring + soiling + temperature
    cf = (ghi_series / 1000.0).clip(0, 1) * system_efficiency
    return cf


def wind_to_capacity_factor(wind_speed_ms, rated_power_mw=1.0):
    """Convert wind speed at hub height to capacity factor using generic power curve.

    Uses a simplified power curve for a modern 5MW+ onshore turbine:
    - Cut-in: 3 m/s
    - Rated: 12 m/s
    - Cut-out: 25 m/s
    """
    ws = np.asarray(wind_speed_ms, dtype=float)
    cf = np.zeros_like(ws)

    # Cubic region (cut-in to rated)
    mask_cubic = (ws >= 3) & (ws < 12)
    cf[mask_cubic] = (ws[mask_cubic] - 3) ** 3 / (12 - 3) ** 3

    # Rated region
    mask_rated = (ws >= 12) & (ws <= 25)
    cf[mask_rated] = 1.0

    # Cut-out
    cf[ws > 25] = 0.0

    return cf


# ---------- Main ----------

def main():
    print("=" * 60)
    print("Solar + Wind Resource Data Downloader")
    print("=" * 60)

    nrel_key = get_nrel_api_key()

    for name, loc in LOCATIONS.items():
        print(f"\n--- {name} ({loc['lat']}, {loc['lon']}) ---")

        if loc["country"] == "US":
            # US: NSRDB 5-min solar + WIND Toolkit 5-min
            download_nsrdb_solar(name, loc["lat"], loc["lon"], api_key=nrel_key)
            download_wind_toolkit(name, loc["lat"], loc["lon"], api_key=nrel_key)
        else:
            # Europe: CAMS solar + ERA5 wind
            download_cams_solar(name, loc["lat"], loc["lon"])
            download_era5_wind(name, loc["lat"], loc["lon"])

    print("\n" + "=" * 60)
    print("Downloads complete. Run process_resource_data.py to convert")
    print("raw data to capacity factor profiles.")
    print("=" * 60)


if __name__ == "__main__":
    main()
