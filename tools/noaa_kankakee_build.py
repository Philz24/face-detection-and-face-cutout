import base64
import gzip
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import psychrolib

STATION_ID = "USW00004880"
STATION_NAME = "GREATER KANKAKEE AIRPORT"
SITE_ADDRESS = "2525 S Kensington Ave, Kankakee, IL 60901"
ELEVATION_M = 191.7
LOCAL_TZ = "America/Chicago"
LST_TZ = "Etc/GMT+6"  # Central Standard Time, UTC-6; LCD timestamps are LST year-round.
START_LOCAL = pd.Timestamp("2011-08-26 00:00:00", tz=LOCAL_TZ)
END_LOCAL = pd.Timestamp("2026-08-26 23:00:00", tz=LOCAL_TZ)
START_UTC = START_LOCAL.tz_convert("UTC")
END_UTC = END_LOCAL.tz_convert("UTC")
TARGET = pd.date_range(START_UTC, END_UTC, freq="h")
OUTFILE = Path("Kankakee_Kensing_NOAA_LCDv2_Hourly_WetBulb_Weather_2011-08-26_to_2026-08-26.csv")
QAFILE = Path("Kankakee_Kensing_NOAA_LCDv2_Hourly_WetBulb_QA.json")
TRANSFER_DIR = Path("transfer")

REQUESTED = [
    "HourlyDryBulbTemperature",
    "HourlyDewPointTemperature",
    "HourlyRelativeHumidity",
    "HourlyWetBulbTemperature",
]
OPTIONAL = ["HourlyStationPressure", "REPORT_TYPE", "STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION"]

# NOAA LCDv2 direct candidates. The first five preserve the URL shapes supplied by the user.
def candidate_urls(year: int):
    return [
        f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/{STATION_ID}.csv",
        f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_{STATION_ID}_{year}.csv",
        f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/by-year/{year}/LCD_{STATION_ID}_{year}.csv",
        f"https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}/{STATION_ID}.csv",
        f"https://www.ncei.noaa.gov/data/local-climatological-data-v2/access/{year}/LCD_{STATION_ID}_{year}.csv",
        # Additional official patterns seen across LCD/LCDv2 migrations.
        f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_{STATION_ID}.csv",
        f"https://www.ncei.noaa.gov/data/local-climatological-data-v2/access/{year}/{STATION_ID}.csv",
        f"https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}/72533604880.csv",
    ]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 NOAA-LCDv2-engineering-analysis/1.0",
    "Accept": "text/csv,text/plain,*/*",
})


def numeric(series: pd.Series) -> pd.Series:
    # LCD may append value flags. Keep the numeric measurement only.
    extracted = series.astype("string").str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def valid_csv_response(resp: requests.Response) -> bool:
    if resp.status_code != 200 or not resp.content:
        return False
    head = resp.content[:10000].decode("utf-8", errors="ignore")
    if "<html" in head.lower() or "<!doctype" in head.lower():
        return False
    return "DATE" in head and ("HourlyDryBulbTemperature" in head or "HourlyRelativeHumidity" in head)


def fetch_url(url: str, timeout=180):
    last = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=timeout, allow_redirects=True)
            last = (r.status_code, len(r.content), r.headers.get("content-type", ""))
            if valid_csv_response(r):
                return r.content, last
        except Exception as exc:
            last = ("EXCEPTION", 0, repr(exc))
        time.sleep(2 ** attempt)
    return None, last


def fetch_via_access_api(year: int):
    endpoint = "https://www.ncei.noaa.gov/access/services/data/v1"
    params = {
        "dataset": "local-climatological-data-v2",
        "stations": STATION_ID,
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31",
        "format": "csv",
        "includeAttributes": "false",
        "includeStationName": "true",
        "includeStationLocation": "true",
        "dataTypes": ",".join(REQUESTED + ["HourlyStationPressure"]),
    }
    last = None
    for attempt in range(3):
        try:
            r = SESSION.get(endpoint, params=params, timeout=240, allow_redirects=True)
            last = (r.status_code, len(r.content), r.headers.get("content-type", ""), r.url)
            if valid_csv_response(r):
                return r.content, last
        except Exception as exc:
            last = ("EXCEPTION", 0, repr(exc), endpoint)
        time.sleep(2 ** attempt)
    return None, last


def load_year(year: int):
    attempts = []
    for url in candidate_urls(year):
        content, meta = fetch_url(url)
        attempts.append({"url": url, "result": meta})
        if content is not None:
            return content, {"year": year, "source": url, "attempts": attempts}

    content, meta = fetch_via_access_api(year)
    attempts.append({"url": "NCEI Access Data Service", "result": meta})
    if content is not None:
        src = meta[3] if isinstance(meta, tuple) and len(meta) >= 4 else "NCEI Access Data Service"
        return content, {"year": year, "source": src, "attempts": attempts}

    raise RuntimeError(f"Unable to retrieve official NOAA LCDv2 data for {year}. Attempts: {attempts}")


def parse_lcd_timestamp(series: pd.Series) -> pd.Series:
    # LCDv2 documentation specifies DATE/TIME in Local Standard Time (no DST).
    # If an explicit offset appears, respect it. Otherwise localize to fixed CST (UTC-6), then convert to UTC.
    s = series.astype("string")
    explicit = s.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", regex=True, na=False)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if explicit.any():
        out.loc[explicit] = pd.to_datetime(s.loc[explicit], errors="coerce", utc=True, format="mixed")
    if (~explicit).any():
        naive = pd.to_datetime(s.loc[~explicit], errors="coerce", format="mixed")
        localized = naive.dt.tz_localize(LST_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        out.loc[~explicit] = localized
    return out


def read_noaa_csv(content: bytes, year: int) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), dtype=str, low_memory=False)
    if "DATE" not in df.columns:
        raise ValueError(f"NOAA {year} file lacks DATE column")
    for col in REQUESTED + ["HourlyStationPressure"]:
        if col not in df.columns:
            df[col] = np.nan
    z = pd.DataFrame(index=df.index)
    z["obs_ts"] = parse_lcd_timestamp(df["DATE"])
    for col in REQUESTED + ["HourlyStationPressure"]:
        z[col] = numeric(df[col])
    z["REPORT_TYPE"] = df["REPORT_TYPE"].astype("string") if "REPORT_TYPE" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    z["STATION"] = df["STATION"].astype("string") if "STATION" in df else STATION_ID
    z["NAME"] = df["NAME"].astype("string") if "NAME" in df else STATION_NAME
    z["source_year"] = year
    z = z.dropna(subset=["obs_ts"])
    # Keep hourly observation report families where report type is supplied. If absent, measurements themselves determine validity.
    if z["REPORT_TYPE"].notna().any():
        rpt = z["REPORT_TYPE"].fillna("").str.strip()
        hourly_mask = rpt.isin(["FM-12", "FM-15", "FM-16"]) | rpt.eq("")
        z = z.loc[hourly_mask]
    return z


def build_candidates(valid: pd.DataFrame) -> pd.DataFrame:
    # Each source observation can be a candidate for its floor and ceil clock hour if within +/-30 minutes.
    v = valid.reset_index(drop=True).copy()
    v["obs_id"] = np.arange(len(v), dtype=np.int64)
    floor_h = v["obs_ts"].dt.floor("h")
    ceil_h = v["obs_ts"].dt.ceil("h")
    a = v.copy(); a["target_hour"] = floor_h
    b = v.copy(); b["target_hour"] = ceil_h
    cand = pd.concat([a, b], ignore_index=True)
    cand = cand.drop_duplicates(["obs_id", "target_hour"])
    cand["delta_seconds"] = (cand["obs_ts"] - cand["target_hour"]).abs().dt.total_seconds()
    cand = cand[cand["delta_seconds"] <= 1800]
    cand = cand[cand["target_hour"].between(START_UTC, END_UTC)]
    cand["is_exact"] = cand["delta_seconds"].eq(0)
    cand["field_count"] = cand[REQUESTED].notna().sum(axis=1)
    return cand


def select_one_per_hour(valid: pd.DataFrame) -> pd.DataFrame:
    cand = build_candidates(valid)
    cand = cand.sort_values(
        ["target_hour", "is_exact", "delta_seconds", "field_count", "obs_ts"],
        ascending=[True, False, True, False, True],
        kind="mergesort",
    )
    used = set()
    chosen_rows = []
    # Greedy chronological selection prevents a :30 observation from being reused by adjacent hours.
    for target_hour, group in cand.groupby("target_hour", sort=True):
        picked = None
        for row in group.itertuples(index=False):
            oid = int(row.obs_id)
            if oid not in used:
                picked = row
                used.add(oid)
                break
        if picked is not None:
            chosen_rows.append(picked)
    if not chosen_rows:
        return pd.DataFrame()
    selected = pd.DataFrame(chosen_rows).set_index("target_hour").sort_index()
    return selected


def std_pressure_pa(elevation_m: float) -> float:
    return 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.2559


def main():
    print(f"Target UTC range: {START_UTC} -> {END_UTC}; rows={len(TARGET)}")
    assert len(TARGET) == 131520
    parts = []
    download_log = []
    for year in range(2011, 2027):
        content, log = load_year(year)
        download_log.append(log)
        part = read_noaa_csv(content, year)
        parts.append(part)
        print(year, log["source"], len(content), "bytes", len(part), "parsed rows")

    obs = pd.concat(parts, ignore_index=True).sort_values("obs_ts").reset_index(drop=True)
    # Broad source window around requested range, because +/-30 minute selection can touch boundaries.
    obs = obs[obs["obs_ts"].between(START_UTC - pd.Timedelta(minutes=30), END_UTC + pd.Timedelta(minutes=30))].copy()

    # NOAA value flags are already stripped by numeric(); impossible RH values are not eligible.
    valid = obs[
        obs["HourlyDryBulbTemperature"].notna()
        & obs["HourlyRelativeHumidity"].notna()
        & obs["HourlyRelativeHumidity"].between(0, 100)
    ].copy()

    selected = select_one_per_hour(valid)
    selected_cols = ["obs_ts"] + REQUESTED + ["HourlyStationPressure", "STATION", "NAME", "source_year"]
    if selected.empty:
        selected_small = pd.DataFrame(columns=selected_cols)
        selected_small.index = pd.DatetimeIndex([], tz="UTC", name="target_hour")
    else:
        selected_small = selected[selected_cols]

    out = pd.DataFrame(index=TARGET)
    out.index.name = "target_hour"
    out = out.join(selected_small, how="left")
    ok = out["obs_ts"].notna()

    psychrolib.SetUnitSystem(psychrolib.SI)
    pressure_pa = std_pressure_pa(ELEVATION_M)
    wb_c = np.full(len(out), np.nan, dtype=float)
    for i, (t_c, rh) in enumerate(zip(out["HourlyDryBulbTemperature"].to_numpy(), out["HourlyRelativeHumidity"].to_numpy())):
        if np.isfinite(t_c) and np.isfinite(rh) and 0.0 <= rh <= 100.0:
            wb_c[i] = psychrolib.GetTWetBulbFromRelHum(float(t_c), float(rh) / 100.0, pressure_pa)

    local = out.index.tz_convert(LOCAL_TZ)
    final = pd.DataFrame({
        "Date": local.strftime("%Y-%m-%d"),
        "Time": local.strftime("%H:%M:%S"),
        "Time_Zone": [x.tzname() for x in local],
        "Wet_Bulb_Temperature_F": wb_c * 1.8 + 32.0,
        "Dry_Bulb_Air_Temperature_F": out["HourlyDryBulbTemperature"].to_numpy(dtype=float) * 1.8 + 32.0,
        "Relative_Humidity_pct": out["HourlyRelativeHumidity"].to_numpy(dtype=float),
        "Dew_Point_Temperature_F": out["HourlyDewPointTemperature"].to_numpy(dtype=float) * 1.8 + 32.0,
        "Timestamp_UTC": out.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Observation_Timestamp_UTC": out["obs_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
        "Wet_Bulb_Method": np.where(ok, "CALCULATED_NOAA_T_RH_STD_PRESSURE", "MISSING"),
        "Data_Status": np.where(ok, "OK", "MISSING_NOAA_OBSERVATION"),
        "Source_Station_ID": STATION_ID,
        "Source_Station_Name": STATION_NAME,
    })
    for col in ["Wet_Bulb_Temperature_F", "Dry_Bulb_Air_Temperature_F", "Relative_Humidity_pct", "Dew_Point_Temperature_F"]:
        final[col] = pd.to_numeric(final[col], errors="coerce").round(2)

    # Direct NOAA wet-bulb is validation only, kept in C in source and converted here to F.
    direct_f = out["HourlyWetBulbTemperature"].to_numpy(dtype=float) * 1.8 + 32.0
    calc = final["Wet_Bulb_Temperature_F"]
    dry = final["Dry_Bulb_Air_Temperature_F"]
    dew = final["Dew_Point_Temperature_F"]
    cmp = pd.DataFrame({"calc": calc, "direct": direct_f}).dropna()
    err = cmp["calc"] - cmp["direct"]

    final.to_csv(OUTFILE, index=False, lineterminator="\n")
    check = pd.read_csv(OUTFILE, dtype=str)

    selected_obs = final.loc[ok, "Observation_Timestamp_UTC"]
    qa = {
        "source": "NOAA/NCEI Local Climatological Data Version 2 (LCDv2)",
        "station_id": STATION_ID,
        "station_name": STATION_NAME,
        "site_address": SITE_ADDRESS,
        "target_start_local": str(START_LOCAL),
        "target_end_local": str(END_LOCAL),
        "target_start_utc": START_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_end_utc": END_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_target_rows": int(len(final)),
        "populated_rows": int(ok.sum()),
        "missing_rows": int((~ok).sum()),
        "coverage_pct": round(float(ok.mean() * 100.0), 4),
        "duplicate_target_timestamps": int(final["Timestamp_UTC"].duplicated().sum()),
        "missing_target_timestamps": int(final["Timestamp_UTC"].isna().sum()),
        "source_observation_reuse": int(selected_obs[selected_obs.notna()].duplicated().sum()),
        "first_populated_target_utc": final.loc[ok, "Timestamp_UTC"].iloc[0] if ok.any() else None,
        "last_populated_target_utc": final.loc[ok, "Timestamp_UTC"].iloc[-1] if ok.any() else None,
        "last_selected_noaa_observation_utc": selected_obs.dropna().iloc[-1] if ok.any() else None,
        "last_available_noaa_observation_utc_in_downloads": obs["obs_ts"].max().strftime("%Y-%m-%dT%H:%M:%SZ") if len(obs) else None,
        "rh_outside_0_100": int(((final["Relative_Humidity_pct"] < 0) | (final["Relative_Humidity_pct"] > 100)).sum()),
        "dewpoint_gt_drybulb": int((dew > dry + 0.1).sum()),
        "wetbulb_gt_drybulb": int((calc > dry + 0.1).sum()),
        "wetbulb_lt_dewpoint": int((calc < dew - 0.1).sum()),
        "validation_n": int(len(cmp)),
        "validation_mean_bias_f": round(float(err.mean()), 4) if len(err) else None,
        "validation_mae_f": round(float(err.abs().mean()), 4) if len(err) else None,
        "validation_rmse_f": round(float(np.sqrt((err ** 2).mean())), 4) if len(err) else None,
        "validation_p95_abs_error_f": round(float(err.abs().quantile(0.95)), 4) if len(err) else None,
        "validation_max_abs_error_f": round(float(err.abs().max()), 4) if len(err) else None,
        "standard_pressure_pa": round(float(pressure_pa), 2),
        "csv_header_rows": 1,
        "csv_data_rows": int(len(check)),
        "csv_exact_columns_order": list(check.columns) == list(final.columns),
        "csv_filename": str(OUTFILE),
        "download_log": download_log,
    }
    QAFILE.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qa.items() if k != "download_log"}, indent=2))

    # Hard QA gates for structure and physical consistency. Missing observations are allowed and explicitly flagged.
    assert len(final) == 131520
    assert len(check) == 131520
    assert qa["duplicate_target_timestamps"] == 0
    assert qa["missing_target_timestamps"] == 0
    assert qa["source_observation_reuse"] == 0
    assert qa["rh_outside_0_100"] == 0
    assert qa["dewpoint_gt_drybulb"] == 0
    assert qa["wetbulb_gt_drybulb"] == 0
    assert qa["wetbulb_lt_dewpoint"] == 0
    assert qa["csv_exact_columns_order"] is True

    # Create text-safe transfer chunks so the CSV can be reconstructed outside GitHub Actions.
    raw = OUTFILE.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9)
    b64 = base64.b64encode(compressed).decode("ascii")
    TRANSFER_DIR.mkdir(exist_ok=True)
    chunk_size = 180_000
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]
    for old in TRANSFER_DIR.glob("part_*.txt"):
        old.unlink()
    for i, chunk in enumerate(chunks):
        (TRANSFER_DIR / f"part_{i:03d}.txt").write_text(chunk, encoding="ascii")
    manifest = {
        "encoding": "base64(gzip(csv))",
        "csv_filename": OUTFILE.name,
        "original_bytes": len(raw),
        "gzip_bytes": len(compressed),
        "base64_chars": len(b64),
        "chunk_size": chunk_size,
        "part_count": len(chunks),
        "parts": [f"part_{i:03d}.txt" for i in range(len(chunks))],
    }
    (TRANSFER_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("TRANSFER_MANIFEST", json.dumps(manifest))


if __name__ == "__main__":
    main()
