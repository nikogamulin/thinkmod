"""Prepare cleaned teaching datasets for Statistical Rethinking 2026.

Reads raw sources from GURS ETN, SURS, Causaris pipelines, police statistics,
economic pulse, ERAR, and forensic audio analysis code. Produces self-contained
CSV files ready for Bayesian modeling exercises.

Usage:
    python prepare_datasets.py [--all | --etn | --municipalities | --forensic | --economic | --crime]

Outputs (in ./):
    sr2026_real_estate.csv        — ETN property transactions (2007-2025)
    sr2026_municipality_panel.csv — Municipality-year panel (demographics, income, permits, crime)
    sr2026_forensic_audio.csv     — Synthetic forensic audio features (9-test framework)
    sr2026_economic_pulse.csv     — Slovenian macroeconomic indicators (1995-2026)
    sr2026_governance.csv         — ERAR post-employment restrictions by institution

Author: Niko Gamulin, PhD
License: MIT
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ETN_ROOT = Path("/home/niko/workspace/auctor/slovenia_elections/data/etn")
CAUSARIS_DATA = Path("/home/niko/workspace/causaris/data")
POLICE_DIR = CAUSARIS_DATA / "police"
ECONOMIC_DIR = CAUSARIS_DATA / "economic_pulse"
ERAR_DIR = CAUSARIS_DATA / "erar"
RE_EXPORTS = CAUSARIS_DATA / "real_estate" / "sqlite_exports"
ELECTION_EXPORTS = CAUSARIS_DATA / "elections" / "sqlite_exports"

OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. ETN Real Estate Transactions
# ---------------------------------------------------------------------------

# Property type mapping (VRSTA_DELA_STAVBE code → English label)
PROPERTY_TYPE_MAP = {
    1: "detached_house",
    2: "apartment",
    3: "parking",
    4: "garage",
    5: "office",
    6: "retail",
    7: "hospitality",
    8: "industrial",
    9: "other_commercial",
    10: "other_commercial",
    11: "other_commercial",
    12: "other_commercial",
    13: "agricultural",
    14: "auxiliary",
    15: "other",
}

# TRZNOST_POSLA: 1 = market, 2 = market (flagged), 4 = unclassified (pre-2015)
MARKET_TRZNOST = {1, 2, 4}


def prepare_etn(
    years: range = range(2007, 2026),
    sample_frac: float | None = None,
) -> pd.DataFrame:
    """Load and clean ETN purchase transactions across multiple years.

    Args:
        years: Range of years to load.
        sample_frac: If set, randomly sample this fraction per year (for testing).

    Returns:
        DataFrame with cleaned transaction records.
    """
    log.info("Preparing ETN real estate dataset for years %d-%d", years.start, years.stop - 1)
    frames: list[pd.DataFrame] = []

    for year in years:
        year_dir = ETN_ROOT / str(year)
        if not year_dir.exists():
            log.warning("Missing ETN directory for %d, skipping", year)
            continue

        # Find POSLI and DELISTAVB files (naming varies by download date)
        posli_files = list(year_dir.glob("*POSLI*.csv"))
        deli_files = list(year_dir.glob("*DELISTAVB*.csv"))

        if not posli_files or not deli_files:
            log.warning("Missing POSLI or DELISTAVB for %d, skipping", year)
            continue

        posli_path = posli_files[0]
        deli_path = deli_files[0]

        try:
            posli = pd.read_csv(posli_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            posli = pd.read_csv(posli_path, encoding="cp1250", low_memory=False)

        try:
            deli = pd.read_csv(deli_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            deli = pd.read_csv(deli_path, encoding="cp1250", low_memory=False)

        # Filter market transactions
        if "TRZNOST_POSLA" in posli.columns:
            posli = posli[posli["TRZNOST_POSLA"].isin(MARKET_TRZNOST)]

        # Join on ID_POSLA
        posli_cols = ["ID_POSLA", "DATUM_SKLENITVE_POGODBE", "POGODBENA_CENA_ODSKODNINA", "TRZNOST_POSLA"]
        posli_cols = [c for c in posli_cols if c in posli.columns]
        merged = deli.merge(posli[posli_cols], on="ID_POSLA", how="inner")

        # Use PRODANA_POVRSINA as area; fall back to POVRSINA_DELA_STAVBE
        merged["_area"] = pd.to_numeric(merged.get("PRODANA_POVRSINA"), errors="coerce")
        if "POVRSINA_DELA_STAVBE" in merged.columns:
            fallback = pd.to_numeric(merged["POVRSINA_DELA_STAVBE"], errors="coerce")
            merged["_area"] = merged["_area"].fillna(fallback)

        # Use per-part price if available; fall back to transaction-level price
        # for single-part transactions
        merged["_price"] = pd.to_numeric(merged.get("POGODBENA_CENA_DELA_STAVBE"), errors="coerce")
        posli_price = pd.to_numeric(merged.get("POGODBENA_CENA_ODSKODNINA"), errors="coerce")
        parts_per_tx = merged.groupby("ID_POSLA")["ID_POSLA"].transform("count")
        single_part = parts_per_tx == 1
        merged.loc[merged["_price"].isna() & single_part, "_price"] = posli_price[merged["_price"].isna() & single_part]

        # Keep rows with valid area and price
        merged = merged.dropna(subset=["_area", "_price"])
        merged = merged[merged["_area"] > 0]
        merged = merged[merged["_price"] > 0]

        if sample_frac is not None:
            merged = merged.sample(frac=sample_frac, random_state=42)

        frames.append(merged)
        log.info("  %d: %d transactions loaded", year, len(merged))

    if not frames:
        log.error("No ETN data loaded")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Build clean output
    out = pd.DataFrame()
    out["transaction_id"] = df["ID_POSLA"]
    out["date"] = pd.to_datetime(df["DATUM_SKLENITVE_POGODBE"], format="%d.%m.%Y", errors="coerce")
    out["year"] = out["date"].dt.year
    out["quarter"] = out["date"].dt.quarter
    out["municipality"] = df["OBCINA"].str.strip()
    out["cadastral_municipality"] = df["IME_KO"].str.strip()
    out["settlement"] = df["NASELJE"].str.strip() if "NASELJE" in df.columns else np.nan

    # Property type
    out["property_type_code"] = pd.to_numeric(df["VRSTA_DELA_STAVBE"], errors="coerce")
    out["property_type"] = out["property_type_code"].map(PROPERTY_TYPE_MAP).fillna("other")

    out["area_m2"] = df["_area"]
    out["usable_area_m2"] = pd.to_numeric(
        df.get("UPORABNA_POVRSINA", pd.Series(dtype=float)),
        errors="coerce",
    )
    out["price_eur"] = df["_price"]
    out["price_per_m2"] = (out["price_eur"] / out["area_m2"]).round(2)
    # Year built: clamp to valid range, NaN otherwise
    year_built_raw = pd.to_numeric(df.get("LETO_IZGRADNJE_DELA_STAVBE"), errors="coerce")
    out["year_built"] = year_built_raw.where(
        (year_built_raw >= 1500) & (year_built_raw <= 2026)
    )

    out["is_new_construction"] = df.get("NOVOGRADNJA", "").astype(str).str.strip().isin(["1", "DA"])
    out["floor"] = pd.to_numeric(df.get("NADSTROPJE_DELA_STAVBE"), errors="coerce")

    # Rooms: NaN out impossible values (> 20 rooms is almost certainly data error)
    rooms_raw = pd.to_numeric(df.get("STEVILO_SOB"), errors="coerce")
    out["rooms"] = rooms_raw.where((rooms_raw > 0) & (rooms_raw <= 20))

    out["total_area_m2"] = pd.to_numeric(df.get("POVRSINA_DELA_STAVBE"), errors="coerce")

    # Coordinates (Gauss-Kruger D48/GK → keep as-is, note in docs)
    out["easting"] = pd.to_numeric(df.get("E_CENTROID"), errors="coerce")
    out["northing"] = pd.to_numeric(df.get("N_CENTROID"), errors="coerce")

    out["market_status"] = pd.to_numeric(df.get("TRZNOST_POSLA"), errors="coerce")

    # Drop rows with missing essential fields
    out = out.dropna(subset=["date", "area_m2", "price_eur"])

    # Filter: keep only ETN-era transactions (2007+), reasonable area (< 5000 m²),
    # reasonable price (< €5M), and price_per_m2 in [10, 15000]
    out = out[out["year"] >= 2007]
    out = out[out["area_m2"] <= 5000]
    out = out[out["price_eur"] <= 5_000_000]
    out = out[(out["price_per_m2"] >= 10) & (out["price_per_m2"] <= 15_000)]

    out = out.sort_values(["date", "municipality"]).reset_index(drop=True)
    log.info("ETN dataset: %d rows, %d columns", len(out), len(out.columns))
    return out


# ---------------------------------------------------------------------------
# 2. Municipality Panel
# ---------------------------------------------------------------------------

def prepare_municipality_panel() -> pd.DataFrame:
    """Build a municipality × year panel from demographics, income, permits, and crime.

    Returns:
        DataFrame with one row per municipality per year.
    """
    log.info("Preparing municipality panel dataset")

    # --- Demographics ---
    # The exported demographics.csv has empty population columns.
    # Instead, build demographics from the ETN real estate data (municipality list)
    # and available SURS data if present.
    demo = pd.DataFrame()
    log.info("  Demographics: using municipality names from income/permits data")

    # --- Income ---
    income_path = RE_EXPORTS / "municipality_income.csv"
    if income_path.exists():
        income = pd.read_csv(income_path)
        income = income[["municipality_code", "municipality_name", "year",
                         "avg_gross_salary_monthly_eur", "avg_net_salary_monthly_eur"]]
        income = income.rename(columns={
            "avg_gross_salary_monthly_eur": "avg_gross_salary",
            "avg_net_salary_monthly_eur": "avg_net_salary",
        })
        income = income.dropna(subset=["avg_gross_salary"])
        log.info("  Income: %d rows", len(income))
    else:
        log.warning("  Income file not found")
        income = pd.DataFrame()

    # --- Construction permits ---
    # SURS table 1970721S reports permits as per-1000-population rates, not counts.
    permits_path = RE_EXPORTS / "construction_permits.csv"
    if permits_path.exists():
        permits = pd.read_csv(permits_path)
        # Values are per-1000-population rates; average across quarters/types per year
        permits_agg = (
            permits.groupby(["municipality_name", "year"])
            .agg(
                permits_per_1000=("permits_issued", "mean"),
            )
            .reset_index()
        )
        permits_agg["permits_per_1000"] = permits_agg["permits_per_1000"].round(2)
        permits_agg = permits_agg.dropna(subset=["municipality_name"])
        log.info("  Permits: %d municipality-year rows", len(permits_agg))
    else:
        log.warning("  Permits file not found")
        permits_agg = pd.DataFrame()

    # --- Crime statistics ---
    crime_frames: list[pd.DataFrame] = []
    for crime_file in sorted(POLICE_DIR.glob("kd20*.csv")):
        year_str = crime_file.stem.replace("kd", "")
        try:
            year = int(year_str)
        except ValueError:
            continue
        try:
            cdf = pd.read_csv(crime_file, sep=";", encoding="cp1250", low_memory=False)
        except Exception:
            try:
                cdf = pd.read_csv(crime_file, sep=";", encoding="utf-8", low_memory=False)
            except Exception as e:
                log.warning("  Could not read %s: %s", crime_file.name, e)
                continue

        # Count offenses by administrative unit (UE)
        if "UpravnaEnotaStoritve" in cdf.columns:
            ue_col = "UpravnaEnotaStoritve"
        elif "PUStoritveKD" in cdf.columns:
            ue_col = "PUStoritveKD"
        else:
            log.warning("  No location column in %s", crime_file.name)
            continue

        counts = cdf.groupby(ue_col).size().reset_index(name="crime_count")
        counts["year"] = year
        counts = counts.rename(columns={ue_col: "admin_unit"})
        crime_frames.append(counts)

    if crime_frames:
        crime = pd.concat(crime_frames, ignore_index=True)
        # Clean admin unit names
        crime["admin_unit"] = crime["admin_unit"].str.strip()
        log.info("  Crime: %d admin-unit-year rows", len(crime))
    else:
        crime = pd.DataFrame()

    # --- Build panel by merging available data ---
    # Start with income as the base (has municipality_name)
    if not income.empty:
        panel = income.copy()
    else:
        log.error("No base data for municipality panel")
        return pd.DataFrame()

    # Drop the always-empty municipality_code column
    if "municipality_code" in panel.columns:
        panel = panel.drop(columns=["municipality_code"])

    # Merge permits on municipality_name + year
    if not permits_agg.empty and not panel.empty:
        panel = panel.merge(
            permits_agg[["municipality_name", "year", "permits_per_1000"]],
            on=["municipality_name", "year"],
            how="left",
        )

    # Compute per-municipality real estate stats from ETN
    etn_path = OUTPUT_DIR / "sr2026_real_estate.csv"
    if etn_path.exists():
        etn = pd.read_csv(etn_path, usecols=["municipality", "year", "price_per_m2", "area_m2", "property_type"])
        # Filter to residential (apartments + houses)
        residential = etn[etn["property_type"].isin(["apartment", "detached_house"])]
        re_stats = (
            residential.groupby(["municipality", "year"])
            .agg(
                median_price_m2=("price_per_m2", "median"),
                mean_price_m2=("price_per_m2", "mean"),
                transaction_count=("price_per_m2", "count"),
                median_area_m2=("area_m2", "median"),
            )
            .reset_index()
        )
        re_stats["median_price_m2"] = re_stats["median_price_m2"].round(2)
        re_stats["mean_price_m2"] = re_stats["mean_price_m2"].round(2)

        # Normalize municipality names for matching
        panel["_mun_lower"] = panel["municipality_name"].str.lower().str.strip()
        re_stats["_mun_lower"] = re_stats["municipality"].str.lower().str.strip()
        panel = panel.merge(
            re_stats[["_mun_lower", "year", "median_price_m2", "mean_price_m2",
                       "transaction_count", "median_area_m2"]],
            on=["_mun_lower", "year"],
            how="left",
        )
        panel = panel.drop(columns=["_mun_lower"])
        log.info("  Real estate stats: merged per-municipality price data")

    # Housing price indices (national level — attach to all municipalities)
    hpi_path = RE_EXPORTS / "housing_price_indices.csv"
    if hpi_path.exists():
        hpi = pd.read_csv(hpi_path)
        hpi_annual = (
            hpi[hpi["index_type"] == "hpi_total"]
            .groupby("year")
            .agg(hpi_total=("index_value", "mean"))
            .reset_index()
        )
        panel = panel.merge(hpi_annual, on="year", how="left")
        log.info("  HPI: merged national housing price index")

    panel = panel.sort_values(["municipality_name", "year"] if "municipality_name" in panel.columns else ["year"])
    panel = panel.reset_index(drop=True)

    # Attach crime as a separate table (different granularity: admin unit, not municipality)
    # Save crime separately as sr2026_crime.csv
    if not crime.empty:
        crime_path = OUTPUT_DIR / "sr2026_crime.csv"
        crime.to_csv(crime_path, index=False)
        log.info("  Crime data saved separately: %d rows → %s", len(crime), crime_path.name)

    log.info("Municipality panel: %d rows, %d columns", len(panel), len(panel.columns))
    return panel


# ---------------------------------------------------------------------------
# 3. Forensic Audio (Synthetic Teaching Dataset)
# ---------------------------------------------------------------------------

def prepare_forensic_audio(n_recordings: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic forensic audio feature dataset based on the 9-test framework.

    Simulates realistic distributions for authentic, TTS-generated, spliced, and
    ENF-injected recordings. Feature ranges are calibrated from the actual
    forensic pipeline in /home/niko/niko-os/projects/forensic-intelligence/.

    The dataset supports:
        - Bayesian classification (is_authentic as outcome)
        - DAG-based causal reasoning (device → features, injection → ENF patterns)
        - Hierarchical models (recordings nested within devices/speakers)
        - Diagnostic calibration (sensitivity/specificity of each test)

    Args:
        n_recordings: Number of synthetic recordings to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with forensic audio features.
    """
    log.info("Generating synthetic forensic audio dataset (n=%d)", n_recordings)
    rng = np.random.default_rng(seed)

    # Recording categories and their proportions
    categories = ["authentic", "tts_generated", "spliced", "enf_injected_naive", "enf_injected_sophisticated"]
    weights = [0.40, 0.25, 0.15, 0.10, 0.10]
    n_per_cat = rng.multinomial(n_recordings, weights)

    rows: list[dict] = []
    recording_id = 0

    devices = ["smartphone_android", "smartphone_ios", "laptop_mic", "usb_mic",
               "covert_gsm", "landline", "voip"]
    device_bandwidth_map = {
        "smartphone_android": (8000, 16000),
        "smartphone_ios": (10000, 20000),
        "laptop_mic": (6000, 16000),
        "usb_mic": (12000, 22000),
        "covert_gsm": (2000, 4000),
        "landline": (300, 3400),
        "voip": (4000, 8000),
    }

    for cat_idx, (category, n_cat) in enumerate(zip(categories, n_per_cat)):
        for i in range(n_cat):
            recording_id += 1
            is_authentic = category == "authentic"
            device = rng.choice(devices) if is_authentic else rng.choice(["voip", "laptop_mic", "usb_mic"])

            bw_lo, bw_hi = device_bandwidth_map[device]
            duration_s = rng.uniform(15, 600)
            # Sample rate must be consistent with device bandwidth (Nyquist)
            min_sr_for_device = int(bw_hi * 2.5)  # need headroom above bandwidth
            valid_rates = [sr for sr in [8000, 16000, 44100, 48000] if sr >= min_sr_for_device]
            if not valid_rates:
                valid_rates = [48000]
            sample_rate = rng.choice(valid_rates)
            nyquist = sample_rate / 2

            # Test 1: Bandwidth (clamped to Nyquist)
            bandwidth_hz = min(rng.uniform(bw_lo * 0.8, bw_hi * 1.1), nyquist * 0.95)
            if category == "tts_generated":
                bandwidth_hz = min(rng.uniform(4000, 8000), nyquist * 0.95)

            # Test 2: ENF presence (SNR at 50 Hz)
            if is_authentic and device in ("covert_gsm", "landline", "smartphone_android"):
                enf_snr_50hz = rng.uniform(2.0, 15.0)  # ENF present in mains-coupled devices
            elif category.startswith("enf_injected"):
                enf_snr_50hz = rng.uniform(5.0, 30.0)  # Injected ENF is strong
            else:
                enf_snr_50hz = rng.uniform(0.5, 2.0)  # Weak or absent

            enf_snr_100hz = enf_snr_50hz * rng.uniform(0.3, 0.8) if is_authentic else enf_snr_50hz * rng.uniform(0.5, 0.7)
            enf_snr_150hz = enf_snr_50hz * rng.uniform(0.1, 0.5) if is_authentic else enf_snr_50hz * rng.uniform(0.2, 0.4)

            # Test 3: Pause distribution (CV of speech pauses)
            if is_authentic:
                pause_cv = rng.uniform(0.3, 0.9)  # Natural variability
                pause_count = int(rng.uniform(5, 50) * (duration_s / 60))
            elif category == "tts_generated":
                pause_cv = rng.uniform(0.05, 0.25)  # Too regular
                pause_count = int(rng.uniform(8, 30) * (duration_s / 60))
            else:
                pause_cv = rng.uniform(0.25, 0.85)
                pause_count = int(rng.uniform(5, 40) * (duration_s / 60))

            pause_mean_s = rng.uniform(0.15, 0.8)

            # Test 4: Noise floor consistency
            if is_authentic:
                noise_floor_mean_db = rng.uniform(-55, -35)
                noise_floor_std_db = rng.uniform(0.5, 3.0)
                noise_step_changes = int(rng.choice([0, 0, 0, 1], p=[0.7, 0.1, 0.1, 0.1]))
            elif category == "spliced":
                noise_floor_mean_db = rng.uniform(-50, -30)
                noise_floor_std_db = rng.uniform(2.0, 8.0)  # Inconsistent
                noise_step_changes = int(rng.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.2]))
            else:
                noise_floor_mean_db = rng.uniform(-60, -40)
                noise_floor_std_db = rng.uniform(0.2, 1.5)  # Too consistent (TTS/synthetic)
                noise_step_changes = 0

            # Test 5: Splice detection (energy discontinuities)
            if category == "spliced":
                splice_candidates = int(rng.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1]))
            elif is_authentic:
                splice_candidates = int(rng.choice([0, 0, 1], p=[0.7, 0.2, 0.1]))
            else:
                splice_candidates = int(rng.choice([0, 1], p=[0.8, 0.2]))

            # Test 6: Phase coherence (ENF phase jumps)
            if category == "spliced":
                phase_jump_count = int(rng.uniform(1, 6))
            elif is_authentic:
                phase_jump_count = int(rng.choice([0, 0, 1], p=[0.8, 0.1, 0.1]))
            else:
                phase_jump_count = 0

            # Test 7: Spectral centroid stability
            if is_authentic:
                spectral_centroid_mean_hz = rng.uniform(800, 2500)
                spectral_centroid_std_hz = rng.uniform(100, 400)
            elif category == "tts_generated":
                spectral_centroid_mean_hz = rng.uniform(1000, 2000)
                spectral_centroid_std_hz = rng.uniform(50, 150)  # Too stable
            else:
                spectral_centroid_mean_hz = rng.uniform(800, 2500)
                spectral_centroid_std_hz = rng.uniform(80, 350)

            # Test 8: Quantization
            if is_authentic:
                effective_bits = rng.choice([16, 16, 16, 24], p=[0.6, 0.2, 0.1, 0.1])
                kl_divergence = rng.uniform(0.001, 0.05)
            elif category == "tts_generated":
                effective_bits = rng.choice([16, 32], p=[0.5, 0.5])
                kl_divergence = rng.uniform(0.0001, 0.01)  # Very clean quantization
            else:
                effective_bits = rng.choice([8, 16], p=[0.3, 0.7])
                kl_divergence = rng.uniform(0.01, 0.15)  # Re-encoding artifacts

            # Test 9: Channel correlation
            is_stereo = rng.random() < 0.4
            if is_stereo:
                if is_authentic:
                    channel_correlation = rng.uniform(0.7, 0.95)
                elif category == "tts_generated":
                    channel_correlation = 1.0  # Duplicated mono
                else:
                    channel_correlation = rng.uniform(0.6, 0.99)
            else:
                channel_correlation = np.nan  # Mono recording

            # ENF injection detection features
            if category == "enf_injected_naive":
                enf_bandwidth_hz = rng.uniform(0.01, 0.04)  # Unnaturally narrow
                harmonic_ratio_100 = rng.uniform(0.0, 0.05)  # No harmonics
                harmonic_ratio_150 = rng.uniform(0.0, 0.03)
                enf_noise_correlation = rng.uniform(-0.01, 0.01)  # Uncorrelated
                phase_jitter = rng.uniform(0.0001, 0.001)  # Too smooth
            elif category == "enf_injected_sophisticated":
                enf_bandwidth_hz = rng.uniform(0.03, 0.08)
                harmonic_ratio_100 = rng.uniform(0.3, 0.7)  # Suspiciously uniform
                harmonic_ratio_150 = rng.uniform(0.1, 0.4)
                enf_noise_correlation = rng.uniform(-0.02, 0.02)
                phase_jitter = rng.uniform(0.001, 0.005)
            elif is_authentic and enf_snr_50hz > 2.0:
                enf_bandwidth_hz = rng.uniform(0.08, 0.5)
                harmonic_ratio_100 = rng.uniform(0.05, 0.9)  # Variable
                harmonic_ratio_150 = rng.uniform(0.01, 0.6)
                enf_noise_correlation = rng.uniform(0.05, 0.4)  # Correlated with noise
                phase_jitter = rng.uniform(0.005, 0.02)
            else:
                enf_bandwidth_hz = np.nan
                harmonic_ratio_100 = np.nan
                harmonic_ratio_150 = np.nan
                enf_noise_correlation = np.nan
                phase_jitter = np.nan

            # Speaker identity (latent variable for hierarchical modeling)
            speaker_id = rng.integers(1, 21)  # 20 speakers

            rows.append({
                "recording_id": recording_id,
                "category": category,
                "is_authentic": int(is_authentic),
                "device": device,
                "speaker_id": speaker_id,
                "duration_s": round(duration_s, 1),
                "sample_rate": int(sample_rate),
                # Test features
                "bandwidth_hz": round(bandwidth_hz, 1),
                "enf_snr_50hz": round(enf_snr_50hz, 3),
                "enf_snr_100hz": round(enf_snr_100hz, 3),
                "enf_snr_150hz": round(enf_snr_150hz, 3),
                "pause_cv": round(pause_cv, 4),
                "pause_count": pause_count,
                "pause_mean_s": round(pause_mean_s, 4),
                "noise_floor_mean_db": round(noise_floor_mean_db, 2),
                "noise_floor_std_db": round(noise_floor_std_db, 3),
                "noise_step_changes": noise_step_changes,
                "splice_candidates": splice_candidates,
                "phase_jump_count": phase_jump_count,
                "spectral_centroid_mean_hz": round(spectral_centroid_mean_hz, 1),
                "spectral_centroid_std_hz": round(spectral_centroid_std_hz, 1),
                "effective_bits": int(effective_bits),
                "kl_divergence": round(kl_divergence, 6),
                "channel_correlation": round(channel_correlation, 4) if not np.isnan(channel_correlation) else np.nan,
                "is_stereo": int(is_stereo),
                # ENF injection features
                "enf_bandwidth_hz_narrow": round(enf_bandwidth_hz, 6) if not np.isnan(enf_bandwidth_hz) else np.nan,
                "harmonic_ratio_100": round(harmonic_ratio_100, 6) if not np.isnan(harmonic_ratio_100) else np.nan,
                "harmonic_ratio_150": round(harmonic_ratio_150, 6) if not np.isnan(harmonic_ratio_150) else np.nan,
                "enf_noise_correlation": round(enf_noise_correlation, 6) if not np.isnan(enf_noise_correlation) else np.nan,
                "phase_jitter": round(phase_jitter, 6) if not np.isnan(phase_jitter) else np.nan,
            })

    df = pd.DataFrame(rows)
    log.info("Forensic audio dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# 4. Economic Pulse
# ---------------------------------------------------------------------------

def prepare_economic_pulse() -> pd.DataFrame:
    """Load and clean Slovenian macroeconomic indicators.

    Returns:
        DataFrame with indicator time series.
    """
    log.info("Preparing economic pulse dataset")

    # Try the final import file first
    import_path = ECONOMIC_DIR / "import" / "economic_pulse_data_final.csv"
    if not import_path.exists():
        import_path = ECONOMIC_DIR / "import" / "economic_pulse_data_20260220.csv"
    if not import_path.exists():
        log.error("Economic pulse data not found")
        return pd.DataFrame()

    df = pd.read_csv(import_path)

    out = pd.DataFrame()
    out["indicator"] = df["indicator_key"]
    out["date"] = pd.to_datetime(df["period_date"], errors="coerce")
    out["period_type"] = df["period_type"]
    out["value"] = pd.to_numeric(df["value"], errors="coerce")
    out["yoy_change_pct"] = pd.to_numeric(df["yoy_change"], errors="coerce")
    out["signal"] = df["signal"]
    out["is_forecast"] = df.get("is_forecast", False)

    out = out.dropna(subset=["date", "value"])

    # Remove duplicates (keep first)
    out = out.drop_duplicates(subset=["indicator", "date"], keep="first")

    out = out.sort_values(["indicator", "date"]).reset_index(drop=True)
    log.info("Economic pulse: %d rows, %d indicators", len(out), out["indicator"].nunique())
    return out


# ---------------------------------------------------------------------------
# 5. Governance (ERAR)
# ---------------------------------------------------------------------------

def prepare_governance() -> pd.DataFrame:
    """Load ERAR post-employment restrictions and build a governance dataset.

    Returns:
        DataFrame with restriction records.
    """
    log.info("Preparing governance dataset from ERAR")

    erar_path = ERAR_DIR / "omejitve_all.json"
    if not erar_path.exists():
        log.error("ERAR data not found at %s", erar_path)
        return pd.DataFrame()

    with open(erar_path, encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []
    for record in data:
        org = record.get("organizacija") or {}
        subj = record.get("subjekt") or {}
        rows.append({
            "record_id": record.get("id"),
            "institution_name": org.get("ime", ""),
            "institution_id": org.get("maticna_stevilka", ""),
            "subject_name": subj.get("ime", ""),
            "subject_id": subj.get("maticna_stevilka", ""),
            "restriction_type": record.get("omejitev_do", ""),
            "start_date": record.get("trajanje_od"),
            "end_date": record.get("trajanje_do"),
            "prohibited_transactions": record.get("st_nedovoljenih_transakcij", 0),
        })

    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["start_year"] = df["start_date"].dt.year
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days

    df = df.sort_values("start_date").reset_index(drop=True)
    log.info("Governance dataset: %d records, %d institutions",
             len(df), df["institution_name"].nunique())
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SR2026 teaching datasets")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets")
    parser.add_argument("--etn", action="store_true", help="ETN real estate transactions")
    parser.add_argument("--municipalities", action="store_true", help="Municipality panel")
    parser.add_argument("--forensic", action="store_true", help="Forensic audio features")
    parser.add_argument("--economic", action="store_true", help="Economic pulse indicators")
    parser.add_argument("--governance", action="store_true", help="ERAR governance data")
    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.all, args.etn, args.municipalities, args.forensic, args.economic, args.governance]):
        args.all = True

    if args.all or args.etn:
        etn = prepare_etn()
        if not etn.empty:
            path = OUTPUT_DIR / "sr2026_real_estate.csv"
            etn.to_csv(path, index=False)
            log.info("Saved %s (%d rows)", path.name, len(etn))

    if args.all or args.municipalities:
        panel = prepare_municipality_panel()
        if not panel.empty:
            path = OUTPUT_DIR / "sr2026_municipality_panel.csv"
            panel.to_csv(path, index=False)
            log.info("Saved %s (%d rows)", path.name, len(panel))

    if args.all or args.forensic:
        forensic = prepare_forensic_audio(n_recordings=200)
        if not forensic.empty:
            path = OUTPUT_DIR / "sr2026_forensic_audio.csv"
            forensic.to_csv(path, index=False)
            log.info("Saved %s (%d rows)", path.name, len(forensic))

    if args.all or args.economic:
        econ = prepare_economic_pulse()
        if not econ.empty:
            path = OUTPUT_DIR / "sr2026_economic_pulse.csv"
            econ.to_csv(path, index=False)
            log.info("Saved %s (%d rows)", path.name, len(econ))

    if args.all or args.governance:
        gov = prepare_governance()
        if not gov.empty:
            path = OUTPUT_DIR / "sr2026_governance.csv"
            gov.to_csv(path, index=False)
            log.info("Saved %s (%d rows)", path.name, len(gov))

    log.info("Done. All datasets saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
