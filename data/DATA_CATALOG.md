# Data Catalog: Statistical Rethinking 2026

Cleaned teaching datasets for Bayesian statistics and causal inference.
All data derived from Slovenian public sources (GURS, SURS, ERAR, Police).

Prepared: 2026-04-23
Script: `prepare_datasets.py`

---

## Datasets

### 1. sr2026_real_estate.csv (26 MB, 191,097 rows)

**Source:** GURS Evidenca trga nepremičnin (ETN), 2007-2025
**Granularity:** Individual property transaction

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | int | GURS transaction ID |
| date | date | Contract signing date |
| year | int | Transaction year |
| quarter | int | Transaction quarter (1-4) |
| municipality | str | Municipality name (občina) |
| cadastral_municipality | str | Cadastral municipality (katastrska občina) |
| settlement | str | Settlement name |
| property_type_code | int | GURS building part type (1-15) |
| property_type | str | English label: apartment, detached_house, garage, office, retail, etc. |
| area_m2 | float | Sold area in m² |
| usable_area_m2 | float | Usable area in m² (when available) |
| price_eur | float | Transaction price in EUR |
| price_per_m2 | float | Price per m² (price_eur / area_m2) |
| year_built | int | Year of construction (1500-2026; invalid values set to NaN) |
| is_new_construction | bool | New construction flag |
| floor | int | Floor number (when applicable) |
| rooms | int | Number of rooms (1-20; 0 and >20 set to NaN) |
| total_area_m2 | float | Total building part area |
| easting | float | Gauss-Kruger easting (D48/GK) |
| northing | float | Gauss-Kruger northing (D48/GK) |
| market_status | int | TRZNOST_POSLA: 1=market, 2=flagged, 4=unclassified |

**Suggested DAGs:**
- `area_m2 → price_eur ← municipality` (confounding by location)
- `year_built → price_eur ← floor` (apartment desirability)
- `municipality → price_per_m2 ← property_type` (interaction effects)

**Lecture mapping:**
- A03 (linear regression): `area_m2 → price_eur` for apartments
- A04 (categories): `property_type` as index variable
- A05 (estimands): total vs. direct effect of municipality on price
- A06 (confounds): `municipality` as fork between school proximity and price
- A07 (controls): `year_built` as potential collider

---

### 2. sr2026_municipality_panel.csv (36 KB, 424 rows)

**Source:** SURS, GURS ETN (aggregated), 2023-2024
**Granularity:** Municipality × year (212 municipalities × 2 years)

| Column | Type | Description |
|--------|------|-------------|
| municipality_name | str | Municipality name (212 unique) |
| year | int | Year |
| avg_gross_salary | float | Average monthly gross salary (EUR) |
| avg_net_salary | float | Average monthly net salary (EUR) |
| permits_per_1000 | float | Residential building permits per 1,000 population (SURS rate) |
| median_price_m2 | float | Median residential price/m² (apartments + houses, from ETN) |
| mean_price_m2 | float | Mean residential price/m² |
| transaction_count | int | Number of residential transactions |
| median_area_m2 | float | Median sold area (m²) |
| hpi_total | float | National housing price index (base 2015 = 100) |

**Suggested DAGs:**
- `avg_gross_salary → median_price_m2 ← permits_per_1000` (supply-demand)
- `avg_gross_salary → permits_per_1000 → median_price_m2` (pipe/mediator)
- `transaction_count → median_price_m2 ← hpi_total` (national trend vs. local activity)

**Lecture mapping:**
- A03: salary → price regression across municipalities
- A05: heterogeneous effects (salary-price relationship varies by transaction volume)
- A06: salary as fork confounding permits-price relationship

---

### 3. sr2026_crime.csv (20 KB, 915 rows)

**Source:** Slovenian Police, 2010-2024
**Granularity:** Administrative unit (policijska uprava) × year

| Column | Type | Description |
|--------|------|-------------|
| admin_unit | str | Police administrative unit name |
| crime_count | int | Number of reported criminal offenses |
| year | int | Year |

**Suggested DAGs:**
- Pair with municipality panel for `income → crime_rate` analysis
- Fork: `urbanization → crime_count` and `urbanization → income`
- Time series: COVID lockdown effect on crime (DiD with 2020 treatment)

**Lecture mapping:**
- A01-A02: Bayesian estimation of crime rate (binomial/Poisson)
- A06: urbanization as fork confounding income-crime association

---

### 4. sr2026_forensic_audio.csv (32 KB, 200 rows)

**Source:** Synthetic, calibrated from ENF analysis pipeline
**Granularity:** Individual audio recording

| Column | Type | Description |
|--------|------|-------------|
| recording_id | int | Unique recording identifier |
| category | str | authentic, tts_generated, spliced, enf_injected_naive, enf_injected_sophisticated |
| is_authentic | int | Binary: 1 = authentic, 0 = manipulated |
| device | str | Recording device type (7 categories) |
| speaker_id | int | Speaker identity (1-20) |
| duration_s | float | Recording duration (seconds) |
| sample_rate | int | Audio sample rate (Hz) |
| bandwidth_hz | float | Effective frequency bandwidth |
| enf_snr_50hz | float | ENF signal-to-noise ratio at 50 Hz |
| enf_snr_100hz | float | ENF SNR at 100 Hz (2nd harmonic) |
| enf_snr_150hz | float | ENF SNR at 150 Hz (3rd harmonic) |
| pause_cv | float | Coefficient of variation of speech pauses |
| pause_count | int | Number of detected speech pauses |
| pause_mean_s | float | Mean pause duration (seconds) |
| noise_floor_mean_db | float | Mean noise floor level (dBFS) |
| noise_floor_std_db | float | Noise floor variability |
| noise_step_changes | int | Number of noise level discontinuities |
| splice_candidates | int | Number of energy discontinuities (splice evidence) |
| phase_jump_count | int | Number of ENF phase jumps |
| spectral_centroid_mean_hz | float | Mean spectral centroid frequency |
| spectral_centroid_std_hz | float | Spectral centroid variability |
| effective_bits | int | Effective quantization bit depth |
| kl_divergence | float | KL divergence (re-encoding evidence) |
| channel_correlation | float | Stereo channel correlation (NaN if mono) |
| is_stereo | int | 1 = stereo recording |
| enf_bandwidth_hz_narrow | float | ENF peak bandwidth (injection indicator) |
| harmonic_ratio_100 | float | 100 Hz / 50 Hz power ratio |
| harmonic_ratio_150 | float | 150 Hz / 50 Hz power ratio |
| enf_noise_correlation | float | ENF-broadband noise envelope correlation |
| phase_jitter | float | ENF phase velocity standard deviation |

**Category distribution:** ~80 authentic, 50 TTS, 30 spliced, 20 naive-injected, 20 sophisticated-injected

**Suggested DAGs:**
- `device → bandwidth_hz → enf_snr_50hz` (pipe: device determines bandwidth which affects ENF detection)
- `category → pause_cv ← speaker_id` (fork: speaker style affects pauses independently of manipulation)
- `is_authentic → splice_candidates ← duration_s` (longer recordings have more candidates even if authentic)
- `category → enf_snr_50hz ← device` (collider: selection into "suspicious" recordings)

**Lecture mapping:**
- A01-A02: Bayesian updating of authenticity belief given sequential test results
- A03: logistic regression on `is_authentic ~ pause_cv + enf_snr_50hz`
- A04: device type as index variable
- A05: total vs. direct effect of device on authenticity score
- A06: device as fork confounding bandwidth-authenticity association
- A07: training database as collider (only submitted recordings observed)

**Unique pedagogical value:** No other teaching dataset connects Bayesian statistics to forensic science. The 9-test framework maps directly to likelihood ratios and Bayes factors.

---

### 5. sr2026_economic_pulse.csv (8 KB, 149 rows)

**Source:** SURS, Eurostat, UMAR, Fiskalni svet (1995-2026)
**Granularity:** National (Slovenia), mixed frequency (monthly/quarterly/annual)

| Column | Type | Description |
|--------|------|-------------|
| indicator | str | Indicator key (13 unique) |
| date | date | Period date |
| period_type | str | monthly, quarterly, annual |
| value | float | Indicator value |
| yoy_change_pct | float | Year-over-year change (%) |
| signal | str | Traffic light: green, yellow, red |
| is_forecast | bool | True if forecast (UMAR/Fiskalni svet) |

**Indicators:** gdp_growth, inflation_hicp, unemployment_rate, avg_gross_wage, real_wage_growth, wage_public_sector, wage_private_sector, unit_labor_cost, consumer_confidence, trade_balance, realestate_price_index, gov_debt_gdp, budget_balance

**Suggested DAGs:**
- `gdp_growth → unemployment_rate ← inflation_hicp` (Phillips curve)
- `avg_gross_wage → inflation_hicp → consumer_confidence` (pipe)
- `gdp_growth → trade_balance ← realestate_price_index` (wealth effect)

**Lecture mapping:**
- A01-A02: sequential updating of GDP forecast with quarterly data
- A03: wage-inflation regression
- A06: GDP as fork confounding wage-employment association

---

### 6. sr2026_governance.csv (640 KB, 4,644 rows)

**Source:** ERAR (Komisija za preprečevanje korupcije RS)
**Granularity:** Individual post-employment restriction record

| Column | Type | Description |
|--------|------|-------------|
| record_id | int | ERAR record ID |
| institution_name | str | Public institution (298 unique) |
| institution_id | str | Institution registration number |
| subject_name | str | Restricted business entity |
| subject_id | str | Entity registration number |
| restriction_type | str | Type of restriction (poslovnega subjekta, etc.) |
| start_date | date | Restriction start date |
| end_date | date | Restriction end date (mostly null = indefinite) |
| prohibited_transactions | int | Count of prohibited transactions |
| start_year | int | Start year |
| duration_days | float | Duration in days (when end_date available) |

**Suggested DAGs:**
- `institution_type → restriction_count ← sector` (fork)
- `restriction_count → prohibited_transactions` (direct effect)
- Selection bias: only institutions above a threshold get monitored (collider)

**Lecture mapping:**
- A01-A02: Poisson model for prohibited transaction counts
- A06: institution type as fork confounding restriction-violation association
- A07: monitoring intensity as collider

---

## Coordinate Reference System

ETN coordinates (easting, northing) use the Slovenian national grid:
- **CRS:** D48/GK (EPSG:3912) — Gauss-Kruger projection
- Convert to WGS84 (lat/lon) using `pyproj`:
  ```python
  from pyproj import Transformer
  transformer = Transformer.from_crs("EPSG:3912", "EPSG:4326", always_xy=True)
  lon, lat = transformer.transform(easting, northing)
  ```

---

## Quick Start

```python
import pandas as pd

# Load datasets
re = pd.read_csv("sr2026_real_estate.csv", parse_dates=["date"])
mun = pd.read_csv("sr2026_municipality_panel.csv")
forensic = pd.read_csv("sr2026_forensic_audio.csv")
econ = pd.read_csv("sr2026_economic_pulse.csv", parse_dates=["date"])
gov = pd.read_csv("sr2026_governance.csv", parse_dates=["start_date", "end_date"])
crime = pd.read_csv("sr2026_crime.csv")

# Example: Ljubljana apartments
lj_apt = re[(re["municipality"] == "LJUBLJANA") & (re["property_type"] == "apartment")]
print(f"Ljubljana apartments: {len(lj_apt):,} transactions")
print(f"Median price/m²: €{lj_apt['price_per_m2'].median():,.0f}")
```

---

## Regeneration

```bash
python prepare_datasets.py --all           # All datasets
python prepare_datasets.py --etn           # Only real estate (slow, reads 277 MB)
python prepare_datasets.py --forensic      # Only forensic audio (fast, synthetic)
python prepare_datasets.py --municipalities # Municipality panel
python prepare_datasets.py --economic      # Economic indicators
python prepare_datasets.py --governance    # ERAR restrictions
```

---

## Data Quality Notes

**Filters applied during preparation:**
- Real estate: year >= 2007, area <= 5,000 m², price <= €5M, price/m² in [10, 15000]
- Real estate: `year_built` outside 1500-2026 set to NaN; `rooms` outside 1-20 set to NaN
- Municipality panel: `permits_per_1000` is a SURS rate (per 1,000 population), not a raw count
- Economic pulse: deduplicated on indicator + date
- Forensic audio: bandwidth clamped to Nyquist frequency (sample_rate / 2)

**Known limitations:**
- Municipality panel covers only 2023-2024 (income data availability)
- Crime data uses administrative units (UE), not municipalities; mapping is approximate
- ETN `market_status = 2` (flagged) transactions included; consider filtering for stricter analysis
- Some ETN municipalities have very few transactions; median prices may be unstable
- Forensic audio is synthetic; distributions calibrated from pipeline but not from large real-world corpus

---

## License

Source data: GURS, SURS, ERAR, Police (CC BY 4.0 Slovenian public data).
Forensic audio: Synthetic (no license restriction).
Preparation script: MIT.
