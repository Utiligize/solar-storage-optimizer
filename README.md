# PV + Storage Is NOT All You Need

**A Scandinavian Challenge to Casey Handmer's Solar+Storage Thesis**

This repository replicates and extends Casey Handmer's solar+battery optimization
model, testing his claim that "PV + Storage is All You Need" against Scandinavian
conditions and real-world grid economics.

Full PDF report: [`output/solar_storage_scandinavia_report.pdf`](output/solar_storage_scandinavia_report.pdf)

---

## Background

On August 6, 2025, Casey Handmer ([@CJHandmer](https://twitter.com/CJHandmer)) published
a deep analysis arguing that rapidly falling solar and battery costs make off-grid
power systems viable for virtually any load, anywhere -- dismissing grid backup as
unnecessary. He shared his [data and Mathematica code](https://drive.google.com/drive/folders/1Jsv34JjNo22NOCd26iBHMP80EPG0xmQE)
on August 7.

His key claims:
- Solar at EUR 200k/MW and batteries at EUR 200k/MWh make off-grid viable
- Massive solar overbuild is cheaper than large batteries for cloudy days
- Geography barely matters (factor of 2 cost variation across 6 locations)
- Grid connections are unnecessary ("DEI for turbines")

We test these claims at 57N latitude (Denmark) where solar yield is 35% lower than
Texas, and compare against actual grid connection costs and DK1 spot electricity prices.

---

## How This Analysis Differs from Handmer's

### 1. Real-world costs vs CapEx-only

**Handmer's model** considers only upfront capital expenditure: solar panels and batteries.
Once purchased, the system runs for free forever. This is the single largest omission
in his analysis and systematically biases results in favour of off-grid.

**Our model** adds the costs that any real project must bear:

| Cost component | Handmer | This analysis |
|---|---|---|
| Solar CapEx | EUR 200k/MW | EUR 200k/MW (same) |
| Battery CapEx | EUR 200k/MWh | EUR 200k/MWh (same) |
| Solar O&M | Not included | EUR 10k/MW/year |
| Battery O&M | Not included | EUR 5k/MWh/year |
| Land lease | Not included | EUR 7.5k/MW/year (~1 ha/MW) |
| Battery replacement | Not included | At year 14, at 50% of original cost |
| Solar degradation | Not included | 0.5%/year (lifetime avg ~94% of year 1) |
| Grid electricity cost | Not applicable | NPV of 25 years of spot + tariffs |
| Discount rate | None (CapEx only) | 5% for all recurring costs |

For a 10 MW solar array with 12 MWh battery over 25 years, these "missing" costs add
approximately EUR 4-5M to the off-grid system cost -- roughly doubling the power system
cost relative to Handmer's CapEx-only model.

### 2. Grid-connected alternative

Handmer does not model a grid-connected alternative at all. His framework only optimizes
solar array size and battery size for a standalone off-grid system. The implicit assumption
is that the grid either does not exist or is too expensive to consider.

We add a grid-connected scenario using:
- **Grid connection fee**: 2,000,000 DKK/MW (~EUR 268k/MW), one-time
- **Electricity**: Historical DK1 spot prices (2023, mean ~87 EUR/MWh)
- **Grid tariffs**: 20 EUR/MWh (60kV DSO) or 16 EUR/MWh (TSO), based on 2026
  Energinet rates. Includes system tariff, network tariff, DSO capacity charge,
  and electricity tax (business rate). PSO was abolished in 2022.
- **100% utilization**: The grid is always available

This allows a direct comparison: what does 1 MWh of delivered energy actually cost
from off-grid solar+storage vs from the grid?

### 3. Grid tariff research

Handmer's analysis has no grid costs because he models no grid scenario. We researched
actual grid tariffs for large industrial loads (10-100+ MW) across all regions in his
analysis:

| Region | Grid tariff (EUR/MWh) | Key components |
|---|---|---|
| Denmark (60kV DSO) | ~20 | Energinet system 9.7 + network 5.8 + DSO 4 + tax 0.5 |
| Denmark (TSO, >80MW) | ~16 | No DSO fee |
| Texas (ERCOT) | ~9 | 4CP transmission ~8 + admin 0.5 |
| Arizona | ~20 | APS bundled delivery |
| California | ~75 | PG&E/SCE non-generation surcharges |
| Maine | ~40 | ISO-NE transmission + delivery |
| Washington | ~15 | BPA hydro territory |
| Britain | ~115 | TNUoS + BSUoS + RO/CfD/FiT + CM + CCL |

Britain stands out with grid tariffs roughly 10x those of Texas. This actually
strengthens Handmer's case for the UK, where the grid is so expensive that off-grid
solar may be competitive despite mediocre solar resources.

### 4. Seasonal resolution at high latitude

Handmer tested 6 locations (32-52N latitude range) and found "minimal variation."
Our Denmark site at 57N exposes the limits of this claim:

- **Annual yield**: Denmark 1,100 kWh/kWp vs Texas 1,700 kWh/kWp (35% less)
- **Winter crisis**: At 57N, December days are 6-7 hours with very low sun angle.
  Multi-day cloud cover is routine. The darkest weeks produce <5% of summer output.
- **Seasonal storage problem**: Batteries sized for overnight storage (12-16h) cannot
  bridge multi-day or multi-week winter deficits. You would need weeks of storage,
  which is economically absurd with lithium-ion at EUR 200k/MWh.

### 5. Utilization as a reliability metric

Handmer optimizes "cost per unit utilization" -- total system cost divided by the
fraction of the year the load runs. This metric implicitly treats partial utilization
as acceptable. For an electric kettle or water pump, it is. For a data center where
each percentage point of downtime on a EUR 50M/MW facility represents ~EUR 500k/year
in stranded capital, it is not.

Our analysis tracks utilization explicitly and shows that off-grid systems in Denmark
plateau at 75-90% utilization for data center loads -- far below the 99.9%+ required
for critical infrastructure.

### 6. Solar data sources

Handmer uses NREL's Solar Power Data for Integration Studies: measured/simulated
5-minute resolution data from ~6,000 PV plants across Texas (2006 weather year).
For his other locations (Arizona, Britain, California, Maine, Washington), he
published only the pre-computed optimization results, not the raw solar profiles.

We generate synthetic 5-minute solar profiles using **pvlib** (open-source solar
modelling library) with:
- Ineichen clear-sky model
- Monthly clearness indices calibrated to known annual yields
- AR(1) cloud variability process with beta-distributed cloud factors

Our synthetic Texas profile (CF 0.192, 1,685 kWh/kWp) validates well against
Handmer's NREL data (CF 0.191, 1,673 kWh/kWp).

---

## Regional Conclusions: Where Does Off-grid Win?

### Off-grid solar+storage is competitive (grid may not be needed)

**Texas, Arizona, Washington** -- These sun-rich regions with low grid tariffs
(EUR 9-20/MWh) and high solar yields (1,500-1,900 kWh/kWp) are where Handmer's
thesis is strongest. Off-grid solar+storage costs are low, utilization can reach
95%+, and the grid's cost advantage is slim. For loads that can tolerate occasional
curtailment, off-grid is viable.

**California** -- Despite excellent solar resources (~1,800 kWh/kWp), California's
extraordinarily high grid tariffs (~75 EUR/MWh in non-generation charges) make
off-grid solar economically attractive even for loads requiring high utilization.
California is perhaps the strongest case for Handmer's thesis in a developed economy.

**Britain** -- Counterintuitively, Britain's terrible grid economics (115 EUR/MWh in
non-commodity charges) make off-grid solar competitive despite mediocre solar resources
(~1,000 kWh/kWp). The grid is so expensive that even inefficient off-grid systems may
be cheaper. However, utilization will be low (similar seasonal problems to Denmark),
making this viable only for loads that tolerate intermittency.

### Grid connection wins (off-grid does not make economic sense)

**Denmark / Scandinavia** -- With moderate grid tariffs (~16-20 EUR/MWh), reasonable
spot prices, and poor winter solar resources, the grid wins for high-CapEx continuous
loads above approximately EUR 7M/MW. This includes essentially all data center
applications. The crossover point is clearly visible in our optimization results.

**Maine / New England** -- Despite relatively high grid tariffs (~40 EUR/MWh), the
solar resource at northern latitudes (similar seasonal issues to Scandinavia) and
high reliability requirements favour grid connection for continuous loads.

### The grid connection time caveat

Even where grid connection is the economically optimal solution, **connection timelines
can be a critical blocker**. In Denmark and across Europe, grid connection for large
loads (50-100+ MW) routinely takes 3-7 years due to:

- **Grid capacity constraints**: Transmission and distribution networks in many regions
  are at or near capacity, requiring reinforcement before new large loads can connect
- **Permitting and environmental review**: New substations and transmission lines face
  lengthy planning processes
- **Queue backlogs**: In many markets (notably ERCOT, PJM, and European TSOs), the
  interconnection queue has grown to years of backlog as data centre and renewable
  energy projects compete for limited grid capacity
- **Equipment lead times**: High-voltage transformers and switchgear have 18-36 month
  delivery times

This means that even if the grid is cheaper in steady-state, a data centre developer
who needs power in 12-18 months may have no choice but to deploy solar+storage as a
bridge solution -- or as the primary power source if the grid connection never
materialises.

Handmer's thesis gains significant practical weight in this context: not because
off-grid is cheaper, but because it is **faster to deploy**. A solar+battery system
can be installed in 6-12 months. In a market where time-to-power determines whether
a project happens at all, the grid's theoretical cost advantage is irrelevant if
it comes with a 5-year wait.

---

## Repository Structure

```
generate_solar_data.py    # Synthetic solar profiles via pvlib (Denmark, Texas)
optimize.py               # Off-grid, grid-connected, and hybrid optimization
generate_report.py        # PDF report with plots
data/                     # Solar profiles, spot prices, optimization results
data/handmer/             # Casey Handmer's original data and results
output/                   # PDF report and plot images
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib pvlib fpdf2

python generate_solar_data.py   # Generate 5-min solar profiles
python optimize.py              # Run all optimizations
python generate_report.py       # Generate PDF report
```

## Data Sources

- **Solar profiles**: pvlib synthetic (Denmark), NREL Solar Integration Study (Texas)
- **DK1 spot prices**: Energi Data Service API (2023 historical)
- **Grid tariffs**: Energinet (2026 rates), ERCOT, NESO, EIA state data
- **Handmer's data**: [Google Drive](https://drive.google.com/drive/folders/1Jsv34JjNo22NOCd26iBHMP80EPG0xmQE)

## License

This analysis is provided for educational and research purposes. Handmer's original
data is shared under his terms. Our code is MIT licensed.
