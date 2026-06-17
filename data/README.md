# Data directory

This directory is intentionally empty in the public repository.

## How to obtain the data

The hourly wind and solar capacity-factor data for Gansu Province used in this study are available upon reasonable request from the corresponding author of the manuscript. Please contact the corresponding author listed in the manuscript's CRediT authorship statement.

## Required input files

Once obtained, the following files should be placed in this directory:

- `甘肃_风电_prediction_result.csv` — Hourly wind prediction results (actual and predicted capacity factors).
- `甘肃_光伏_prediction_result.csv` — Hourly solar (PV) prediction results (actual and predicted capacity factors).

## Expected CSV format

Each file should contain the columns:

| Column | Description | Unit |
|--------|-------------|------|
| `hour` | Hour of the year (1–8760) | — |
| `actual_pu` | Observed capacity factor | per unit |
| `predicted_pu` | Predicted capacity factor | per unit |

## After adding data

Update `config.py` (`DataPaths`) if the filenames differ, then run the experiment scripts from the repository root.
