# Data directory

This directory contains the input datasets used by the model code and scripts.

## Included files

| File | Description | Format |
|------|-------------|--------|
| `数据.xlsx` | Hourly wind and solar generation profiles by province (8,760 h), including `Wind_generation` and `Solar_generation` sheets. | Excel |
| `风电数据.xlsx` | Gansu hourly wind generation profile. | Excel |
| `光电数据.xlsx` | Gansu hourly solar (PV) generation profile. | Excel |
| `甘肃_风电_prediction_result.csv` | Hourly wind prediction results for Gansu (actual and predicted capacity factors). | CSV |
| `甘肃_光伏_prediction_result.csv` | Hourly solar (PV) prediction results for Gansu (actual and predicted capacity factors). | CSV |

## Source

The provincial wind/solar generation profiles trace to the open dataset accompanying Wang et al. (2023), *Nature Communications* 14, 5379 (https://doi.org/10.1038/s41467-023-40670-7). The Gansu prediction result CSVs were generated with the KAN forecasting pipeline in this repository.

## Usage

Place all files in this directory (`github_submission_package/data/`). The file paths and sheet names are referenced in `config.py` and in the figure-generation scripts. If you rename any file, update the corresponding paths in `config.py`.
