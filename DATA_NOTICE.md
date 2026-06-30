# Data Notice

The following input datasets are included in the `data/` directory of this repository:

- `数据.xlsx` — Hourly wind and solar generation profiles by province (8,760 h).
- `风电数据.xlsx` — Gansu hourly wind generation profile.
- `光电数据.xlsx` — Gansu hourly solar (PV) generation profile.
- `甘肃_风电_prediction_result.csv` — Gansu hourly wind prediction results (actual and predicted capacity factors).
- `甘肃_光伏_prediction_result.csv` — Gansu hourly solar (PV) prediction results (actual and predicted capacity factors).

The provincial generation profiles trace to the open dataset accompanying Wang et al. (2023), *Nature Communications* 14, 5379 (https://doi.org/10.1038/s41467-023-40670-7). The Gansu prediction result CSVs were generated with the KAN forecasting pipeline provided in this repository.

Researchers who wish to use the code with alternative input data should ensure the filenames and sheet names match the expectations in `config.py` and the figure-generation scripts.
