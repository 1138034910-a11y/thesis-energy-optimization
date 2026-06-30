# Scale-dependent hydrogen-battery substitution in large-scale renewable bases under carbon pricing: A two-stage stochastic optimization framework

**Target journal:** *Journal of Energy Storage* (JES)  
**Article type:** Full Length Article  
**First author:** Haoshuang Cheng  
**Corresponding author:** LeiMing Li (19920005@upc.edu.cn)  
**Affiliation:** School of Economics and Management, China University of Petroleum (East China), Qingdao 266580, China

**Repository:** https://github.com/1138034910-a11y/thesis-energy-optimization

---

## Repository Overview

This repository contains the data and code for reproducing the paper:

> **Scale-dependent hydrogen-battery substitution in large-scale renewable bases under carbon pricing: A two-stage stochastic optimization framework**

The workflow covers probabilistic forecasting (KAN), Gaussian Copula scenario generation, representative-day clustering, two-stage stochastic MILP optimization, and post-hoc Storage Substitution Elasticity (SSE) analysis. This repository includes the input datasets and all model code so that the reported results can be reproduced. Result files and manuscript documents are **not** included in this repository.

---

## Directory Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── config.py                          # Project configuration
├── DATA_NOTICE.md                     # Data availability statement
├── GITHUB_SETUP.md                    # Instructions for creating/pushing the repo
├── create_and_push_repo.ps1           # PowerShell helper to create the GitHub repo
├── references.bib                     # Bibliography (BibTeX)
│
├── data/                              # Input datasets (wind/solar profiles and KAN prediction results)
│   ├── 数据.xlsx
│   ├── 风电数据.xlsx
│   ├── 光电数据.xlsx
│   ├── 甘肃_风电_prediction_result.csv
│   ├── 甘肃_光伏_prediction_result.csv
│   └── README.md
│
├── src/                               # Core model implementation
│   ├── stochastic_model.py
│   ├── deterministic_model.py
│   ├── kan_predictor.py
│   ├── scenario_generator.py
│   ├── representative_days.py
│   ├── sensitivity_analysis.py
│   ├── baseline_predictors_FIXED.py
│   └── analysis_FIXED.py
│
├── experiments/                       # Experiment scripts
│   ├── run_full_experiment_v3.py
│   ├── run_h2_sensitivity_v3_rigorous_vss.py
│   ├── run_carbon_price_sensitivity_final.py
│   ├── run_copula_sensitivity_v3.py
│   ├── run_scenario_count_tractability.py
│   ├── run_mipstart_robustness_400t.py
│   ├── run_sse_cost_sensitivity.py
│   └── ...
│
└── scripts/                           # Figure generation and post-processing
    ├── generate_all_paper_figures_unified.py
    ├── generate_si_figures.py
    ├── nature_style.py
    ├── fill_si_tables_from_experiments.py
    ├── compute_sse_confidence.py
    ├── plot_encroachment_evidence.py
    ├── plot_mac_crossover.py
    ├── plot_storage_energy_cost.py
    └── ...
```

---

## Quick Start

1. Install dependencies (preferably in a virtual environment):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Confirm that the input data files listed in `data/README.md` are present.

3. Run the main experiments:

   ```powershell
   python experiments\run_h2_sensitivity_v3_rigorous_vss.py
   python experiments\run_carbon_price_sensitivity_final.py
   ```

4. Generate figures and tables locally:

   ```powershell
   python scripts\generate_all_paper_figures_unified.py
   python scripts\fill_si_tables_from_experiments.py
   ```

---

## Data and Code Availability

- The input datasets (provincial wind/solar generation profiles and Gansu KAN prediction results) are included in the `data/` directory.
- The source paper PDF and supplementary reference documents are excluded; see Wang et al. (2023), *Nature Communications* 14, 5379.
- The optimization model, scenario-generation scripts, KAN forecasting code, and solver configuration files are publicly available at: https://github.com/1138034910-a11y/thesis-energy-optimization

---

## Citation

If you use this code or data, please cite the manuscript once published.
