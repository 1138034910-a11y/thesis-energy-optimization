# Scale-dependent hydrogen-battery substitution in large renewable bases

This repository contains the code and configuration files that support the manuscript:

> **Scale-dependent hydrogen-battery substitution in large renewable bases: A two-stage stochastic optimization approach under carbon pricing**  
> Target journal: *Energy Conversion and Management* (ECM)

## What is included

- `config.py` — Economic, physical, solver, and scenario-generation parameters used in the case study.
- `src/` — Core Python modules:
  - `kan_predictor.py` — Kolmogorov–Arnold network for probabilistic wind/solar forecasting.
  - `scenario_generator.py` — Gaussian Copula scenario generation and k-means++ reduction.
  - `representative_days.py` — Time-aggregation of the 8,760-hour horizon.
  - `stochastic_model.py` — Two-stage stochastic MILP (Gurobi).
  - `deterministic_model.py` — Expected-value deterministic benchmark.
  - `sensitivity_analysis.py`, `analysis_FIXED.py`, `baseline_predictors_FIXED.py` — Supporting analysis and benchmarking.
- `experiments/` — Scripts used to produce the sensitivity results reported in the paper:
  - `run_h2_sensitivity_v3_rigorous_vss.py`
  - `run_carbon_price_sensitivity_final.py`
  - `run_full_experiment_v3.py`
  - `run_mipstart_robustness_400t.py`
  - `run_no_copula_validation.py`
  - `run_seed_robustness_final.py`
  - `run_endogenous_h2_capacity.py`
  - `ablation_copula_independent_400t.py`
  - `plot_h2_substitution_journal.py`
- `scripts/` — Post-processing and figure-generation scripts:
  - `generate_all_paper_figures_unified.py`
  - `generate_si_figures.py`
  - `compute_sse_confidence.py`
  - `cross_validate_data.py`
  - `nature_style.py` (shared plotting style)

## Data availability

**The raw hourly wind and solar capacity-factor data for Gansu Province are not included in this repository.** They are available upon reasonable request from the corresponding author of this paper (see the manuscript's CRediT and corresponding-author statement).

To reproduce the optimization results, place the requested data files in the `data/` directory and ensure their paths match those defined in `config.py` (`DataPaths`). A detailed description of the required input format is provided in `data/README.md`.

## Dependencies

Key Python packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

A standalone Gurobi license (or Gurobi WLS) is required to solve the MILP.

## Reproducing the main results

1. Obtain the input data as described in `data/README.md` and place it in `data/`.
2. Configure parameters in `config.py` if needed.
3. Run the full base-case experiment:
   ```bash
   python experiments/run_full_experiment_v3.py
   ```
4. Run the hydrogen-scale sensitivity:
   ```bash
   python experiments/run_h2_sensitivity_v3_rigorous_vss.py
   ```
5. Run the carbon-price sensitivity:
   ```bash
   python experiments/run_carbon_price_sensitivity_final.py
   ```

## Citation

If you use this code, please cite the manuscript once published.

## License

This repository is released under the MIT License (see `LICENSE`).
