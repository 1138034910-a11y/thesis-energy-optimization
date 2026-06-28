# Scale-dependent hydrogen-battery substitution in large renewable bases: A two-stage stochastic optimization approach under carbon pricing

**Target journal:** *Energy Conversion and Management* (ECM)  
**Article type:** Full Length Article  
**First author:** Haoshuang Cheng  
**Corresponding author:** LeiMing Li (19920005@upc.edu.cn)  
**Affiliation:** School of Economics and Management, China University of Petroleum (East China), Qingdao 266580, China

---

## Repository Overview

This repository contains the complete workflow for the paper:

> **Scale-dependent hydrogen-battery substitution in large renewable bases: A two-stage stochastic optimization approach under carbon pricing**

The workflow covers probabilistic forecasting (KAN), Gaussian Copula scenario generation, representative-day clustering, two-stage stochastic MILP optimization, and post-hoc substitution-elasticity analysis.

---

## Directory Structure

```
.
├── AGENTS.md                          # Project memory and decisions
├── README.md                          # This file
├── config.py                          # Project configuration
├── gurobi.lic                         # Gurobi license
│
├── main_ecm.tex                       # Compiled LaTeX manuscript source
├── main_ecm.pdf                       # Compiled PDF (main text + SI)
├── main_ecm_bibliography.tex          # Generated bibliography
├── main_ecm_supplementary.tex         # Generated supplementary material
├── main_ecm.{aux,out,log,spl}         # LaTeX compilation artifacts
│
├── manuscript_complete_v2.md          # Authoritative merged manuscript (Markdown)
├── manuscript_complete_v2_ecm_work.md # Working version of the manuscript
│
├── paper/                             # Manuscript section sources
│   ├── section_00_abstract_and_frontmatter.md
│   ├── section_01_introduction.md
│   ├── ...
│   ├── section_08_supplementary_material.md
│   ├── cover_letter.md
│   ├── DOI_VALIDATION_REPORT.md
│   ├── manuscript_complete_v2.md
│   ├── manuscript_complete_v2.docx
│   └── references.bib
│
├── src/                               # Core model implementation
│   ├── stochastic_model.py
│   ├── deterministic_model.py
│   ├── kan_predictor.py
│   ├── scenario_generator.py
│   ├── representative_days.py
│   └── sensitivity_analysis.py
│
├── experiments/                       # Experiment scripts
│   ├── run_full_experiment_v3.py
│   ├── run_h2_sensitivity_v3_rigorous_vss.py
│   ├── run_carbon_price_sensitivity_final.py
│   ├── run_endogenous_h2_capacity.py
│   └── ...
│
├── scripts/                           # Figure generation and audit scripts
│   ├── generate_all_paper_figures_unified.py
│   ├── nature_style.py
│   ├── merge_manuscript.py
│   ├── verify_dois.py
│   └── ...
│
├── data/                              # Raw input data
├── results/                           # Figures, tables, and solver logs
│   ├── figures_paper/                 # Current paper figures
│   ├── figures_journal/               # Journal-format figures
│   ├── tables/                        # CSV/JSON result tables
│   ├── logs/                          # Solver logs
│   └── archive/                       # Archived intermediate tables
│
├── references/                        # Reference PDFs
├── archive/                           # Old files, backups, and superseded scripts
├── submission_package/                # Files prepared for journal submission
└── .venv/                             # Python virtual environment
```

---

## LaTeX Compilation Pipeline

From the project root, run:

```powershell
pandoc manuscript_complete_v2_ecm_work.md -o body_raw2.tex `
  --from=markdown+tex_math_dollars+tex_math_single_backslash
python clean_body.py
python fix_math_blocks.py
python convert_citations_to_cite.py
python generate_bibliography_tex.py
python assemble_main_ecm.py
python fix_math_envs.py
python fix_final_issues.py
python clean_si.py
pdflatex main_ecm.tex
pdflatex main_ecm.tex
```

`pandoc` is at `D:\Anaconda3\Library\bin\pandoc.exe` and `pdflatex` is from TeX Live 2026.

---

## DOCX Export

```powershell
pandoc manuscript_complete_v2.md -o paper\manuscript_complete_v2.docx `
  --from=markdown+tex_math_dollars --resource-path=paper;results/figures_paper
```

Close Word before running to avoid file-lock errors.

---

## Data and Code Availability

- The hourly wind and solar capacity factor data for Gansu Province are available upon reasonable request from the corresponding author of [20].
- The optimization model, scenario-generation scripts, and solver configuration files are publicly available in a GitHub repository (URL to be added before submission).

---

## Key Manuscript Metrics

- Main-text prose: ~7,900 words (limit 9,000)
- Main figures: 8
- Main tables: 5
- Total pages: 50 (main text + references + supplementary)
- References: 53 (all cited in main text)
