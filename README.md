# NYC Mayor 2025 — Data Integration & Feature Matrix Scaffold

This is a tiny, battery‑included scaffold to **finish data integration** and **build the feature matrices** for the two‑compartment model (EV vs Remaining/E‑Day). It mirrors our agreed steps.

## Layout
```
src/
  utils.py              # IO, keys, transforms (logit, zscore), validation helpers
  build_base.py         # Merge all sources to `out/model_base_by_ed.parquet`
  build_turnout.py      # Build scenario weights to `out/turnout_scenarios.parquet`
  build_features.py     # Emit `out/features_ev_by_ed.parquet` & `out/features_r_by_ed.parquet`
  validate.py           # CI‑style checks on joins, coverage, distributions
sample/
  config.yml            # Paths + tunables you will customize
scripts/
  run_all.sh            # One‑shot pipeline runner
Makefile                # `make all` runs the core steps
data/                   # Drop your CSVs/Parquets here
out/                    # Pipeline outputs land here
```

## Quick start
1. Copy your inputs into `data/` and update `sample/config.yml` (or copy to `config.yml` at repo root).
2. Run either:
   ```bash
   make all
   # or
   bash scripts/run_all.sh
   ```

## Expected inputs (by default paths)
- `data/primary_final_round_by_ed.csv`:
  - `borough,AD,ED,pri_mam_share,pri_cuo_share,pri_total_valid`
- `data/early_votes_by_ed.csv`:
  - `borough,AD,ED,ev25_total,ev_rate,ev_weight,ev_party_dem,ev_party_rep,ev_party_oth`
- `data/sliwa21_by_ed.csv`:
  - `borough,AD,ED,sliwa21_share_by_ed,Total21`\
- `data/reg_by_ed.csv`:
  - `borough,AD,ED,regs_total_2025, reg_dem,reg_rep,reg_oth`
- `data/acs_by_ed.csv` (ED‑mapped):
  - `borough,AD,ED,age_18_29_share,student_share,income_pctl,renter_share,recent_mover_share,dist_to_ev_site_m`
- `data/polls_citywide.csv`:
  - single row: `poll_mam_city_share,poll_cuo_city_share,poll_sli_city_share`

If some columns are not available, leave placeholders; the code will warn and continue when possible.

## Outputs
- `out/model_base_by_ed.parquet`
- `out/turnout_scenarios.parquet`
- `out/features_ev_by_ed.parquet`
- `out/features_r_by_ed.parquet`

## Notes
- Keys: we canonicalize `(borough, AD, ED)` and add `ed_uid = borough_id*10000 + AD*100 + ED` where `borough_id` follows {1:MN,2:BX,3:BK,4:QN,5:SI} unless the file already contains a numeric borough code.
- We clamp probabilities with `eps=1e-6` before applying `logit`.
- MECE: pick one from each “or” pair; the default config selects `student_share` and `income_pctl` once and flips sign in E‑Day where needed.
# PredictNYC
# PredictNYC
# PredictNYC
