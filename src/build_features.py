import pandas as pd
import numpy as np
from .utils import load_config, zscore, save_parquet


def main():
    cfg = load_config()
    P = cfg["paths"]
    base = pd.read_parquet(P["out_base"])

    # Choose feature variants
    use_student = cfg["feature_choices"]["use_student_share"]
    use_income_pct = cfg["feature_choices"]["use_income_percentile"]
    use_dist = cfg["feature_choices"]["use_dist_to_ev_site"]

    # EV block
    X_EV = base[["ed_uid", "o_M", "o_C", "o_S"]].copy()
    if "age_18_29_share" in base.columns:
        X_EV["ev__z_age_18_29"] = zscore(base["age_18_29_share"])
    if use_student and "student_share" in base.columns:
        X_EV["ev__z_student"] = zscore(base["student_share"])
    if "new_regs_2024_25" in base.columns and "regs_total_2025" in base.columns:
        ratio = (base["new_regs_2024_25"] / base["regs_total_2025"]).replace(
            [np.inf, -np.inf], np.nan
        )
        X_EV["ev__z_new_regs_rate"] = zscore(ratio.fillna(ratio.mean()))
    if use_income_pct and "income_pctl" in base.columns:
        X_EV["ev__z_income_centered"] = zscore(base["income_pctl"])
    if use_dist and "dist_to_ev_site_m" in base.columns:
        X_EV["ev__z_ev_site_dist"] = zscore(base["dist_to_ev_site_m"])

    # Remaining/E-Day block
    X_R = base[["ed_uid", "o_M", "o_C", "o_S"]].copy()
    if use_income_pct and "income_pctl" in base.columns:
        # Complementary sign to avoid double counting
        X_R["eday__z_income_centered_neg"] = -zscore(base["income_pctl"])
    if "ev_rate" in base.columns:
        X_R["eday__z_remaining_gap"] = zscore(
            base["ev_rate"].mean() - base["ev_rate"]
        )  # simple proxy; refine if you add target bins
    if "renter_share" in base.columns:
        X_R["eday__z_renter"] = zscore(base["renter_share"])
    if "age_18_29_share" in base.columns:
        X_R["eday__z_age_18_29"] = zscore(base["age_18_29_share"])
    X_R["eday__z_gop_base"] = zscore(base["sliwa21_share_by_ed"])

    # Save
    save_parquet(X_EV, P["out_features_ev"])
    save_parquet(X_R, P["out_features_r"])
    print(f"✅ Wrote features: EV={X_EV.shape}  R={X_R.shape}")


if __name__ == "__main__":
    main()
