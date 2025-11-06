import pandas as pd
import numpy as np
from pathlib import Path
from .utils import (
    load_config,
    read_csv_safe,
    save_parquet,
    make_ed_uid,
    logit,
    require_cols,
)
import argparse

def _has_cols(df, cols):
    return all(c in df.columns for c in cols)


def _eb_shrink_by_borough(df, share_col, count_col, borough_col="borough", tau=2000.0):
    """
    Empirical-Bayes shrink: w_i = n_i/(n_i + tau); share_eb = w_i*share_i + (1-w_i)*borough_mean.
    """
    s = df[share_col].astype(float)
    n = df[count_col].fillna(0.0).astype(float)
    # Borough prior mean (ignoring NaNs)
    bmean = df.groupby(borough_col)[share_col].transform("mean")
    w = n / (n + float(tau))
    return w * s + (1.0 - w) * bmean


def main():

    ap = argparse.ArgumentParser(
        description="Assemble inputs for feature matrix, set turnout with --t"
    )
    ap.add_argument("--t", default="baseline")
    args = ap.parse_args()
    t_map = {"baseline": "T1800000_r04", "high": ""}
    if args.t is not None:
        turnout = args.t
    else:
        turnout

    cfg = load_config()
    P = cfg["paths"]

    pri = read_csv_safe(P["primary_final"])
    ev = read_csv_safe(P["early_votes"])
    sl = read_csv_safe(P["sliwa21"])
    # Normalize keys exactly once, before any merge
    for df, name in [(pri, "pri"), (ev, "ev"), (sl, "sl")]:
        df["borough"] = df["borough"].astype(str).str.strip().str.upper()
        df["AD"] = pd.to_numeric(df["AD"], errors="coerce").astype("Int32")
        df["ED"] = pd.to_numeric(df["ED"], errors="coerce").astype("Int32")

    # Try to read registration; make optional
    try:
        rg = read_csv_safe(P["registration"])
        require_cols(rg, ["borough", "AD", "ED"], "registration")
        rg["ed_uid"] = make_ed_uid(rg, cfg["borough_map"])
        # Soft-require; if missing, we'll fill later
        needed = [
            "reg_dem",
            "reg_rep",
            "reg_oth",
        ]
        for c in needed:
            if c not in rg.columns:
                rg[c] = np.nan
        has_registration = True
    except Exception as e:
        print(
            f"⚠️  Registration file optional and was not used ({e}). Proceeding without it."
        )
        rg = None
        has_registration = False

    try:
        acs = read_csv_safe(P["acs"])  # optional
    except Exception as _e:
        acs = None

    polls = read_csv_safe(P["polls"])

    # Allocate undecided voters
    sums = polls.sum(axis=1)
    rem = 100 * np.ones_like(len(polls)) - sums
    mam_share = cfg.get("late_deciders").get("mam")
    cuo_share = cfg.get("late_deciders").get("cuo")
    sli_share = cfg.get("late_deciders").get("sli")

    polls["zohran_mamdani"] += mam_share * rem
    polls["andrew_cuomo"] += cuo_share * rem
    polls["curtis_sliwa"] += sli_share * rem


    # Standardize keys (skip optional None frames)
    for df, name in (
        [(pri, "primary"), (ev, "early_votes"), (sl, "sliwa21")]
        + ([(rg, "registration")] if rg is not None else [])
        + ([(acs, "acs")] if acs is not None else [])
    ):
        require_cols(df, ["borough", "AD", "ED"], name)
        df["ed_uid"] = make_ed_uid(df, cfg["borough_map"])

    # Core columns
    require_cols(pri, ["pri_mam_share", "pri_cuo_share", "pri_total_valid"], "primary")
    require_cols(
        ev,
        [
            "ev25_total",
            "ev_rate",
            "ev_weight",
            "ev_party_dem",
            "ev_party_rep",
            "ev_party_oth",
        ],
        "early_votes",
    )

    require_cols(sl, ["sliwa21_share_by_ed", "Total21"], "sliwa21")
    # ACS optional; if present, must have expected cols
    if acs is not None:
        require_cols(
            acs,
            [
                "age_18_29_share",
                "student_share",
                "income_pctl",
                "renter_share",
                "recent_mover_share",
                "dist_to_ev_site_m",
            ],
            "acs",
        )

    # Merge
    base = pri.merge(
        ev, on=["borough", "AD", "ED", "ed_uid"], how="outer", suffixes=("", "")
    )
    base = base.merge(sl, on=["borough", "AD", "ED", "ed_uid"], how="outer")
    # if has_registration:
    #     base = base.merge(rg, on=["borough", "AD", "ED", "ed_uid"], how="outer")
    if acs is not None:
        base = base.merge(acs, on=["borough", "AD", "ED", "ed_uid"], how="outer")

    # Ensure registration columns exist even if file not provided
    for c in ["reg_dem", "reg_rep", "reg_oth"]:
        if c not in base.columns:
            base[c] = np.nan
    
    # --- Build Sliwa EB-smoothed proxy ---
    
    # Use two-party total if available, else fall back to Total21 for EB weight
    count_col = "two_party_total" if "two_party_total" in base.columns else "Total21"
    # If count_col is entirely missing or zero, create a minimal positive count to avoid zero weights
    if count_col not in base or base[count_col].isna().all():
        base[count_col] = 0.0
    # EB smoothing per borough
    base["sliwa21_share_eb"] = _eb_shrink_by_borough(
        base, "sliwa21_share_by_ed", count_col, borough_col="borough", tau=2000.0
    )

    # Optional GOP registration fallback for missing EB (rare)
    if {"reg_rep", "reg_dem"}.issubset(base.columns):
        gop_reg_share = (
            base["reg_rep"]
            / (base["reg_rep"] + base["reg_dem"]).replace([np.inf, -np.inf], np.nan)
        ).astype(float)
        gop_reg_share = gop_reg_share.fillna(gop_reg_share.mean())
        miss = base["sliwa21_share_eb"].isna()
        if miss.any():
            # Blend 50/50 with borough mean
            bor_mean = base.groupby("borough")["sliwa21_share_by_ed"].transform("mean")
            base.loc[miss, "sliwa21_share_eb"] = (
                0.5 * gop_reg_share[miss] + 0.5 * bor_mean[miss]
            )
    # --- DIAGNOSTICS: what's making o_M/o_S blow up? ---

    def _summ(df, col):
        s = pd.to_numeric(df[col], errors="coerce")
        print(
            f"[diag] {col}: n={len(s):,}, null={int(s.isna().sum()):,}, "
            f"+inf={(s==np.inf).sum()}, -inf={(s==-np.inf).sum()}, "
            f"min={s.replace([np.inf,-np.inf], np.nan).min()}, "
            f"max={s.replace([np.inf,-np.inf], np.nan).max()}"
        )

    # cast the key inputs to numeric once
    for c in ["pri_mam_share", "sliwa21_share_by_ed", "sliwa21_share_eb"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    _summ(base, "pri_mam_share")
    _summ(base, "sliwa21_share_by_ed")

    # Show a few rows that will make logit explode (0/1/NaN)
    bad_M_src = (
        base["pri_mam_share"].isna()
        | (base["pri_mam_share"] <= 0.0)
        | (base["pri_mam_share"] >= 1.0)
    )
    print(f"[diag] pri_mam_share problematic rows: {int(bad_M_src.sum()):,}")
    if bad_M_src.any():
        print(
            base.loc[bad_M_src, ["ed_uid", "borough", "AD", "ED", "pri_mam_share"]]
            .head(25)
            .to_string(index=False)
        )

    # Offsets
    base["o_M"] = logit(base["pri_mam_share"])
    base["o_C"] = 0.0
    alpha_S = float(cfg.get("sliwa_alpha", 0.25))
    base["o_S"] = alpha_S * logit(base["sliwa21_share_eb"].clip(1e-6, 1 - 1e-6))

    # Save
    outp = P["out_base"]
    save_parquet(base, outp)

    # Polls — keep as a one-row aux file for later scripts to read
    polls_out = Path("out/polls_citywide.parquet")
    polls.to_parquet(polls_out, index=False)

    # Coverage report
    print(f"✅ Wrote {len(base):,} ED rows → {outp}")


if __name__ == "__main__":
    main()
