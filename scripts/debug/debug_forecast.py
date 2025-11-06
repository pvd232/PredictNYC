#!/usr/bin/env python3
# Pure diagnostics: READS existing outputs only. Never calls the fitter.

import numpy as np
import pandas as pd
from pathlib import Path


def _summ(df, cols, name):
    print(f"\n=== {name} ===")
    for c in cols:
        if c not in df.columns:
            print(f"⚠️  Missing column: {c}")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        print(
            f"{c:20s}  n={len(s):5d}  null={int(s.isna().sum()):5d}  "
            f"zeros={(s==0).sum():5d}  infs={(~np.isfinite(s)).sum():5d}  "
            f"min={s.replace([np.inf,-np.inf],np.nan).min():.6f}  "
            f"mean={s.replace([np.inf,-np.inf],np.nan).mean():.6f}  "
            f"max={s.replace([np.inf,-np.inf],np.nan).max():.6f}"
        )


def _head(df, name, n=5, cols=None):
    print(f"\n--- {name} sample ---")
    if cols:
        try:
            print(df[cols].head(n).to_string(index=False))
            return
        except Exception:
            pass
    print(df.head(n).to_string(index=False))


def _read(p):
    if not p.exists():
        print(f"❌ Missing: {p}")
        return None
    print(f"✅ Loaded: {p}")
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def safe_group_weighted(df, by):
    """
    Returns weighted city/borough/AD aggregates but NEVER throws if weights sum to 0.
    Falls back to unweighted mean and sets T=0 in that group.
    """
    if df is None or df.empty:
        return None

    d = df.copy()
    for c in ["p_M", "p_C", "p_S", "Ti"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["p_M", "p_C", "p_S"]:
        if c in d.columns:
            d[c] = d[c].clip(0.0, 1.0)

    def _fn(g):
        Ti = g.get("Ti", pd.Series([], dtype=float)).fillna(0.0).to_numpy(float)
        P = g[["p_M", "p_C", "p_S"]].to_numpy(float)
        wsum = float(np.nansum(Ti))
        if not np.isfinite(wsum) or wsum <= 0.0:
            m = np.nanmean(P, axis=0)
            Ttot = 0.0
        else:
            m = np.average(P, weights=Ti, axis=0)
            Ttot = wsum
        m = np.nan_to_num(m, nan=0.0)
        return pd.Series({"p_M": m[0], "p_C": m[1], "p_S": m[2], "T": Ttot})

    out = d.groupby(by, dropna=False).apply(_fn).reset_index()
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "out"

    base_p = out_dir / "model_base_by_ed.parquet"
    turn_p = out_dir / "turnout_scenarios.parquet"
    ed_p = out_dir / "forecast_ed_by_scenario.csv"
    ad_p = out_dir / "forecast_ad_by_scenario.csv"
    bo_p = out_dir / "forecast_borough_by_scenario.csv"
    ct_p = out_dir / "forecast_city_by_scenario.csv"

    base = _read(base_p)
    turn = _read(turn_p)
    ed = _read(ed_p)
    ad = _read(ad_p)
    bo = _read(bo_p)
    ct = _read(ct_p)

    if base is not None:
        print(f"\nbase: {len(base):,} rows x {len(base.columns)} cols")
        _summ(
            base,
            [
                "pri_mam_share",
                "sliwa21_share_by_ed",
                "sliwa21_share_eb",
                "o_M",
                "o_S",
                "ev25_total",
                "Total21",
            ],
            "BASE",
        )

    if turn is not None:
        print(f"\nturnout: {len(turn):,} rows x {len(turn.columns)} cols")
        tcols = [
            c for c in turn.columns if c == "Ti" or c.startswith(("Ti_", "wEV", "wR"))
        ]
        _summ(turn, tcols, "TURNOUT")
        # ensure per-scenario completeness
        if "scenario_tag" in turn.columns:
            chk = turn.groupby("scenario_tag")["Ti"].sum(min_count=1)
            print("\nScenario total Ti by tag:\n", chk.to_string())

    if ed is not None:
        print(f"\ned: {len(ed):,} rows x {len(ed.columns)} cols")
        _summ(ed, ["p_M", "p_C", "p_S", "Ti", "wEV", "wR"], "ED")
        # do probs sum to ~1?
        if set(["p_M", "p_C", "p_S"]).issubset(ed.columns):
            s = ed[["p_M", "p_C", "p_S"]].sum(axis=1)
            print(
                f"\nED probs sum: mean={s.mean():.6f} min={s.min():.6f} max={s.max():.6f}"
            )
        _head(
            ed,
            "ED sample",
            cols=[
                "ed_uid",
                "borough",
                "AD",
                "ED",
                "scenario_tag",
                "p_M",
                "p_C",
                "p_S",
                "Ti",
                "wEV",
                "wR",
            ],
        )

    # Recompute safe aggregates (won't crash)
    if ed is not None:
        ad_safe = safe_group_weighted(ed, ["scenario_tag", "borough", "AD"])
        bo_safe = safe_group_weighted(ed, ["scenario_tag", "borough"])
        ct_safe = safe_group_weighted(ed, ["scenario_tag"])

        if ad_safe is not None:
            print(f"\n(ad recompute) {len(ad_safe):,} rows")
            _head(ad_safe, "AD agg (safe)")
        if bo_safe is not None:
            print(f"\n(borough recompute) {len(bo_safe):,} rows")
            _head(bo_safe, "Borough agg (safe)")
        if ct_safe is not None:
            print(f"\n(city recompute) {len(ct_safe):,} rows")
            print(ct_safe.to_string(index=False))

    print("\n✅ Diagnostics complete (no weighted-average crashes).")


if __name__ == "__main__":
    main()
