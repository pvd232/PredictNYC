import pandas as pd
import numpy as np
from .utils import load_config

import unicodedata as ud
import re


def normalize_uid(s: str) -> str:
    if pd.isna(s):
        return s
    # Unicode normalize + strip
    s = ud.normalize("NFC", str(s)).strip()
    # Replace weird dashes/minus & soft hyphen with ASCII '-'
    s = re.sub(r"[\u2010-\u2015\u2212\u00AD]", "-", s)
    # Collapse internal whitespace around dashes
    s = re.sub(r"\s*-\s*", "-", s)
    return s



def main():
    cfg = load_config()
    P = cfg["paths"]
    base = pd.read_parquet(P["out_base"])
    X_EV = pd.read_parquet(P["out_features_ev"])
    X_R  = pd.read_parquet(P["out_features_r"])
    scen = pd.read_parquet(P["out_turnout"])



    # 0) Quick counts
    print(f"[validate] rows in base: {len(base):,}")
    print(f"[validate] distinct ed_uid (raw): {base['ed_uid'].nunique():,}")

    # 1) Show exact dupes (raw)
    dupes_raw = base[base.duplicated("ed_uid", keep=False)].sort_values("ed_uid")
    if not dupes_raw.empty:
        print("\n[validate] RAW duplicate ed_uid groups (top 50 rows):")
        print(
            dupes_raw[["ed_uid", "AD", "ED", "borough"]]
            .head(50)
            .to_string(index=False)
        )

    # 2) Normalize and re-check (catches QN-AD23 vs QN-AD23 etc.)
    norm = base.assign(ed_uid_norm=base["ed_uid"].map(normalize_uid))
    print(f"[validate] distinct ed_uid (normalized): {norm['ed_uid_norm'].nunique():,}")

    dupes_norm = norm[norm.duplicated("ed_uid_norm", keep=False)].sort_values("ed_uid_norm")
    if not dupes_norm.empty:
        print("\n[validate] NORMALIZED duplicate ed_uid groups (top 50 rows):")
        print(
            dupes_norm[
                ["ed_uid", "ed_uid_norm", "AD", "ED", "borough"]
            ]
            .head(50)
            .to_string(index=False)
        )

    # 3) Check for upstream many-to-many explosion:
    dupe_combo = base.duplicated(["AD", "ED", "borough"], keep=False)
    if dupe_combo.any():
        print("\n[validate] Duplicate (AD, ED, borough) combos detected:")
        print(
            base.loc[dupe_combo, ["AD", "ED", "borough", "ed_uid"]]
            .sort_values(["AD", "ED", "borough"])
            .head(50)
            .to_string(index=False)
        )

    # 4) If you want to *enforce* normalization before the assert, uncomment:
    # base["ed_uid"] = base["ed_uid"].map(normalize_uid)

    # 5) Final hard assert (keep this)
    assert base["ed_uid"].is_unique, "ed_uid must be unique in base"


    # Key coverage
    assert base["ed_uid"].is_unique, "ed_uid must be unique in base"
    assert set(base["ed_uid"]) == set(X_EV["ed_uid"]) == set(X_R["ed_uid"]), "Feature keys mismatch"
    # Turnout weights
    assert ((scen["wEV"] >= 0) & (scen["wEV"] <= 1)).all(), "wEV out of bounds"
    assert ((scen["wR"]  >= 0) & (scen["wR"]  <= 1)).all(), "wR out of bounds"

    # Offsets finite
    for o in ["o_M","o_C","o_S"]:
        assert np.isfinite(base[o]).all(), f"{o} contains non-finite values"

    # Z-score sanity (rough)
    for df, name in [(X_EV,"EV"), (X_R,"R")]:
        zcols = [c for c in df.columns if "__z_" in c]
        for c in zcols:
            s = df[c].dropna()
            assert s.std(ddof=0) > 0, f"{name}:{c} has zero variance"

    print("✅ Validation passed.")

if __name__ == "__main__":
    main()
