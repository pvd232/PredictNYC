#!/usr/bin/env python3
# Build ../data/mayor_2021_general_by_ed.csv from:
#   - ../data/original/2021_mayor.csv  (paired key/value rows)
#   - ../data/ed_manifest/ed_borough_map.csv  (universe of (borough,AD,ED))
#
# Output columns:
#   borough,AD,ED,votes_sliwa,votes_adams,two_party_total,
#   sliwa21_share_by_ed,Total21
#
# Guarantees:
#   * Exactly the same (borough,AD,ED) rows as the map (no fewer, no more)
#   * No NaNs in sliwa21_share_by_ed (imputed if needed)

import csv
import numpy as np
import pandas as pd

SRC = "../data/original/2021_mayor.csv"
MAP = "../data/ed_manifest/ed_borough_map.csv"
OUT = "../data/mayor_2021_general_by_ed.csv"


# ---------------- tiny helpers ----------------
def to_int(x):
    try:
        return int(str(x).strip().strip('"').strip("'"))
    except Exception:
        return None


def is_adams(s):
    return isinstance(s, str) and ("adams" in s.lower()) and ("eric" in s.lower())


def is_sliwa(s):
    return isinstance(s, str) and ("sliwa" in s.lower())


BALLOT_UNITS = {
    "Public Counter",
    "Manually Counted Emergency",
    "Absentee / Military",
    "Affidavit",
    "Early Voting",
}


def _weighted_mean(series, weights):
    series = pd.to_numeric(series, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = series.notna() & (weights > 0)
    if mask.any():
        return float((series[mask] * weights[mask]).sum() / weights[mask].sum())
    return np.nan


# ---------------- load map (KEY UNIVERSE) ----------------
m = pd.read_csv(MAP, dtype=str)
if "borough" not in m.columns and "borough_code" in m.columns:
    m["borough"] = m["borough_code"]

need = {"borough", "AD", "ED"}
missing = need - set(m.columns)
if missing:
    raise ValueError(f"Map file missing columns: {missing}")

universe = m[["borough", "AD", "ED"]].copy()
universe["borough"] = universe["borough"].astype(str).str.strip().str.upper()
universe["AD"] = universe["AD"].apply(to_int)
universe["ED"] = universe["ED"].apply(to_int)
universe = universe.dropna(subset=["borough", "AD", "ED"]).drop_duplicates()

# Check dup keys in map
dups = universe[universe.duplicated(["borough", "AD", "ED"], keep=False)]
if not dups.empty:
    print("⚠️ Map has duplicate (borough,AD,ED) keys (showing first 20):")
    print(dups.head(20).to_string(index=False))

# ---------------- parse the paired key/value 2021 CSV ----------------
recs = []
with open(SRC, "r", encoding="utf-8-sig", newline="") as f:
    r = csv.reader(f)
    for row in r:
        if not row:
            continue
        n = len(row)
        if n % 2 != 0:
            # skip malformed lines
            continue
        keys = row[: n // 2]
        vals = row[n // 2 :]
        recs.append(dict(zip(keys, vals)))

df = pd.DataFrame.from_records(recs)

# keep only Mayor / General 2021 / IN-PLAY
df = df[
    (df.get("Office/Position Title") == "Mayor")
    & (df.get("Event", "").str.contains("General Election 2021", na=False))
    & (df.get("EDAD Status").fillna("IN-PLAY").eq("IN-PLAY"))
].copy()

# coerce types we need
df["AD"] = df["AD"].apply(to_int)
df["ED"] = df["ED"].apply(to_int)
df["Tally"] = df["Tally"].apply(lambda x: int(str(x)) if str(x).isdigit() else 0)

# aggregate by (AD, ED) across the raw
g = df.groupby(["AD", "ED"], dropna=True)

votes_adams = g.apply(
    lambda t: t.loc[t["Unit Name"].apply(is_adams), "Tally"].sum()
).rename("votes_adams")
votes_sliwa = g.apply(
    lambda t: t.loc[t["Unit Name"].apply(is_sliwa), "Tally"].sum()
).rename("votes_sliwa")
Total21 = g.apply(
    lambda t: t.loc[~t["Unit Name"].isin(BALLOT_UNITS), "Tally"].sum()
).rename("Total21")

agg = pd.concat([votes_sliwa, votes_adams, Total21], axis=1).reset_index()

# ---------------- attach to universe (this guarantees counts match map) ----------------
# NOTE: We merge by AD/ED; borough comes from the map and is authoritative.
out = universe.merge(agg, on=["AD", "ED"], how="left", validate="one_to_one")

# numeric hygiene
for c in ["votes_sliwa", "votes_adams", "Total21"]:
    if c not in out.columns:
        out[c] = 0
    out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")

# derived fields
out["two_party_total"] = (out["votes_sliwa"] + out["votes_adams"]).astype("int64")

# share — set NaN where two-party total is 0; will impute below
out["sliwa21_share_by_ed"] = np.where(
    out["two_party_total"] > 0,
    out["votes_sliwa"] / out["two_party_total"].replace(0, np.nan),
    np.nan,
).astype(float)

# ---------------- thorough imputation of missing shares ----------------
# Weighted AD mean → weighted borough mean → weighted citywide → 0.5
# weights = two_party_total
out["__w"] = out["two_party_total"].astype(float)
out["__s"] = out["sliwa21_share_by_ed"]

# weighted AD mean
ad_means = out.groupby(["borough", "AD"], as_index=False).apply(
    lambda g: pd.Series({"__ad_mean": _weighted_mean(g["__s"], g["__w"])})
)
out = out.merge(ad_means, on=["borough", "AD"], how="left")

# weighted borough mean
bor_means = out.groupby("borough", as_index=False).apply(
    lambda g: pd.Series({"__bor_mean": _weighted_mean(g["__s"], g["__w"])})
)
out = out.merge(bor_means, on="borough", how="left")

# weighted citywide mean
wsum = out["__w"].sum()
city_mean = float((out["__s"] * out["__w"]).sum() / wsum) if wsum > 0 else np.nan
if not np.isfinite(city_mean):
    city_mean = 0.5  # neutral ultimate fallback

miss = out["sliwa21_share_by_ed"].isna()
impute = (
    out.loc[miss, "__ad_mean"]
    .where(out.loc[miss, "__ad_mean"].notna(), out.loc[miss, "__bor_mean"])
    .fillna(city_mean)
)
out.loc[miss, "sliwa21_share_by_ed"] = impute
# final guard
out["sliwa21_share_by_ed"] = out["sliwa21_share_by_ed"].fillna(city_mean).clip(0.0, 1.0)

# cleanup temps
out.drop(
    columns=["__w", "__s", "__ad_mean", "__bor_mean"], inplace=True, errors="ignore"
)

# ---------------- final ordering, types, asserts ----------------
out["AD"] = out["AD"].astype("Int32")
out["ED"] = out["ED"].astype("Int32")

out = out[
    [
        "borough",
        "AD",
        "ED",
        "votes_sliwa",
        "votes_adams",
        "two_party_total",
        "sliwa21_share_by_ed",
        "Total21",
    ]
].sort_values(["borough", "AD", "ED"])

# hard sanity: row counts must match map
n_map = len(universe)
n_out = len(out)
print(f"[counts] map rows={n_map:,} | output rows={n_out:,}")
assert n_out == n_map, f"Output rows ({n_out}) must equal map rows ({n_map})"

# no NaNs in share
n_null_share = int(out["sliwa21_share_by_ed"].isna().sum())
print(f"[nulls] sliwa21_share_by_ed NaNs: {n_null_share}")
assert n_null_share == 0, "There are still NaNs in sliwa21_share_by_ed"

# no missing boroughs
assert (
    out["borough"].notna().all() & out["borough"].ne("").all()
), "Missing borough values after join"

# Save
out.to_csv(OUT, index=False)
print(f"✅ Wrote {OUT} with {len(out):,} rows (exactly matches map).")
