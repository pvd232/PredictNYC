import re, numpy as np, pandas as pd

RAW = "../data/original/mayoral_votes_full.csv"
MAP = "../data/ed_manifest/ed_borough_map.csv"
MAMDANI_ID, CUOMO_ID = "254286", "254052"

df = pd.read_csv(RAW, dtype=str)
df_MAP = pd.read_csv(MAP, dtype=str)
df.columns = df.columns.str.strip()

rank_cols = [c for c in df.columns if "mayor" in c.lower() and "choice" in c.lower()]
rank_cols = sorted(
    rank_cols, key=lambda c: int(re.search(r"choice\s+(\d+)", c.lower()).group(1))
)


def parse_num(s, pat):
    m = re.search(pat, s or "")
    return int(m.group(1)) if m else np.nan


df["AD"] = df["Precinct"].apply(lambda s: parse_num(s, r"AD:\s*(\d+)")).astype("Int32")
df["ED"] = df["Precinct"].apply(lambda s: parse_num(s, r"ED:\s*(\d+)")).astype("Int32")


def top_two(row):
    for c in rank_cols:
        v = str(row[c]).strip().lower()
        if v in ("", "nan", "none", "undervote", "overvote"):
            continue
        cid = re.sub(r"\D+", "", v)
        if cid == MAMDANI_ID:
            return "Mamdani"
        if cid == CUOMO_ID:
            return "Cuomo"
    return None


df["final_choice"] = df.apply(top_two, axis=1)

ed = (
    df.dropna(subset=["AD", "ED", "final_choice"])
    .groupby(["AD", "ED", "final_choice"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# ensure both candidates exist as columns
for col in ("Mamdani", "Cuomo"):
    if col not in ed.columns:
        ed[col] = 0

# make mapping dtypes match and dedupe
df_MAP = df_MAP.rename(columns={"borough_code": "borough"})
df_MAP["AD"] = df_MAP["AD"].astype("Int32")
df_MAP["ED"] = df_MAP["ED"].astype("Int32")
df_MAP = df_MAP.drop_duplicates(subset=["AD", "ED"])

# ✅ merge on keys instead of Series comparisons
ed = ed.merge(df_MAP[["AD", "ED", "borough"]], on=["AD", "ED"], how="left")

ed["pri_total_valid"] = ed["Mamdani"] + ed["Cuomo"]
ed["pri_mam_share"] = ed["Mamdani"] / ed["pri_total_valid"].replace(0, np.nan)
ed["pri_cuo_share"] = 1 - ed["pri_mam_share"]
ed.drop(columns=["Cuomo","Mamdani"],inplace=True)

# Quick validation: ensure no borough values are null
null_boroughs = ed[ed["borough"].isna() | ed["borough"].eq("")]
if null_boroughs.empty:
    print("✅ All rows have a valid borough value.")
else:
    print(f"⚠️ Found {len(null_boroughs)} rows with missing boroughs:")
    print(null_boroughs[["AD", "ED"]].head(20).to_string(index=False))
# --- Back-fill missing (AD,ED) from the map with zeros ---
# Universe of keys from the map
key_universe = df_MAP[["AD", "ED", "borough"]].drop_duplicates()

# Left-join existing ED results onto the full key universe
ed_full = key_universe.merge(ed, on=["AD", "ED", "borough"], how="left")

# For rows that didn't exist in the primary, set all outputs to 0
for c in ["pri_total_valid", "pri_mam_share", "pri_cuo_share"]:
    if c not in ed_full.columns:
        ed_full[c] = 0
    ed_full[c] = pd.to_numeric(ed_full[c], errors="coerce").fillna(0)

# Optional: tidy types & sort
ed_full["AD"] = ed_full["AD"].astype("Int32")
ed_full["ED"] = ed_full["ED"].astype("Int32")
ed_full = ed_full.sort_values(["borough", "AD", "ED"])

# Overwrite ed with the completed DataFrame
ed = ed_full

ed.to_csv("../data/primary_final_round_by_ed.csv", index=False)
