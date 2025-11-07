# early_votes_parser.py
import pandas as pd

IN_CSV = "./data/original/thecity_ev_dataset.csv"
OUT_CSV = "./data/ev_by_ed.csv"
MAP = "./data/ed_manifest/ed_borough_map.csv"
df = pd.read_csv(IN_CSV)

# normalize column names
df.columns = df.columns.str.strip().str.lower()

# coerce numeric fields (ignore non-numeric junk)
num_like = [
    "ct_ev_dem",
    "ct_ev_rep",
    "ct_ev_oth",
    "ct_ev_total",
    "total",
    "pct_total_100",
]
for c in num_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# --- build required outputs ---
# ev25_total: just the early-vote total count
ev25_total = df.get("ct_ev_total", pd.Series(0, index=df.index, dtype="float64"))

# 1) Split "ed" -> ED, AD (as int32 columns)
ed_ad = df.get("ed", pd.Series(index=df.index, dtype="object"))
df[["AD", "ED"]] = ed_ad.astype(str).str.extract(r"\s*(\d+)\s*/\s*(\d+)\s*$")
df["ED"] = pd.to_numeric(df["ED"], errors="raise").astype("int32")
df["AD"] = pd.to_numeric(df["AD"], errors="raise").astype("int32")

# 2) Merge borough from the map (on AD, ED) — normalize + enforce 1:1
valid = {"MN", "BX", "BK", "QN", "SI"}
alias = {
    "MANHATTAN": "MN",
    "NEW YORK": "MN",
    "NY": "MN",
    "BRONX": "BX",
    "KINGS": "BK",
    "BROOKLYN": "BK",
    "QUEENS": "QN",
    "RICHMOND": "SI",
    "STATEN ISLAND": "SI",
    "MN": "MN",
    "BX": "BX",
    "BK": "BK",
    "QN": "QN",
    "SI": "SI",
}


def to_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper().replace(".", "")
    s = alias.get(s, s)
    s = s.replace("-", " ").strip()
    s = alias.get(s, s)
    return s if s in valid else None


m = pd.read_csv(MAP, dtype=str)

if "borough" not in m.columns and "borough_code" in m.columns:
    m["borough"] = m["borough_code"]

required = {"AD", "ED", "borough"}
missing = required - set(m.columns)
if missing:
    raise ValueError(f"Mapping file missing columns: {missing}")

m = m[["AD", "ED", "borough"]].copy()
m["AD"] = pd.to_numeric(m["AD"], errors="coerce").astype("Int32")
m["ED"] = pd.to_numeric(m["ED"], errors="coerce").astype("Int32")
m["borough"] = m["borough"].map(to_code)
df = df.merge(m, on=["AD", "ED"], how="left", validate="many_to_one")

# ev_rate: prefer provided percent (pct_total_100), else fall back to ct_ev_total / total
if "pct_total_100" in df.columns and df["pct_total_100"].notna().any():
    ev_rate = df["pct_total_100"] / 100.0
else:
    denom = df.get("total", pd.Series(pd.NA, index=df.index, dtype="float64"))
    ev_rate = ev25_total / denom.replace(0, pd.NA)


# party shares among early voters (safe divide)
def safe_div(n, d):
    return (n / d).where(d != 0)


ev_party_dem = safe_div(df.get("ct_ev_dem", 0), ev25_total)
ev_party_rep = safe_div(df.get("ct_ev_rep", 0), ev25_total)
ev_party_oth = safe_div(df.get("ct_ev_oth", 0), ev25_total)

# ev_weight: each ED's share of citywide early votes (sums to 1)
total_ev = pd.to_numeric(ev25_total, errors="coerce").fillna(0).sum()
ev_weight = (
    (ev25_total / total_ev)
    if total_ev > 0
    else pd.Series(0.0, index=df.index, dtype="float64")
)

out = pd.DataFrame(
    {
        "ED": df["ED"].values,
        "AD": df["AD"].values,
        "borough": df["borough"].values,  # already normalized to codes,
        "neighborhood": df["name"].values,
        "ev25_total": ev25_total,
        "ev_rate": ev_rate,
        "ev_weight": ev_weight,
        "ev_party_dem": ev_party_dem,
        "ev_party_rep": ev_party_rep,
        "ev_party_oth": ev_party_oth,
    }
)

# Fill only numeric columns
num_cols = [
    "ev25_total",
    "ev_rate",
    "ev_weight",
    "ev_party_dem",
    "ev_party_rep",
    "ev_party_oth",
]
out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Keep keys as ints (tidy)
out["AD"] = pd.to_numeric(out["AD"], errors="raise").astype("int32")
out["ED"] = pd.to_numeric(out["ED"], errors="raise").astype("int32")

# Final sanity guards
valid = {"MN", "BX", "BK", "QN", "SI"}
bad_boro = ~out["borough"].isin(valid)
assert (
    not bad_boro.any()
), f"Unexpected boroughs in EV data: {out.loc[bad_boro,'borough'].unique().tolist()}"

out.to_csv(OUT_CSV, index=False)
print(f"✅ Wrote {len(out):,} rows → {OUT_CSV}")
print(out.head())


# Optional, add neighborhood to MASTER ED/AD/borough file
def update_ed_borough_mapping():
    m_new = pd.read_csv(MAP, dtype=str)
    out_copy = out.rename(columns={"borough": "borough_code"})

    def norm_keys(df):
        df = df.copy()
        df["AD"] = (
            pd.to_numeric(df["AD"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.zfill(2)
        )
        df["ED"] = (
            pd.to_numeric(df["ED"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.zfill(3)
        )
        return df

    out_copy = norm_keys(out_copy)
    m_new = norm_keys(m_new)

    out_copy["borough"] = m_new["borough_name"]
    m_new = m_new.rename(columns={"borough_name": "borough"})

    mm = m_new.merge(
        out_copy[["ED", "AD", "borough_code", "neighborhood"]],
        how="left",
        on=["ED", "AD", "borough_code"],
    ).assign(neighborhood=lambda d: d["neighborhood"].fillna(d["borough"]))
    mm.to_csv("./data/ed_manifest/ed_borough_map.csv", index=False)


update_ed_borough_mapping()
