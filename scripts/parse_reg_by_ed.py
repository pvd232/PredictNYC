#!/usr/bin/env python3
# Build registration_by_ad.csv from the CityCouncil_Active_Inactive.pdf
# and AD→borough mapping (via ed_borough_map.csv or ed_borough_mapping.csv).
#
# Output columns (now per-ED):
#   borough,AD,ED,regs_total_2025,reg_dem,reg_rep,reg_oth,new_regs_2024_25
#
# Notes:
# - "reg_oth" = Conservative + Working Families + Other + Blank (from the PDF).
# - Per-ED rows evenly split each AD's counts; per-ED sums equal the AD totals.
#
# Requirements: pip install pdfplumber pandas

import re
import pandas as pd
import pdfplumber

PDF = "./data/original/CityCouncil_Active_Inactive.pdf"
MAP1 = "./data/ed_manifest/ed_borough_map.csv"
OUT = "./data/reg_by_ed.csv"


def split_evenly(total, n):
    # returns a list of n integers that sum to total
    base = total // n
    rem = total % n
    parts = [base] * n
    for i in range(rem):
        parts[i] += 1
    return parts


# --- load AD→borough and ED lists ---
if not MAP1:
    raise FileNotFoundError("Missing ed_borough_map.csv (or ed_borough_mapping.csv)")

m = pd.read_csv(MAP1, dtype=str)
if "borough" not in m.columns and "borough_code" in m.columns:
    m["borough"] = m["borough_code"]

m = m[["AD", "ED", "borough"]].dropna()
m["AD"] = m["AD"].astype(int)

# AD -> borough (mode)
ad_to_borough = (
    m.groupby("AD")["borough"]
    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
    .to_dict()
)

# AD -> sorted list of zero-padded ED strings
eds_by_ad = {}
for _, row in m.iterrows():
    ad_val = int(row["AD"])
    try:
        ed_val = int(row["ED"])
        ed_str = f"{ed_val:03d}"
    except Exception:
        raise ValueError("no ED value")        
    if ad_val not in eds_by_ad:
        eds_by_ad[ad_val] = []
    eds_by_ad[ad_val].append(ed_str)
for ad_val in list(eds_by_ad.keys()):
    eds_by_ad[ad_val] = sorted(eds_by_ad[ad_val])

# --- parse PDF to collect per-AD totals ---
ad_totals = {}  # AD -> dict(dem=, rep=, con=, wf=, oth=, blank=, total=)

ad_re = re.compile(r"^\s*AD:\s*(\d{2,3})\b")
tot_re = re.compile(
    r"^\s*Total:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)

current_ad = None

with pdfplumber.open(PDF) as pdf:
    for page in pdf.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            m_ad = ad_re.match(line)
            if m_ad:
                current_ad = int(m_ad.group(1))
                continue
            m_tot = tot_re.match(line)
            if m_tot and current_ad is not None:
                dem, rep, con, wf, oth, blank, total = map(int, m_tot.groups())
                d = ad_totals.setdefault(
                    current_ad,
                    {
                        "dem": 0,
                        "rep": 0,
                        "con": 0,
                        "wf": 0,
                        "oth": 0,
                        "blank": 0,
                        "total": 0,
                    },
                )
                d["dem"] += dem
                d["rep"] += rep
                d["con"] += con
                d["wf"] += wf
                d["oth"] += oth
                d["blank"] += blank
                d["total"] += total

# --- build per-ED rows (evenly distribute AD totals across EDs) ---
rows = []
for ad in sorted(ad_totals.keys()):
    d = ad_totals[ad]
    borough = ad_to_borough.get(ad)
    ed_list = eds_by_ad.get(ad, [])    
    n = len(ed_list)
    dem_parts = split_evenly(d["dem"], n)
    rep_parts = split_evenly(d["rep"], n)
    con_parts = split_evenly(d["con"], n)
    wf_parts = split_evenly(d["wf"], n)
    oth_parts = split_evenly(d["oth"], n)
    blank_parts = split_evenly(d["blank"], n)

    for i, ed_str in enumerate(ed_list):
        dem_i = dem_parts[i]
        rep_i = rep_parts[i]
        con_i = con_parts[i]
        wf_i = wf_parts[i]
        oth_i = oth_parts[i]
        blank_i = blank_parts[i]

        reg_oth_i = con_i + wf_i + oth_i + blank_i
        total_i = dem_i + rep_i + reg_oth_i  # equals dem+rep+con+wf+oth+blank
        rows.append(
            {
                "borough": borough,
                "AD": ad,
                "ED": ed_str,
                "regs_total_2025": total_i,
                "reg_dem": dem_i,
                "reg_rep": rep_i,
                "reg_oth": reg_oth_i,
            }
        )
num = 0
for l in eds_by_ad.values():
    num += len(l)
assert len(rows) == num, print(f"ED: {str(num)}, OUT: {str(len(rows))}")

for row in rows:
    assert(row["regs_total_2025"] >= 1)
# for ed in eds_by_ad.values():

#     ed = int(row["ED"])

print("len", len(rows))
out = pd.DataFrame(
    rows,
    columns=[
        "borough",
        "AD",
        "ED",
        "regs_total_2025",
        "reg_dem",
        "reg_rep",
        "reg_oth",
    ],
).sort_values(["borough", "AD", "ED"])

out.to_csv(OUT, index=False)
print(f"✅ Wrote {OUT} with {len(out):,} rows (per-ED, evenly split)")
