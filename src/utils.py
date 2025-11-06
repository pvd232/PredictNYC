import pandas as pd
import numpy as np
import yaml
from pathlib import Path

EPS = 1e-6

def load_config():
    # Try repo-root config.yml; fallback to sample/config.yml
    for p in [Path("config.yml"), Path("sample/config.yml")]:
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("No config.yml found (checked ./config.yml and ./sample/config.yml)")

def borough_to_id(borough, mapping):
    if pd.api.types.is_integer_dtype(pd.Series([borough])):
        return int(borough)
    b = str(borough).strip().upper()
    rev = {k.upper(): v for k, v in mapping.items()}
    if b in rev:
        return rev[b]
    # Try numeric in string
    try:
        return int(b)
    except:
        raise ValueError(f"Unrecognized borough: {borough}")

# def make_ed_uid(df,mapping):
#     print("mapping", mapping)
#     boro_col = df["borough"].apply(lambda x: borough_to_id(x, mapping))
#     ad_num = df["AD"].astype(int)
#     print("ad_num", ad_num)
#     ed_num = df["ED"].astype(int)
#     ad = f"{ad_num:02d}"  # 2 digits: 01..87
#     ed = f"{ed_num:03d}"  # 3 digits: 001..200+
#     return f"{boro_col}-AD{ad}-ED{ed}"


def make_ed_uid(df, mapping):
    print("mapping", mapping)
    

    boro = (
        df["borough"].apply(lambda x: borough_to_id(x, mapping)).astype(str).str.strip()
    )
    ad = df["AD"].astype("Int64").astype(str).str.zfill(2)
    ed = df["ED"].astype("Int64").astype(str).str.zfill(3)
    return boro + "-AD" + ad + "-ED" + ed


def logit(p):
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def zscore(s, eps=1e-9):
    s = s.astype(float)
    m = s.mean()
    v = s.std(ddof=0)
    if v < eps:
        return (s - m)
    return (s - m) / v

def read_csv_safe(path):
    return pd.read_csv(path)

def save_parquet(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    return True
