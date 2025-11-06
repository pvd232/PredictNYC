from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_DIR = Path("./dem_primary")
OUT_CSV = Path("./mayoral_votes_full.csv")


def get_engine():
    try:
        import python_calamine  # noqa: F401

        return "calamine"
    except Exception:
        return "openpyxl"


def process_one(path: Path):
    engine = get_engine()
    try:
        df = pd.read_excel(path, sheet_name="Sheet1", dtype=str, engine=engine)
        df.insert(0, "source_file", path.name)

        # --- keep only mayor columns ---
        mayor_cols = [c for c in df.columns if "mayor" in c.lower()]
        if not mayor_cols:
            print(f"⚠️  No mayor columns found in {path.name}")
            return pd.DataFrame()

        keep_cols = ["source_file", "Cast Vote Record"] + mayor_cols
        df = df[keep_cols]
        print(f"{path.name:25s} -> {len(df):7,d} rows | {len(mayor_cols)} mayor cols")
        return df

    except Exception as e:
        print(f"❌ Error reading {path.name}: {e}")
        return pd.DataFrame()


def main():
    files = sorted(INPUT_DIR.glob("2025P*V1_*.xlsx"))
    print(f"Found {len(files)} Excel files total")

    with ThreadPoolExecutor(max_workers=8) as ex:
        dfs = [
            f.result() for f in as_completed([ex.submit(process_one, f) for f in files])
        ]

    combined = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote combined mayoral dataset: {len(combined):,} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
