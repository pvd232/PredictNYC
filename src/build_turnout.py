#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import load_config, read_csv_safe, save_parquet


def main():
    cfg = load_config()
    P = cfg["paths"]

    # --- Load base ---
    base = (
        pd.read_parquet(P["out_base"])
        if str(P["out_base"]).endswith(".parquet")
        else read_csv_safe(P["out_base"])
    )

    # --- Ensure required columns ---
    must = ["ed_uid", "borough", "AD", "ED", "ev25_total", "Total21"]
    for c in must:
        if c not in base.columns:
            base[c] = np.nan

    base["ev25_total"] = pd.to_numeric(base["ev25_total"], errors="coerce").fillna(0.0)
    base["Total21"] = pd.to_numeric(base["Total21"], errors="coerce").fillna(0.0)

    # Fallback registration reference
    if "regs_total_2025" in base.columns:
        base["regs_total_2025"] = pd.to_numeric(
            base["regs_total_2025"], errors="coerce"
        ).fillna(0.0)
    else:
        base["regs_total_2025"] = 0.0

    # --- Reference shapes ---
    ev_ref = base["ev25_total"].copy()
    if ev_ref.sum() <= 0.0:
        raise ValueError("ev25_total is empty — check inputs.")

    day_ref = base["Total21"].copy()
    if day_ref.sum() <= 0.0:
        if base["regs_total_2025"].sum() > 0.0:
            day_ref = base["regs_total_2025"].copy()
        else:
            day_ref = pd.Series(1.0, index=base.index)

    # --- Scenarios ---
    scenarios = cfg.get("scenarios", {})
    T_list = scenarios.get("T", [])
    rho_list = scenarios.get("rho", [])
    if not T_list or not rho_list:
        raise ValueError("config.yml missing scenarios: { T: [...], rho: [...] }")

    out_rows = []

    for T in T_list:
        for rho in rho_list:
            tag = f"T{int(T)}_r{str(rho).replace('.', '')}"

            # Conceptual tweak:
            # We *allow* global EV scaling to hit rho*T,
            # but we keep EV's internal spatial pattern fixed (no reweighting).
            EV_target = float(T) * float(rho)
            ED_target = float(T) - EV_target

            # Keep proportional shape identical to observed EV25
            ev_scale = EV_target / max(ev_ref.sum(), 1e-9)
            EV_i = ev_ref.values * ev_scale

            # E-Day scaling as before
            day_scale = ED_target / max(day_ref.sum(), 1e-9)
            ED_i = day_ref.values * day_scale

            Ti = EV_i + ED_i

            with np.errstate(invalid="ignore", divide="ignore"):
                wEV = np.divide(EV_i, Ti, out=np.zeros_like(Ti), where=Ti > 0)
            wEV = np.clip(wEV, 0.0, 1.0)
            wR = 1.0 - wEV

            out_rows.append(
                pd.DataFrame(
                    {
                        "ed_uid": base["ed_uid"],
                        "borough": base.get("borough"),
                        "AD": base.get("AD"),
                        "ED": base.get("ED"),
                        "scenario_tag": tag,
                        "Ti": Ti.astype(float),
                        "wEV": wEV.astype(float),
                        "wR": wR.astype(float),
                    }
                )
            )

    out = pd.concat(out_rows, ignore_index=True)
    save_parquet(out, P["out_turnout"])
    print(
        f"✅ Wrote {len(out):,} rows → {P['out_turnout']} "
        f"(EDs={base['ed_uid'].nunique():,}, scenarios={len(T_list)*len(rho_list)})"
    )


if __name__ == "__main__":
    main()
