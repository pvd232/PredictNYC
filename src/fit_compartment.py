#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import yaml
import pandas as pd
import numpy as np


# ---------------- paths/helpers ----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | Path) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else (repo_root() / p)


def read_config(path: str | Path):
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config at: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ---------------- math utils ----------------
def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    S = np.clip(E.sum(axis=1, keepdims=True), 1e-18, None)
    return E / S


# ---------------- poll handling ----------------
def load_poll_strict(polls_path: str | Path) -> pd.Series:
    """
    Expects columns zohran_mamdani, andrew_cuomo, curtis_sliwa.
    If eric_adams exists, we fold it into Cuomo.
    Returns normalized shares as: Mamdani, Cuomo, Sliwa
    """
    # --- Load and average all polls from parquet ---
    polls_path = Path("out/polls_citywide.parquet")
    if not polls_path.exists():
        raise FileNotFoundError(f"Expected {polls_path}; run polling ingest first.")

    polls_df = pd.read_parquet(polls_path)
    if polls_df.empty:
        raise ValueError(f"{polls_path} is empty — no polls available.")

    cols = {c.lower(): c for c in polls_df.columns}
    need = ["zohran_mamdani", "andrew_cuomo", "curtis_sliwa"]
    if not all(k in cols for k in need):
        raise KeyError(f"{polls_path} missing columns {need}; found {list(polls_df.columns)}")

    # --- Compute simple mean across all polls ---
    m = polls_df[cols["zohran_mamdani"]].mean()
    c = polls_df[cols["andrew_cuomo"]].mean()
    s = polls_df[cols["curtis_sliwa"]].mean()

    # Fold Eric Adams into Cuomo if present
    if "eric_adams" in cols:
        c += polls_df[cols["eric_adams"]].mean()

    vec = np.array([m, c, s], float)
    vec = vec / vec.sum()
    poll_row = pd.Series({"Mamdani": vec[0], "Cuomo": vec[1], "Sliwa": vec[2]})

    print(f"✅ Poll target (average of {len(polls_df)} polls):", poll_row.to_dict())


    vec = np.array([m, c, s], float)
    tot = float(np.nansum(vec))
    if tot <= 0:
        raise ValueError("Poll row sums to zero/NaN; cannot normalize.")
    vec = vec / tot
    return pd.Series({"Mamdani": vec[0], "Cuomo": vec[1], "Sliwa": vec[2]})


def shift_to_poll_means(P_df: pd.DataFrame, Ti: pd.Series, poll_row: pd.Series):
    """
    Find gamma (3-vector) so that weighted mean of softmax(base+gamma) matches poll target.
    If Ti is all zeros, return zeros.
    """
    Ti = pd.to_numeric(Ti, errors="coerce").fillna(0.0)
    if float(Ti.sum()) <= 0:
        return {"M": 0.0, "C": 0.0, "S": 0.0}

    target = np.array(
        [poll_row["Mamdani"], poll_row["Cuomo"], poll_row["Sliwa"]], float
    )
    target = target / target.sum()

    eps = 1e-9
    base_logits = np.log(np.clip(P_df[["p_M", "p_C", "p_S"]].values, eps, 1 - eps))
    W = Ti.values / max(Ti.sum(), eps)

    gamma = np.zeros(3, float)
    for _ in range(50):
        P_shift = softmax(base_logits + gamma.reshape(1, 3))
        mean = (W.reshape(-1, 1) * P_shift).sum(axis=0)
        resid = mean - target
        if np.linalg.norm(resid) < 1e-10:
            break
        # 3x3 Jacobian
        J = np.zeros((3, 3))
        for k in range(3):
            for l in range(3):
                J[k, l] = np.sum(
                    W * P_shift[:, k] * ((1.0 if k == l else 0.0) - P_shift[:, l])
                )
        try:
            step = np.linalg.solve(J, resid)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(J) @ resid
        gamma -= step
    return {"M": gamma[0], "C": gamma[1], "S": gamma[2]}


# ---------------- tag helpers ----------------
_TAG_RE = re.compile(r"^T(?P<T>\d+)_r(?P<r>\d+)$")


def parse_tag(tag: str) -> tuple[int, float] | None:
    m = _TAG_RE.match(str(tag))
    if not m:
        return None
    T = int(m.group("T"))
    r_pct = int(m.group("r"))  # e.g., 04 -> 4 (%)
    return (T, r_pct / 100.0)


def pick_ref_tag_from_available(
    tags: list[str], T_ref: float | int | None, rho_ref: float | None
) -> str:
    """
    If (T_ref, rho_ref) is present, use it; otherwise pick the nearest (min L1 distance).
    If T_ref/rho_ref are None, return the first tag.
    """
    if not tags:
        raise ValueError("No scenario_tag values available.")
    if T_ref is None or rho_ref is None:
        return tags[0]

    desired = (int(T_ref), float(rho_ref))
    scored = []
    for t in tags:
        pr = parse_tag(t)
        if pr is None:
            continue
        dist = abs(pr[0] - desired[0]) + abs(pr[1] - desired[1])
        scored.append((dist, t))
    if not scored:
        return tags[0]
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][1]


# ---------------- main ----------------
def main():
    cfg = read_config("config.yml")
    P = cfg["paths"]

    p_base = resolve_path(P["out_base"])
    p_turn = resolve_path(P["out_turnout"])
    p_fev = resolve_path(P["out_features_ev"])
    p_fr = resolve_path(P["out_features_r"])
    p_polls = resolve_path(P["polls"])

    print("Using files:")
    print("  base        ->", p_base)
    print("  turnout     ->", p_turn)
    print("  features EV ->", p_fev)
    print("  features R  ->", p_fr)
    print("  polls       ->", p_polls)

    # Load artifacts
    base = pd.read_parquet(p_base)
    fev = pd.read_parquet(p_fev)
    fr = pd.read_parquet(p_fr)
    turn = pd.read_parquet(p_turn)

    # Expect long-form turnout
    need_turn = {"ed_uid", "scenario_tag", "Ti", "wEV", "wR"}
    if not need_turn.issubset(turn.columns):
        raise KeyError(
            f"Turnout must have columns {need_turn}; found {set(turn.columns)}"
        )

    # Key checks for features
    for df_ in (fev, fr):
        assert set(base["ed_uid"]) == set(
            df_["ed_uid"]
        ), "Key mismatch among model inputs."

    # Merge static features once
    df0 = base.merge(fev, on="ed_uid").merge(fr, on="ed_uid", suffixes=("_EV", "_R"))

    # Priors (finite by construction)
    eps = 1e-6
    mam = (
        pd.to_numeric(df0["pri_mam_share"], errors="coerce")
        .fillna(0.5)
        .clip(eps, 1 - eps)
    )
    sli = (
        pd.to_numeric(df0["sliwa21_share_eb"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    o_M = logit(mam)
    o_S = float(cfg.get("sliwa_alpha", 0.5)) * logit(np.clip(sli, eps, 1 - eps))
    o_C = np.zeros_like(o_M)
    base_logits = np.vstack([o_M, o_C, o_S]).T  # [n,3]

    # EV vs R contrast (S up, C down; M neutral)
    sc_see_saw = float(cfg.get("sc_see_saw", 0.9))
    turnout_elasticity = float(cfg.get("turnout_elasticity", 1.2))
    b_vec = turnout_elasticity * np.array([0.0, -sc_see_saw, +sc_see_saw]).reshape(1, 3)
    logits_EV = base_logits + b_vec
    logits_R = base_logits - b_vec

    # Polls (normalize; Eric Adams → Cuomo)
    poll_row = load_poll_strict(p_polls)
    print("✅ Poll target:", poll_row.to_dict())

    # Choose reference scenario from the actual tags
    tags = sorted(map(str, turn["scenario_tag"].dropna().unique().tolist()))
    if not tags:
        raise ValueError("No scenario_tag values found in turnout.")

    T_ref = cfg.get("scenarios", {}).get("T_ref", None)
    rho_ref = cfg.get("scenarios", {}).get("rho_ref", None)
    ref_tag = pick_ref_tag_from_available(tags, T_ref, rho_ref)
    if (T_ref is not None and rho_ref is not None) and ref_tag not in tags:
        print(
            f"⚠️ Desired ref (T_ref={T_ref}, rho_ref={rho_ref}) not found; using nearest: {ref_tag}"
        )
    print(f"🎯 Calibrating once at reference scenario: {ref_tag}")

    # Reference mixture for gamma calibration
    ref = turn[turn["scenario_tag"] == ref_tag][["ed_uid", "Ti", "wEV", "wR"]]
    df_ref = df0.merge(ref, on="ed_uid", how="left")

    P_EV_pre = softmax(logits_EV)
    P_R_pre = softmax(logits_R)
    P_mix_ref = pd.DataFrame(
        {
            "p_M": df_ref["wEV"].fillna(0).values * P_EV_pre[:, 0]
            + df_ref["wR"].fillna(0).values * P_R_pre[:, 0],
            "p_C": df_ref["wEV"].fillna(0).values * P_EV_pre[:, 1]
            + df_ref["wR"].fillna(0).values * P_R_pre[:, 1],
            "p_S": df_ref["wEV"].fillna(0).values * P_EV_pre[:, 2]
            + df_ref["wR"].fillna(0).values * P_R_pre[:, 2],
        }
    )
    gamma_ref = shift_to_poll_means(P_mix_ref, df_ref["Ti"].fillna(0.0), poll_row)
    print(f"   → gamma_ref = {gamma_ref}")

    # Optional C/S anti-coupling during calibration
    couple_sc = float(cfg.get("couple_sc", 0.85))  # 0=off, 1=perfect anti-coupling
    if couple_sc > 0:
        gM, gC, gS = gamma_ref["M"], gamma_ref["C"], gamma_ref["S"]
        a = 0.5 * (gC + gS)
        d = 0.5 * (gS - gC)
        a_new = (1.0 - couple_sc) * a
        gamma_ref = {"M": gM, "C": a_new - d, "S": a_new + d}

    gamma_vec = np.array([gamma_ref["M"], gamma_ref["C"], gamma_ref["S"]]).reshape(1, 3)

    # Sliwa floor enforcement only where Sliwa>50% in 2021
    sliwa_floor = float(cfg.get("sliwa_floor", 0.55))
    majority_mask = sli.values > 0.5
    print(f"🧭 Sliwa strongholds: {int(majority_mask.sum())} EDs (>50% in 2021)")

    def apply_sliwa_floor(L: np.ndarray):
        if not majority_mask.any():
            return
        idx = np.where(majority_mask)[0]
        frac = 0.50
        tiny = 1e-9
        target_S = np.maximum(sli.values, sliwa_floor).clip(0.501, 0.98)
        A = np.exp(L[idx, 0])
        B = np.exp(L[idx, 1])
        S = np.exp(L[idx, 2])
        t = target_S[idx]
        S_star = (t / (1 - t)) * (A + B)
        L[idx, 2] += frac * (
            np.log(np.clip(S_star, tiny, None)) - np.log(np.clip(S, tiny, None))
        )

    # Compute per-scenario
    out_rows = []
    for tag in tags:
        sub = turn[turn["scenario_tag"] == tag][["ed_uid", "Ti", "wEV", "wR"]]
        dfi = df0.merge(sub, on="ed_uid", how="left")

        Ti = pd.to_numeric(dfi["Ti"], errors="coerce").fillna(0.0).astype(float)
        wEV = pd.to_numeric(dfi["wEV"], errors="coerce").fillna(0.0).astype(float)
        wR = pd.to_numeric(dfi["wR"], errors="coerce").fillna(0.0).astype(float)

        logits_EV_shift = (logits_EV + gamma_vec).copy()
        logits_R_shift = (logits_R + gamma_vec).copy()
        apply_sliwa_floor(logits_EV_shift)
        apply_sliwa_floor(logits_R_shift)

        P_EV = softmax(logits_EV_shift)
        P_R = softmax(logits_R_shift)

        P_final = pd.DataFrame(
            {
                "ed_uid": dfi["ed_uid"],
                "p_M": wEV * P_EV[:, 0] + wR * P_R[:, 0],
                "p_C": wEV * P_EV[:, 1] + wR * P_R[:, 1],
                "p_S": wEV * P_EV[:, 2] + wR * P_R[:, 2],
                "Ti": Ti,
                "wEV": wEV,
                "wR": wR,
                "scenario_tag": tag,
            }
        ).merge(base[["ed_uid", "borough", "AD", "ED"]], on="ed_uid", how="left")

        out_rows.append(P_final)

    all_ed = pd.concat(out_rows, ignore_index=True)

    # Save + aggregates
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    ed_out = out_dir / "forecast_ed_by_scenario.csv"
    all_ed.to_csv(ed_out, index=False)

    # Normalization before agg
    cols_num = ["p_M", "p_C", "p_S", "Ti", "wEV", "wR"]
    all_ed[cols_num] = all_ed[cols_num].apply(pd.to_numeric, errors="coerce")
    all_ed[["p_M", "p_C", "p_S"]] = all_ed[["p_M", "p_C", "p_S"]].clip(0.0, 1.0)
    all_ed["Ti"] = all_ed["Ti"].fillna(0.0)

    def agg(df_in: pd.DataFrame, by):
        keep = by + ["p_M", "p_C", "p_S", "Ti"]
        df = df_in[keep].copy()

        def _safe_weighted_mean(g: pd.DataFrame) -> pd.Series:
            Ti = g["Ti"].fillna(0.0).to_numpy(float)
            P = g[["p_M", "p_C", "p_S"]].to_numpy(float)
            wsum = np.nansum(Ti)
            if not np.isfinite(wsum) or wsum <= 0.0:
                m = np.nanmean(P, axis=0)
                Ttot = 0.0
            else:
                m = np.average(P, weights=Ti, axis=0)
                Ttot = float(wsum)
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
            return pd.Series({"p_M": m[0], "p_C": m[1], "p_S": m[2], "T": Ttot})

        try:
            res = df.groupby(by, dropna=False).apply(
                _safe_weighted_mean, include_groups=False
            )
        except TypeError:
            res = df.groupby(by, dropna=False).apply(_safe_weighted_mean)
        return res.reset_index()

    agg(all_ed, ["scenario_tag", "borough", "AD"]).to_csv(
        out_dir / "forecast_ad_by_scenario.csv", index=False
    )
    agg(all_ed, ["scenario_tag", "borough"]).to_csv(
        out_dir / "forecast_borough_by_scenario.csv", index=False
    )
    agg(all_ed, ["scenario_tag"]).to_csv(
        out_dir / "forecast_city_by_scenario.csv", index=False
    )

    print("✅ Wrote:")
    print(" -", ed_out)
    print(" -", out_dir / "forecast_ad_by_scenario.csv")
    print(" -", out_dir / "forecast_borough_by_scenario.csv")
    print(" -", out_dir / "forecast_city_by_scenario.csv")


if __name__ == "__main__":
    main()
