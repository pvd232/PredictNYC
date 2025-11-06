#!/usr/bin/env python3
import pandas as pd, geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.patches as mpatches
from matplotlib import cm
from pathlib import Path
from shapely.geometry import mapping
import argparse
import folium

# ---------- CONFIG ----------
ED_FORECAST_CSV = "./out/forecast_ed_by_scenario.csv"
ED_SHAPES_PATH = "./map/nyc_ed.geojson"
PARKS_PATH = "./map/nyc_parks.geojson"  # NYC Open Data Parks Properties
SCENARIO = None  # e.g. "T1800000_r04"; if None => render ALL scenarios
OUT_DIR = Path("./map")
OUT_BASENAME = "results"  # file → ed_map_<tag>.png
# --------------------------------


# --- helpers ---
def parse_uid(series):
    s = series.astype(str)
    ad = s.str.extract(r"AD(\d+)")[0].fillna("00").astype(str).str.zfill(2)
    ed = s.str.extract(r"ED(\d+)")[0].fillna("000").astype(str).str.zfill(3)
    return ad + ed


def ensure_shape_key(g):
    g = g.copy()
    if "elect_dist" not in g.columns:
        raise ValueError("Could not find 'elect_dist' column in shapefile.")
    g["ED_key"] = g["elect_dist"].astype(str).str.strip()
    return g[["ED_key", "geometry"]]


def winner_and_margin(df):
    P = df[["p_M", "p_C", "p_S"]].astype(float).to_numpy()
    top_idx = P.argmax(axis=1)
    names = np.array(["Mamdani", "Cuomo", "Sliwa"])[top_idx]

    sortedP = np.sort(P, axis=1)
    margin = (sortedP[:, -1] - sortedP[:, -2]).astype(float)

    # runner-up index: zero out winner, then argmax
    P_ru = P.copy()
    P_ru[np.arange(len(P_ru)), top_idx] = -1
    runner_idx = P_ru.argmax(axis=1)
    return names, margin, runner_idx


def pal_mamdani(t):
    return to_hex(cm.Blues(0.30 + 0.70 * t))


def pal_cuomo(t):
    return to_hex(cm.Greens(0.30 + 0.70 * t))  # Progressive/1912 vibe


def pal_sliwa(t):
    return to_hex(cm.Reds(0.30 + 0.70 * t))


def color_for(label, strength):
    t = float(np.clip(strength, 0.0, 1.0))  # already scaled outside
    if label == "Mamdani":
        return pal_mamdani(t)
    if label == "Cuomo":
        return pal_cuomo(t)
    if label == "Sliwa":
        return pal_sliwa(t)
    return "#9aa0a6"


def blend_hex(c1, c2, w=0.5):
    """Linear blend of two #RRGGBB colors with weight w for c1."""
    w = float(np.clip(w, 0.0, 1.0))
    a = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i : i + 2], 16) for i in (1, 3, 5))
    m = tuple(int(round(w * a[i] + (1 - w) * b[i])) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*m)


def label_from_idx(idx):
    return np.array(["Mamdani", "Cuomo", "Sliwa"])[idx]


def render_one(mdf, shapes, parks, tag, static=True):
    # winners/margins/runner per ED
    mdf = mdf.copy()
    mdf["ED_key"] = parse_uid(mdf["ed_uid"])
    mdf["winner"], mdf["margin"], mdf["runner_idx"] = winner_and_margin(mdf)

    # merge with polygons 
    geo = shapes.merge(
        mdf[["ED_key", "borough", "margin", "runner_idx", "p_M", "p_C", "p_S"]],
        on="ED_key",
        how="left",
    )

    # color assignment
    def row_color(r):
        if pd.isna(r["winner"]):  # no forecast: light gray
            return "#bdbdbd"
        richness = float(np.clip(0.30 + 0.70 * r["margin"], 0.30, 1.0))
        return color_for(r["winner"], richness)

    geo["fill"] = geo.apply(row_color, axis=1)

    # ========= INTERACTIVE BRANCH =========
    if not static:
        # Ensure WGS84 for Leaflet
        if geo.crs is None or geo.crs.to_epsg() != 4326:
            geo = geo.to_crs(epsg=4326)
        if parks is not None and not parks.empty and parks.crs != geo.crs:
            parks = parks.to_crs(geo.crs)

        # Prepare safe tooltip fields
        for c in ["p_M", "p_C", "p_S", "margin"]:
            if c in geo.columns:
                geo[c] = pd.to_numeric(geo[c], errors="coerce").astype(float)
        for c in ["p_M", "p_C", "p_S"]:
            if c in geo.columns:
                geo[c + "_pct"] = (geo[c] * 100).round(1).astype("Float64").astype(
                    str
                ) + "%"

        # Minimal properties to avoid JSON serialization issues
        keep = [
            "ED_key",
            "p_M_pct",
            "p_C_pct",
            "p_S_pct",
            "borough", 
            "fill",
            "geometry",
        ]
        keep = [c for c in keep if c in geo.columns]
        g = geo[keep].copy()

        # Style function uses the precomputed hex color in "fill"
        def style_fn(feat):
            return {
                "fillColor": feat["properties"].get("fill", "#bdbdbd"),
                "color": "white",
                "weight": 0.2,
                "fillOpacity": 0.85,
            }

        # Build map
        m = folium.Map(location=[40.71, -73.94], zoom_start=10, tiles="cartodbpositron")

        # ED polygons first so parks can sit on top
        fields = [
            c
            for c in ["ED_key", "p_M_pct", "p_C_pct", "p_S_pct"]
            if c in g.columns
        ]
        aliases = ["ED",   "Mamdani", "Cuomo", "Sliwa"][
            : len(fields)
        ]
        borough_map = {
            "BK": "Brooklyn",
            "QN": "Queens",
            "MN": "Manhattan",
            "SI": "Statan Island",
            "BX": "Bronx",
        }
        s = g["ED_key"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
        g["ED_key"] = g["borough"].map(borough_map) + " " + s.str[:2] + "-" + s.str[-3:]

        folium.GeoJson(
            g,
            name=f"EDs ({tag})",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=fields, aliases=aliases, sticky=False, labels=True
            ),
            highlight_function=None,
            zoom_on_click=False,
        ).add_to(m)

        # Parks overlay
        if parks is not None and not parks.empty:
            try:
                if hasattr(parks.geometry, "is_valid"):
                    bad = ~parks.geometry.is_valid
                    if bad.any():
                        parks.loc[bad, "geometry"] = parks.loc[bad, "geometry"].buffer(0)
                parks = parks[parks.geometry.notnull()].copy()
                p_features = [{"type":"Feature","geometry":mapping(geom),"properties":{}} for geom in parks.geometry]
                parks = {"type":"FeatureCollection","features":p_features}
            except Exception as e:
                print(f"⚠️ Could not add parks overlay: {e}")
            def park_style(_):
                return {
                    "fillColor": "#cfcfcf",
                    "color": "#cfcfcf",
                    "weight": 0,
                    "fillOpacity": 1.0,
                }

            folium.GeoJson(parks, style_function=park_style, name="Parks").add_to(m)

        # Small, fixed legend (top-right)
        legend_html = f"""
        <div style="position: fixed; top: 12px; right: 12px; z-index: 9999;
             background: rgba(255,255,255,0.92); padding: 10px 12px; border-radius: 8px;
             box-shadow: 0 1px 6px rgba(0,0,0,0.2); font: 12px/1.2 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
          <div style="font-weight:600; margin-bottom:6px;">Leading candidate (shaded by margin)</div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <div style="width:84px; height:10px; border-radius:6px;
              background: linear-gradient(90deg, #eff3ff, #bdd7e7, #6baed6, #3182bd, #08519c);"></div>
            <span>Mamdani</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <div style="width:84px; height:10px; border-radius:6px;
              background: linear-gradient(90deg, #e5f5e0, #a1d99b, #74c476, #31a354, #006d2c);"></div>
            <span>Cuomo</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <div style="width:84px; height:10px; border-radius:6px;
              background: linear-gradient(90deg, #fee5d9, #fcae91, #fb6a4a, #de2d26, #a50f15);"></div>
            <span>Sliwa</span>
          </div>          
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_html = OUT_DIR / f"{OUT_BASENAME}_{tag}.html"
        m.save(out_html)
        print(f"✅ Saved {out_html}")
        return
    # ========= END INTERACTIVE BRANCH =========

    # ========= STATIC BRANCH =========

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    geo.plot(ax=ax, color=geo["fill"], edgecolor="#ffffff", linewidth=0.05, zorder=1)
    ax.set_axis_off()
    if parks is not None and not parks.empty:
        parks.plot(ax=ax, color="#cfcfcf", edgecolor="none", zorder=10)
    ax.set_title(
        f"NYC Mayoral Forecast — Leading Candidate by Precinct  ({tag})",
        pad=10,
        fontsize=13,
        weight="bold",
    )
    legend_patches = [
        mpatches.Patch(color=pal_mamdani(0.8), label="Mamdani"),
        mpatches.Patch(color=pal_cuomo(0.8), label="Cuomo"),
        mpatches.Patch(color=pal_sliwa(0.8), label="Sliwa"),
        mpatches.Patch(color="#cfcfcf", label="Park"),
        mpatches.Patch(color="#bdbdbd", label="No data"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        handlelength=1.8,
        handleheight=1.2,
        columnspacing=1.6,
        prop={"size": 10},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"{OUT_BASENAME}_{tag}.png"
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"✅ Saved {out_png}")


def main():
    ap = argparse.ArgumentParser(        )
    ap.add_argument("--static", default="True")
    args = ap.parse_args()
    static = args.static
    if static == "False":
        static = False
    else:
        static = True
    # load forecasts (all scenarios)
    f = pd.read_csv(ED_FORECAST_CSV)
    f.columns = [c.strip() for c in f.columns]
    if "scenario_tag" not in f.columns:
        raise KeyError("forecast CSV must include 'scenario_tag'.")

    # pick scenarios
    tags = (
        [SCENARIO] if SCENARIO else sorted(f["scenario_tag"].dropna().unique().tolist())
    )
    tags = tags[1:2]
    if not tags:
        raise ValueError("No scenarios found in forecast CSV.")

    # load shapes
    shapes = ensure_shape_key(gpd.read_file(ED_SHAPES_PATH))

    # parks (best-effort)
    parks = None
    try:
        parks_gdf = gpd.read_file(PARKS_PATH)
        if parks_gdf.crs != shapes.crs:
            parks_gdf = parks_gdf.to_crs(shapes.crs)
        # prefer a light filter for large/known parks, else just keep biggest by area
        name_cols = [c for c in parks_gdf.columns if "name" in c.lower()]
        if name_cols:
            ncol = name_cols[0]
            keep = {
                "CENTRAL PARK",
                "PROSPECT PARK",
                "FLUSHING MEADOWS CORONA PARK",
            }
            parks = parks_gdf[parks_gdf[ncol].str.upper().isin(keep)]
        else:
            parks_gdf["area"] = parks_gdf.geometry.area
            parks = parks_gdf.sort_values("area", ascending=False).head(200)
    except Exception as e:
        print(f"⚠️ Could not overlay parks: {e}")

    # render each scenario
    for tag in tags:
        mdf = f[f["scenario_tag"] == tag].copy()
        if mdf.empty:
            print(f"⚠️ No rows for scenario {tag}; skipping.")
            continue
        render_one(mdf, shapes, parks, tag, static)


if __name__ == "__main__":
    main()
