#!/usr/bin/env python3
import pandas as pd, geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.patches as mpatches
from matplotlib import cm
from pathlib import Path
from shapely.geometry import mapping
import argparse
import folium
import html

# ---------- CONFIG ----------
ED_FORECAST_CSV = "./out/forecast_ed_by_scenario.csv"
ED_SHAPES_PATH = "./data/geo/nyc_ed.geojson"
PARKS_PATH = "./data/geo/nyc_parks.geojson"  # NYC Open Data Parks Properties
SCENARIO = None  # e.g. "T1800000_r04"; if None => render ALL scenarios
OUT_DIR = Path("./map")
OUT_BASENAME = "results"  # file → ed_map_<tag>.png

# --------------------------------
borough_map_long = {
    "BK": "Brooklyn",
    "QN": "Queens",
    "MN": "Manhattan",
    "SI": "Staten Island",  # fix typo
    "BX": "Bronx",
}
inverse_borough_map_long = {
    "Brooklyn": "BK",
    "Queens": "QN",
    "Manhattan": "MN",
    "Staten Island": "SI",  # fix typo
    "Bronx": "BX",
}


def inject_tooltip_css(m):
    css = """
  <style>
  /* === Container === */
  .leaflet-tooltip.nyt-tooltip{
    padding:12px 14px; border:0; border-radius:12px;
    background:rgba(255,255,255,0.96) !important;
    box-shadow:0 10px 28px rgba(0,0,0,.18);
    color:#111827; z-index:99999;
    max-width:320px;                 /* cap width on desktop/tablet */
  }
  .leaflet-tooltip.nyt-tooltip .leaflet-tooltip-content{ margin:0; overflow:hidden; }

  /* === Base table Folium emits when labels=true === */
  .nyt-tooltip table{
    border-collapse:collapse;
    font:600 13px/1.25 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  }
  .nyt-tooltip .hdr-nei { font-weight:700; }                  /* “Greenpoint” */
  .nyt-tooltip .hdr-ed  { color:#9ca3af; font-weight:600; }   /* “Brooklyn 50-010” */

  /* Row 1: Title line */
  .nyt-tooltip tbody tr:nth-child(1) th{ display:none; }
  .nyt-tooltip tbody tr:nth-child(1) td{
    display:flex; align-items:baseline; justify-content:flex-start;
    gap:10px; padding-left:0px; width:100%; text-align:left;
  }

  /* Row 2: subhead “Candidate | Pct.” */
  .nyt-tooltip tbody tr:nth-child(2) th,
  .nyt-tooltip tbody tr:nth-child(2) td{
    color:#9ca3af; font-weight:600;  /* gray-400 */
  }

  /* Dividers before candidate rows (rows 3–5) */
  .nyt-tooltip tbody tr:nth-child(n+3){ border-top:1px solid #e5e7eb; } /* gray-200 */

  /* We render our own content in TD for rows 3–5; hide TH */
  .nyt-tooltip tbody tr:nth-child(n+3) th { display:none; }
  .nyt-tooltip tbody tr:nth-child(n+3) td { color:#111827; font-weight:600; padding:6px 0; }

  /* === Ordered rows (custom HTML injected via row1_html/row2_html/row3_html) === */
  .nyt-tooltip .cand-row{
    display:flex; align-items:center; justify-content:flex-start; gap:10px; /* left-align name */
  }
  .nyt-tooltip .dot{
    width:8px; height:8px; border-radius:999px; display:inline-block;
    margin-right:6px; flex:0 0 8px;
  }
  .nyt-tooltip .cand-name{
    color:#6b7280; font-weight:500; text-align:left; flex:0 1 auto;          /* light gray, not bold */
  }
  .nyt-tooltip .cand-pct{
    color:#111827; font-weight:600; text-align:right; margin-left:auto;      /* % pinned right */
    min-width:3.5em;
  }

  /* Optional: if you *don’t* inline a color on the dot, these are backups */
  .nyt-tooltip .dot-m{ background:#3182bd; }  /* Mamdani (blue) */
  .nyt-tooltip .dot-c{ background:#31a354; }  /* Cuomo (green) */
  .nyt-tooltip .dot-s{ background:#de2d26; }  /* Sliwa (red)  */

  /* --- Mobile: keep tooltip narrower and a bit tighter --- */
  @media (max-width: 480px){
    .leaflet-tooltip.nyt-tooltip{ max-width:240px; padding:10px 12px; }
    .nyt-tooltip table{ font-size:12px; }
    .nyt-tooltip .cand-row{ gap:8px; }
    .nyt-tooltip .cand-pct{ min-width:3.2em; }
  }
</style>

    """
    m.get_root().header.add_child(folium.Element(css))


def _sorted_rows_for_tooltip(df):
    """
    Given df with p_M, p_C, p_S, adds row1_html,row2_html,row3_html ordered by share desc.
    Also returns which candidate is 1st/2nd/3rd if you need it later.
    """
    label_meta = np.array(
        [
            ("Mamdani", "m", "#3182bd", "p_M"),
            ("Cuomo", "c", "#31a354", "p_C"),
            ("Sliwa", "s", "#de2d26", "p_S"),
        ],
        dtype=object,
    )

    def make_rows(r):
        vals = [
            (
                label_meta[i][0],
                label_meta[i][1],
                label_meta[i][2],
                float(r[label_meta[i][3]]),
            )
            for i in range(3)
        ]
        # sort by prob desc
        vals.sort(key=lambda x: x[3], reverse=True)
        rows = []
        for name, key, color, p in vals:
            pct = f"{p*100:.1f}%"
            # Left dot + name + right percentage (all inside td)
            rows.append(
                f'<div class="cand-row">'
                f'<span class="dot dot-{key}" style="background:{color}"></span>'
                f'<span class="cand-name">{name}</span>'
                f'<span class="cand-pct">{pct}</span>'
                f"</div>"
            )
        return pd.Series(
            {
                "row1_html": rows[0],
                "row2_html": rows[1],
                "row3_html": rows[2],
                "rank1": vals[0][0],
                "rank2": vals[1][0],
                "rank3": vals[2][0],
            }
        )

    add = df.apply(make_rows, axis=1)
    for c in ["row1_html", "row2_html", "row3_html", "rank1", "rank2", "rank3"]:
        df[c] = add[c]
    return df


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

def validate_parks(parks):
    if parks is not None and not parks.empty:
        try:
            if hasattr(parks.geometry, "is_valid"):
                bad = ~parks.geometry.is_valid
                if bad.any():
                    parks.loc[bad, "geometry"] = parks.loc[bad, "geometry"].buffer(0)
            parks = parks[parks.geometry.notnull()].copy()
            p_features = [
                {"type": "Feature", "geometry": mapping(geom), "properties": {}}
                for geom in parks.geometry
            ]
            parks = {"type": "FeatureCollection", "features": p_features}
            return parks
        except Exception as e:
            print(f"⚠️ Could not add parks overlay: {e}")

def park_style(_):
            return {
                "fillColor": "#cfcfcf",
                "color": "#cfcfcf",
                "weight": 0,
                "fillOpacity": 1.0,
            }


def render_one(mdf, shapes, parks, tag, static=True):
    # winners/margins/runner per ED
    mdf = mdf.copy()
    mdf["ED_key"] = parse_uid(mdf["ed_uid"])
    mdf["winner"], mdf["margin"], mdf["runner_idx"] = winner_and_margin(mdf)

    # merge with polygons
    geo = shapes.merge(
        mdf[
            [
                "ED",
                "AD",
                "ED_key",
                "borough",
                "margin",
                "winner",
                "runner_idx",
                "p_M",
                "p_C",
                "p_S",
            ]
        ],
        on="ED_key",
        how="left",
    )

    # color assignment
    def row_color(r):
        if pd.isna(r["winner"]):  # no forecast: light gray
            return "#cfcfcf"
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
            "ED",
            "AD",
            "ED_key",
            "p_M",
            "p_C",
            "p_S",
            "p_M_pct",
            "p_C_pct",
            "p_S_pct",
            "borough",
            "neighborhood",
            "fill",
            "geometry",
        ]
        keep = [c for c in keep if c in geo.columns]
        g = geo[keep].copy()

        # Style function uses the precomputed hex color in "fill"
        def style_fn(feat):
            return {
                "fillColor": feat["properties"].get("fill", "#cfcfcf"),
                "color": "white",
                "weight": 0.2,
                "fillOpacity": 0.85,
            }

        # Build map
        m = folium.Map(location=[40.71, -73.94], zoom_start=10, tiles="cartodbpositron")

        # ED polygons first so parks can sit on top
        borough_map = {
            "BK": "Brooklyn",
            "QN": "Queens",
            "MN": "Manhattan",
            "SI": "Statan Island",
            "BX": "Bronx",
        }

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

        s = g["ED_key"].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
        g["ED_key"] = g["borough"].map(borough_map) + " " + s.str[:2] + "-" + s.str[-3:]

        b_map = norm_keys(pd.read_csv("./data/ed_manifest/ed_borough_map.csv"))
        g = norm_keys(g)
        g = g.merge(b_map[["neighborhood", "ED", "AD"]], how="left", on=["ED", "AD"])

        # Add a fake header row: "Candidate"  |  "Pct."
        g["hdr_pct"] = "Pct."  # constant text for the subheader row

        # Build map
        m = folium.Map(location=[40.71, -73.94], zoom_start=10, tiles="cartodbpositron")
        inject_tooltip_css(m)

        # build g
        # Pretty “Borough AD-ED” label

        # Build a separate display label; do NOT overwrite ED_key
        g["ed_label"] = (
            g["borough"].map(borough_map_long) + " " + g["AD"] + "-" + g["ED"]
        )

        # Later in the tooltip:
        g["hdr_pct"] = "Pct."
        g["header_html"] = g.apply(
            lambda r: (
                f'<span class="hdr-nei">{html.escape(str(r.get("neighborhood") or r.get("borough") or ""))}</span>'
                f'<span class="hdr-ed">{html.escape(str(r.get("ed_label") or ""))}</span>'
            ),
            axis=1,
        )

        fields  = ["header_html", "hdr_pct", "row1_html", "row2_html", "row3_html"]
        aliases = ["", "Candidate", "", "", ""]  # names live inside the HTML now
        g = _sorted_rows_for_tooltip(g)
        gj = folium.GeoJson(
            g,
            name=f"EDs ({tag})",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=fields,
                aliases=aliases,
                labels=True,
                sticky=False,
                class_name="nyt-tooltip",
            ),
            highlight_function=lambda feat: {
                "weight": 0.2,
                "color": "white",
                "fillOpacity": 0.85,
            },
        )
        gj.add_to(m)

        # Parks overlay
        parks = validate_parks(parks)

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
        return geo
    # ========= END INTERACTIVE BRANCH =========

    # ========= STATIC BRANCH =========

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    geo.plot(ax=ax, color=geo["fill"], edgecolor="#ffffff", linewidth=0.05, zorder=1)
    ax.set_axis_off()
    if parks is not None and not parks.empty:
        parks.plot(ax=ax, color="#cfcfcf", edgecolor="none", zorder=10)
    legend_patches = [
        mpatches.Patch(color=pal_mamdani(0.8), label="Mamdani"),
        mpatches.Patch(color=pal_cuomo(0.8), label="Cuomo"),
        mpatches.Patch(color=pal_sliwa(0.8), label="Sliwa"),
        mpatches.Patch(color="#cfcfcf", label="No data"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
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
    plt.savefig(out_png, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    print(f"✅ Saved {out_png}")
    return geo

# ---------- BOROUGH MAP + PNGs ----------
# ---------- Borough interactive directly from CSV ----------

BORO_LONG = {
    "BK": "Brooklyn",
    "QN": "Queens",
    "MN": "Manhattan",
    "BX": "Bronx",
    "SI": "Staten Island",
}


def _winner_margin_from_probs(df):
    P = df[["p_M", "p_C", "p_S"]].astype(float).to_numpy()
    top = P.argmax(axis=1)
    names = np.array(["Mamdani", "Cuomo", "Sliwa"])[top]
    srt = np.sort(P, axis=1)
    margin = (srt[:, -1] - srt[:, -2]).astype(float)
    return names, margin


def _pick_color(row):
    richness = float(np.clip(0.30 + 0.70 * row["margin"], 0.30, 1.0))
    return color_for(row["winner"], richness)

def _borough_shapes_from_ED(
    ED_shapes_path=ED_SHAPES_PATH, ed_map_path="./data/ed_manifest/ed_borough_map.csv"
):
    """
    Build borough polygons by dissolving ED shapes, using ed_borough_map to tag each ED with a borough code.
    Returns GeoDataFrame with columns ['borough','geometry'] in EPSG:4326.
    """
    g = gpd.read_file(ED_shapes_path).copy()
    if "elect_dist" not in g.columns:
        raise ValueError("ED shapefile must have 'elect_dist' column (AD2+ED3).")
    # Parse AD/ED from elect_dist
    s = g["elect_dist"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    g["AD"] = s.str[:2]
    g["ED"] = s.str[-3:]

    # Map AD/ED → borough code via your manifest
    bmap = pd.read_csv(ed_map_path)
    for c in ("AD", "ED"):
        bmap[c] = pd.to_numeric(bmap[c], errors="coerce").astype("Int64").astype(str)
    bmap["AD"] = bmap["AD"].str.zfill(2)
    bmap["ED"] = bmap["ED"].str.zfill(3)
    # Normalize borough col name(s)
    boro_col = "borough"
    if "borough_code" in bmap.columns and boro_col not in bmap.columns:
        bmap.rename(columns={"borough_code": boro_col}, inplace=True)

    g = g.merge(bmap[["AD", "ED", boro_col]], on=["AD", "ED"], how="left")
    if g[boro_col].isna().any():
        missing_n = int(g[boro_col].isna().sum())
        print(
            f"⚠️ {missing_n} ED polygons lack borough code; they will be dropped for dissolve."
        )
        g = g[g[boro_col].notna()].copy()

    # Dissolve to borough
    boros = g.dissolve(by=boro_col, as_index=False, aggfunc="first")[
        ["borough", "geometry"]
    ]

    # CRS → WGS84
    if boros.crs is None or boros.crs.to_epsg() != 4326:
        boros = boros.to_crs(epsg=4326)
    return boros


def make_borough_interactive(
    scenario_tag=None,
    borough_csv_path="./out/forecast_borough_by_scenario.csv",
    out_dir=OUT_DIR,
    out_basename=OUT_BASENAME,
):
    """
    Build a borough-only interactive HTML using the borough CSV.
    - scenario_tag: if None, uses the first scenario in the file (sorted)
    Output: <out_dir>/<out_basename>_boroughs_<tag>.html
    """
    # 1) Load data
    df = pd.read_csv(borough_csv_path)

    tags = sorted(df["scenario_tag"].dropna().unique().tolist())
    if not tags:
        raise ValueError("No scenario_tag values in borough CSV.")
    tag = scenario_tag or tags[0]
    bdf = df[df["scenario_tag"] == tag].copy()

    if bdf.empty:
        raise ValueError(f"No rows for scenario_tag={tag} in borough CSV.")

    # 2) Shapes: dissolve EDs → borough polygons
    boros_geo = _borough_shapes_from_ED()  # uses ED_SHAPES_PATH + ed_borough_map
    boros_geo["borough"] = boros_geo["borough"].map(inverse_borough_map_long)

    # 3) Join data to shapes
    geo = boros_geo.merge(
        bdf[["borough", "p_M", "p_C", "p_S"]], on="borough", how="left"
    )
    if geo[["p_M", "p_C", "p_S"]].isna().any().any():
        print("⚠️ Some borough(s) missing p_M/p_C/p_S; defaulting to gray.")

    # 4) Winner/margin + colors
    geo["p_M"] = pd.to_numeric(geo["p_M"], errors="coerce")
    geo["p_C"] = pd.to_numeric(geo["p_C"], errors="coerce")
    geo["p_S"] = pd.to_numeric(geo["p_S"], errors="coerce")
    geo["winner"], geo["margin"] = _winner_margin_from_probs(geo.fillna(0.0))
    geo["fill"] = geo.apply(
        lambda r: "#cfcfcf" if pd.isna(r["p_M"]) else _pick_color(r), axis=1
    )

    # 5) Tooltip fields (reuse NYT CSS)
    geo["name"] = geo["borough"].map(BORO_LONG).fillna(geo["borough"])
    geo["p_M_pct"] = (geo["p_M"] * 100).round(1).astype("Float64").astype(str) + "%"
    geo["p_C_pct"] = (geo["p_C"] * 100).round(1).astype("Float64").astype(str) + "%"
    geo["p_S_pct"] = (geo["p_S"] * 100).round(1).astype("Float64").astype(str) + "%"
    geo["hdr_pct"] = "Pct."
    geo["header_html"] = geo.apply(
        lambda r: f'<span class="hdr-nei">{html.escape(str(r["name"]))}</span>'
        f'<span class="hdr-ed">Borough-wide</span>',
        axis=1,
    )

    # 6) Build map
    m = folium.Map(location=[40.71, -73.94], zoom_start=10, tiles="cartodbpositron")
    inject_tooltip_css(m)

    def style_fn(feat):
        return {
            "fillColor": feat["properties"].get("fill", "#cfcfcf"),
            "color": "white",
            "weight": 1.0,
            "fillOpacity": 0.85,
        }

    fields  = ["header_html", "hdr_pct", "row1_html", "row2_html", "row3_html"]
    aliases = ["", "Candidate", "", "", ""]  # names live inside the HTML now

    geo = _sorted_rows_for_tooltip(geo)
    folium.GeoJson(
        data=geo[
            [
                "header_html",
                "hdr_pct",
                "row1_html",
                "row2_html",
                "row3_html",
                "fill",
                "geometry",
            ]
        ],
        name=f"Boroughs ({tag})",
        style_function=style_fn,
        highlight_function=lambda f: {
            "weight": 2.5,
            "color": "#111",
            "fillOpacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            labels=True,
            sticky=False,
            class_name="nyt-tooltip",
            localize=True
        ),
    ).add_to(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"{out_basename}_boroughs_{tag}.html"
    m.save(out_html)
    print(f"✅ Saved {out_html}")


def save_static_pngs_by_borough(
    geo_ed, parks, tag, out_dir=OUT_DIR, out_basename=OUT_BASENAME
):
    """
    Makes *one PNG per borough*, showing ED-level detail within that borough only.
    Output: <out_dir>/<out_basename>_<BORO>_{tag}.png
    """
    g = geo_ed.copy()
    for b in ["MN", "BK", "QN", "BX", "SI"]:
        sub = g[g["borough"] == b].copy()
        if sub.empty:
            print(f"⚠️ No EDs for borough {b}; skipping.")
            continue
        # Optional: clip parks to borough bbox for speed
        parks_sub = None
        if parks is not None and not parks.empty:
            try:
                parks_sub = parks[parks.geometry.intersects(sub.unary_union)].copy()
            except Exception:
                parks_sub = parks

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        sub.plot(
            ax=ax, color=sub["fill"], edgecolor="#ffffff", linewidth=0.08, zorder=1
        )
        if parks_sub is not None and not parks_sub.empty:
            parks_sub.plot(ax=ax, color="#cfcfcf", edgecolor="none", zorder=10)

        ax.set_axis_off()

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{out_basename}_{b}_{tag}.png"
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight", dpi=300, transparent = True)
        plt.close(fig)
        print(f"🖼️ Saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
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
        geo_ed = render_one(mdf, shapes, parks, tag, static)
        # 1) Borough-level interactive HTML (dissolved polygons & borough-only tooltip)
        make_borough_interactive(tag)

        # 2) One PNG per borough (ED detail, cropped to borough)
        save_static_pngs_by_borough(geo_ed, parks, tag)


if __name__ == "__main__":
    main()
