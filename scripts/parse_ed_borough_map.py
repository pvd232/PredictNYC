#!/usr/bin/env python3
# Fast local AD→ED mapper with robust backends (PyMuPDF → pdfminer → OCR optional)
""" 
python scripts/parse_ed_borough_map.py \
--pdf-dir ./data/ed_manifest/ad_pdfs \
--start 23 --end 87 \
--out complete_ed_borough_mapping.csv \
--workers 12 --max-pages 4 --strict-3-digit --ocr
"""
import re, sys, argparse, concurrent.futures as cf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

BORO_NAME_TO_CODE = {
    "NEW YORK": "MN",
    "KINGS": "BK",
    "BRONX": "BX",
    "QUEENS": "QN",
    "RICHMOND": "SI",
}
BORO_CODE_TO_NAME = {
    "MN": "Manhattan",
    "BK": "Brooklyn",
    "BX": "Bronx",
    "QN": "Queens",
    "SI": "Staten Island",
}


def _try_import_pymupdf():
    try:
        import fitz  # PyMuPDF

        if hasattr(fitz, "open"):
            return fitz
    except Exception:
        pass
    return None


def _try_import_pdfminer():
    try:
        from pdfminer.high_level import extract_text

        return extract_text
    except Exception:
        return None


def _try_import_ocr():
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

        return fitz, pytesseract, Image
    except Exception:
        return None, None, None


def infer_boroughs_from_text(text: str) -> List[str]:
    ups = (text or "").upper()
    hits = {code for name, code in BORO_NAME_TO_CODE.items() if name in ups}
    return sorted(hits)


def make_ed_pattern(ad: int, strict_3_digit: bool) -> re.Pattern:
    return (
        re.compile(rf"(?<!\d)(\d{{3}})\s*/\s*{ad}(?!\d)")
        if strict_3_digit
        else re.compile(rf"(?<!\d)(\d{{1,3}})\s*/\s*{ad}(?!\d)")
    )


def scrape_eds(text: str, pat: re.Pattern) -> List[int]:
    if not text:
        return []
    out = set()
    for m in pat.finditer(text):
        try:
            v = int(m.group(1))
            if 1 <= v <= 999:
                out.add(v)
        except Exception:
            pass
    return sorted(out)


def ed_uid(
    ad: int, ed: int, bcode: Optional[str] = None, flag: Optional[str] = None
) -> str:
    eds = f"{ed:03d}"
    return f"{(flag or (bcode or 'UNK')).upper()}-AD{ad}-ED{eds}"


def load_existing(old_csv: Optional[Path]) -> Dict[Tuple[int, int], Dict[str, str]]:
    if not old_csv or not old_csv.exists():
        return {}
    df = pd.read_csv(old_csv, dtype=str)
    need = {"AD", "ED", "borough_code", "borough_name", "ed_uid"}
    if not need.issubset(df.columns):
        if {"AD", "ED", "borough"}.issubset(df.columns):
            df = df.assign(
                borough_code=df["borough"].str.upper().str.strip(),
                borough_name=df["borough"].map(BORO_CODE_TO_NAME),
            )
            df["ed_uid"] = [
                ed_uid(int(a), int(e), b)
                for a, e, b in zip(df["AD"], df["ED"], df["borough_code"])
            ]
        else:
            raise ValueError(
                f"Existing mapping missing {need}. Found {list(df.columns)}"
            )
    df["AD"] = df["AD"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    df["ED"] = df["ED"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return {
        (int(r.AD), int(r.ED)): {
            "borough_code": (r.borough_code or "").upper().strip(),
            "borough_name": r.borough_name or "",
            "ed_uid": r.ed_uid or "",
        }
        for _, r in df.iterrows()
    }


# ---------- backends ----------
def extract_text_pymupdf(pdf_path: Path, max_pages: int) -> str:
    fitz = _try_import_pymupdf()
    if fitz is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        limit = min(max_pages, len(doc))
        out = []
        for i in range(limit):
            pg = doc[i]
            t = pg.get_text("text") or pg.get_text()
            if t:
                out.append(t)
        doc.close()
        return "\n".join(out)
    except Exception:
        return ""


def extract_text_pdfminer(pdf_path: Path, max_pages: int) -> str:
    extract_text = _try_import_pdfminer()
    if extract_text is None:
        return ""
    try:
        # pdfminer allows selecting page numbers (0-based)
        return extract_text(str(pdf_path), page_numbers=set(range(max_pages)))
    except Exception:
        return ""


def extract_text_ocr(pdf_path: Path, max_pages: int, dpi: int = 200) -> str:
    fitz, pytesseract, Image = _try_import_ocr()
    if not (fitz and pytesseract and Image):
        return ""
    try:
        doc = fitz.open(pdf_path)
        limit = min(max_pages, len(doc))
        out = []
        for i in range(limit):
            pix = doc[i].get_pixmap(dpi=dpi)
            import io

            img = Image.open(io.BytesIO(pix.tobytes("png")))
            out.append(pytesseract.image_to_string(img))
        doc.close()
        return "\n".join(out)
    except Exception:
        return ""


def process_one_ad(
    ad: int, pdf_dir: Path, max_pages: int, use_ocr: bool, strict_3: bool, engine: str
) -> Dict:
    pdf = pdf_dir / f"ad_{ad}.pdf"
    if not pdf.exists():
        return {
            "ad": ad,
            "ok": False,
            "reason": "missing_pdf",
            "eds": [],
            "boros": [],
            "pdf": str(pdf),
        }

    text = ""
    if engine in ("auto", "pymupdf"):
        text = extract_text_pymupdf(pdf, max_pages)
        if engine == "pymupdf":  # do not fall through
            pass
    if engine in ("auto", "pdfminer") and not text:
        text = extract_text_pdfminer(pdf, max_pages)
    if use_ocr and not text:
        text = extract_text_ocr(pdf, max_pages)

    pat = make_ed_pattern(ad, strict_3)
    eds = scrape_eds(text, pat)
    panel_text = extract_footer_text(
        pdf, rel=(0.00, 0.96, 0.15, 1.0), max_pages=2, use_ocr=use_ocr
    )
    boros = infer_boroughs_from_text(panel_text)
    return {"ad": ad, "ok": True, "eds": eds, "boros": boros, "pdf": str(pdf)}


def to_rows(results: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for ad, res in sorted(results.items()):
        eds, boros = res.get("eds", []), res.get("boros", [])
        if len(boros) == 1:
            bcode, flag = boros[0], None
        elif len(boros) == 0:
            bcode, flag = None, "UNK"
        else:
            bcode, flag = None, "MULTI"
        for ed in eds:
            rows.append(
                {
                    "AD": ad,
                    "ED": ed,
                    "borough_code": (bcode or ""),
                    "borough_name": BORO_CODE_TO_NAME.get(bcode or "", ""),
                    "ed_uid": ed_uid(ad, ed, bcode, flag),
                    "inferred_boroughs_from_pdf": ",".join(boros),
                }
            )
    return rows


def merge_with_existing(new_rows, existing):
    merged, conflicts = {}, []
    for (ad, ed), info in existing.items():
        merged[(ad, ed)] = {
            "AD": ad,
            "ED": ed,
            "borough_code": info["borough_code"],
            "borough_name": info["borough_name"],
            "ed_uid": info["ed_uid"],
            "inferred_boroughs_from_pdf": "",
        }
    for r in new_rows:
        key = (int(r["AD"]), int(r["ED"]))
        if key not in merged:
            merged[key] = r
            continue
        old, new = (merged[key]["borough_code"] or "").upper(), (
            r["borough_code"] or ""
        ).upper()
        if not old and new:
            merged[key]["borough_code"] = new
            merged[key]["borough_name"] = BORO_CODE_TO_NAME.get(new, "")
            if not merged[key]["ed_uid"]:
                merged[key]["ed_uid"] = ed_uid(*key, new)
        if not merged[key]["ed_uid"]:
            merged[key]["ed_uid"] = r["ed_uid"]
        if r.get("inferred_boroughs_from_pdf"):
            merged[key]["inferred_boroughs_from_pdf"] = r["inferred_boroughs_from_pdf"]
        if new and old and new != old:
            conflicts.append(
                {
                    "AD": key[0],
                    "ED": key[1],
                    "existing_borough_code": old,
                    "pdf_inferred_borough_code": new,
                }
            )
    out = [
        {
            "AD": k[0],
            "ED": k[1],
            "borough_code": v["borough_code"],
            "borough_name": v["borough_name"],
            "ed_uid": v["ed_uid"] or ed_uid(k[0], k[1], v["borough_code"] or None),
            "inferred_boroughs_from_pdf": v.get("inferred_boroughs_from_pdf", ""),
        }
        for k, v in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]
    return out, conflicts


def extract_footer_text(
    pdf_path: Path,
    rel=(0.02, 0.74, 0.38, 0.98),  # (lx, ty, rx, by) as fractions
    max_pages: int = 2,
    use_ocr: bool = False,
    dpi: int = 200,
) -> str:
    """
    rel = (left_x_frac, top_y_frac, right_x_frac, bottom_y_frac)
    e.g., (0.02, 0.74, 0.38, 0.98) ≈ left 2%..38% and bottom 26% of the page.
    """
    txt = ""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        for i in range(min(max_pages, len(doc))):
            pg = doc[i]
            R = pg.rect
            clip = fitz.Rect(
                R.x0 + rel[0] * R.width,
                R.y0 + rel[1] * R.height,
                R.x0 + rel[2] * R.width,
                R.y0 + rel[3] * R.height,
            )
            t = pg.get_text("text", clip=clip)
            if t:
                txt += ("\n" if txt else "") + t
        doc.close()
    except Exception:
        pass

    if txt or not use_ocr:
        return txt

    # OCR fallback on cropped render
    try:
        import fitz, io
        from PIL import Image
        import pytesseract

        doc = fitz.open(pdf_path)
        for i in range(min(max_pages, len(doc))):
            pg = doc[i]
            R = pg.rect
            clip = fitz.Rect(
                R.x0 + rel[0] * R.width,
                R.y0 + rel[1] * R.height,
                R.x0 + rel[2] * R.width,
                R.y0 + rel[3] * R.height,
            )
            pix = pg.get_pixmap(dpi=dpi, clip=clip)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            t = pytesseract.image_to_string(img)
            if t:
                txt += ("\n" if txt else "") + t
        doc.close()
    except Exception:
        pass
    return txt


def trim_cols(out):
    filename = out
    df = pd.read_csv(filename)
    df.drop(columns=["inferred_boroughs_from_pdf"], inplace=True)
    df.to_csv(out, index=False)


def build_ad_borough_map():
    df = pd.read_csv("../data/ed_manifest/ed_borough_map.csv")
    # Collapse to unique pairs of 'col1' and 'col2'
    unique_pairs_df = df[["AD", "borough_code", "borough_name"]].drop_duplicates()
    print("\nDataFrame with unique pairs of 'col1' and 'col2':")
    print(unique_pairs_df)

    unique_pairs_df.to_csv("../data/ed_manifest/ad_borough_map.csv", index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Scan local AD PDFs (23..87), robust backends."
    )
    ap.add_argument("--pdf-dir", default="./ad_pdfs")
    ap.add_argument("--start", type=int, default=23)
    ap.add_argument("--end", type=int, default=87)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument(
        "--ocr", action="store_true", help="Use OCR only if text backends fail."
    )
    ap.add_argument(
        "--strict-3-digit", action="store_true", help="Match ED as exactly 3 digits."
    )
    ap.add_argument(
        "--engine",
        choices=["auto", "pymupdf", "pdfminer"],
        default="auto",
        help="Backend preference: auto (PyMuPDF→pdfminer), or force one.",
    )
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    ads = list(range(args.start, args.end + 1))

    results: Dict[int, Dict] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                process_one_ad,
                ad,
                pdf_dir,
                args.max_pages,
                args.ocr,
                args.strict_3_digit,
                args.engine,
            ): ad
            for ad in ads
        }
        for fut in cf.as_completed(futs):
            ad = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "ad": ad,
                    "ok": False,
                    "reason": f"exception:{e}",
                    "eds": [],
                    "boros": [],
                    "pdf": str(pdf_dir / f"ad_{ad}.pdf"),
                }
            results[ad] = res
            status = "OK" if res.get("ok") else f"ERR({res.get('reason','')})"
            sys.stderr.write(
                f"AD {ad:02d} -> {status} | EDs={len(res.get('eds',[]))} | Boros={','.join(res.get('boros',[]))}\n"
            )

    new_rows = to_rows(results)
    existing = load_existing(Path(args.old)) if args.old else {}
    merged_rows, conflicts = merge_with_existing(new_rows, existing)

    out_df = pd.DataFrame(
        merged_rows,
        columns=[
            "AD",
            "ED",
            "borough_code",
            "borough_name",
            "ed_uid",
            "inferred_boroughs_from_pdf",
        ],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    if conflicts:
        sys.stderr.write(
            f"⚠️ {len(conflicts)} borough conflicts (kept existing; see inferred hints).\n"
        )
    n_ok = sum(1 for r in results.values() if r.get("ok"))
    n_err = sum(1 for r in results.values() if not r.get("ok"))
    sys.stderr.write(f"Done. OK={n_ok}, ERR={n_err}. Rows={len(out_df)} → {args.out}\n")

    # Uncomment to remove excess inferred_boroughs_from_pdf column
    # trim_cols(args.out)


if __name__ == "__main__":
    main()
