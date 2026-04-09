"""
data_pipeline.py — TESS light curve acquisition and dual-view preprocessing.

Pipeline:
  1. Load TCE metadata CSV and apply data-quality filters.
  2. Cross-reference TIC IDs with the NASA Confirmed Exoplanet Archive to
     assign ground-truth binary labels (1 = confirmed planet, 0 = false positive).
  3. Download each light curve via the MAST API (using lightkurve).
  4. Phase-fold the light curve on the TCE period and epoch.
  5. Extract two views:
       - Global view : full phase-folded curve, binned to global_view_length pts.
       - Local view  : zoomed window around the transit, binned to local_view_length pts.
  6. Save as .npy arrays and a labels CSV.

Key corrections vs original plan (from CSV analysis):
  - Epoch column is tce_time0bt (BTJD = BJD - 2457000), NOT tce_time0bk.
  - tce_disp_pname does NOT exist in this CSV; labels come from Exoplanet Archive only.
  - tce_model_snr uses -1 as a sentinel (259 rows); filtered before SNR threshold.
  - 99 rows have < 2 transits and cannot be reliably phase-folded; filtered out.
  - 2714 rows are duplicate TIC IDs (multi-TCE stars); we keep the highest-SNR
    TCE per star to avoid data leakage in train/val/test splits.

Usage:
    python src/data_pipeline.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# NASA Exoplanet Archive TAP — confirmed planets with TIC IDs
_EXOPLANET_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+tic_id+from+ps+where+tic_id+is+not+null"
    "+and+default_flag=1&format=csv"
)


# ── Label utilities ───────────────────────────────────────────────────────────

def fetch_confirmed_tic_ids() -> set[int]:
    try:
        response = requests.get(_EXOPLANET_ARCHIVE_URL, timeout=120)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        
        confirmed_ids = set()
        for line in lines[1:]:
            # Fix: Strip quotes and 'TIC' prefix before checking
            clean_line = line.replace('"', '').replace('TIC', '').strip()
            if clean_line.isdigit():
                confirmed_ids.add(int(clean_line))
                
        return confirmed_ids
    except Exception as exc:
        logger.warning(f"Exoplanet archive fetch failed: {exc}")
        return set()


def assign_labels(df: pd.DataFrame, confirmed_ids: set[int]) -> pd.Series:
    """
    Assign binary labels: 1 = confirmed planet, 0 = false positive / unknown.

    NOTE: tce_disp_pname does NOT exist in TESS Sectors 1-13 TCE CSV.
    Labels are assigned solely via cross-referencing the NASA Exoplanet Archive.
    """
    labels = pd.Series(0, index=df.index, dtype=np.int8)

    if confirmed_ids:
        mask_archive = df["ticid"].isin(confirmed_ids)
        labels[mask_archive] = 1
        logger.info(f"  Labelled {mask_archive.sum():,} positive entries via Exoplanet Archive.")
    else:
        logger.warning(
            "  No confirmed IDs available — all labels are 0. "
            "Training on this data will not converge correctly."
        )

    positive_count = labels.sum()
    total = len(labels)
    logger.info(
        f"  Label balance → positive: {positive_count:,} ({positive_count/total*100:.1f}%) "
        f"negative: {total - positive_count:,} ({(total - positive_count)/total*100:.1f}%)"
    )
    return labels


# ── Data cleaning ─────────────────────────────────────────────────────────────

def clean_and_filter(df: pd.DataFrame, cfg_data: dict) -> pd.DataFrame:
    """
    Apply all data-quality filters to the raw TCE metadata DataFrame.

    Filters applied (in order):
      1. Remove sentinel SNR rows (tce_model_snr == -1).
      2. Apply SNR threshold (>= snr_threshold).
      3. Remove TCEs with fewer than min_transits transits.
      4. Remove duplicate TIC IDs — keep highest-SNR TCE per star to avoid
         data leakage between train/val/test splits.
    """
    original_len = len(df)
    logger.info(f"Cleaning dataset. Starting rows: {original_len:,}")

    # ── Step 1: remove sentinel -1 SNR rows ──────────────────────────────────
    sentinel_mask = df["tce_model_snr"] == -1.0
    df = df[~sentinel_mask].copy()
    logger.info(f"  After removing SNR=-1 sentinels : {len(df):,} rows (removed {sentinel_mask.sum():,})")

    # ── Step 2: SNR threshold ─────────────────────────────────────────────────
    snr_threshold = cfg_data["snr_threshold"]
    df = df[df["tce_model_snr"] >= snr_threshold].reset_index(drop=True)
    logger.info(f"  After SNR >= {snr_threshold}            : {len(df):,} rows")

    # ── Step 3: minimum transit count ────────────────────────────────────────
    min_transits = cfg_data.get("min_transits", 2)
    df = df[df["tce_num_transits"] >= min_transits].reset_index(drop=True)
    logger.info(f"  After num_transits >= {min_transits}       : {len(df):,} rows")

    # ── Step 4: deduplicate TIC IDs ───────────────────────────────────────────
    # Sort by SNR descending so idxmax() picks the best TCE per star
    df = (
        df.sort_values("tce_model_snr", ascending=False)
          .drop_duplicates(subset="ticid", keep="first")
          .reset_index(drop=True)
    )
    logger.info(f"  After dedup (1 TCE per TIC ID)  : {len(df):,} rows")
    logger.info(f"  Total removed by all filters    : {original_len - len(df):,} rows")

    return df


# ── Phase folding & view extraction ──────────────────────────────────────────

def _bin_to_fixed_length(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    phase_min: float = -0.5,
    phase_max: float = 0.5,
) -> np.ndarray:
    """
    Bin (phase, flux) onto a uniform grid of n_bins points.
    Uses median aggregation (robust to stellar flares/outliers) then linearly
    interpolates any empty bins.
    """
    bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        vals = flux[bin_idx == i]
        if len(vals) > 0:
            binned[i] = np.median(vals)

    nan_mask = np.isnan(binned)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        interp = interp1d(
            bin_centres[~nan_mask],
            binned[~nan_mask],
            kind="linear",
            fill_value="extrapolate",
        )
        binned[nan_mask] = interp(bin_centres[nan_mask])
    elif nan_mask.all():
        binned[:] = 0.0

    return binned.astype(np.float32)


def extract_dual_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    global_length: int = 2048,
    local_length: int = 201,
    local_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a raw (time, flux) light curve into two phase-folded views.

    IMPORTANT: t0 must be in the same time system as `time`.
    lightkurve returns time in BTJD (BJD - 2457000) when using TESS data.
    The CSV column tce_time0bt is already in BTJD, so no conversion needed.

    Global view : full phase space [-0.5, 0.5], binned to global_length.
    Local view  : transit window [-(local_fraction/2), +(local_fraction/2)],
                  binned to local_length for fine-grained transit morphology.
    """
    # Phase-fold: centre transit at phase = 0
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # map to [-0.5, 0.5]

    # Robust flux normalisation: (flux - median) / std
    median = np.median(flux)
    std = np.std(flux)
    flux_norm = (flux - median) / (std + 1e-8)

    half_local = local_fraction / 2.0

    global_view = _bin_to_fixed_length(
        phase, flux_norm, global_length, -0.5, 0.5
    )
    local_view = _bin_to_fixed_length(
        phase, flux_norm, local_length, -half_local, half_local
    )
    return global_view, local_view

# ── Download helpers ──────────────────────────────────────────────────────────

def _download_light_curve(
    tic_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        # 1. SPEED FIX: Only search for official "SPOC" data
        search = lk.search_lightcurve(
            f"TIC {tic_id}", mission="TESS", author="SPOC", exptime="short"
        )
        if len(search) == 0:
            search = lk.search_lightcurve(
                f"TIC {tic_id}", mission="TESS", author="SPOC", exptime="long"
            )
        if len(search) == 0:
            return None

        # 2. SECTOR FIX: Only process the first 2 available sectors
        search = search[:2]

        # 3. STORAGE FIX: Save directly to the E:\ drive
        cache_path = Path("data/raw_cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        
        collection = search.download_all(download_dir=str(cache_path))
        
        # Safety check: ensure the download actually yielded files
        if collection is None or len(collection) == 0:
            return None
            
        # 4. MATH FIX: Disable median division to prevent inverted planets
        lc = collection.stitch(corrector_func=lambda x: x)
        lc = lc.remove_nans().remove_outliers(sigma=5.0)

        # Prefer systematics-corrected flux
        if hasattr(lc, "pdcsap_flux"):
            flux = lc.pdcsap_flux.value
        else:
            flux = lc.flux.value

        time = lc.time.value
        return time, flux

    except Exception as exc:
        print(f"\n[DOWNLOAD ERROR] TIC {tic_id} failed: {type(exc).__name__} - {exc}")
        return None

# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_dataset(config: dict) -> None:
    cfg_data  = config["data"]
    cfg_paths = config["paths"]

    raw_csv = Path(cfg_paths["raw_csv"])
    out_dir = Path(cfg_paths["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    logger.info(f"Loading TCE metadata from {raw_csv}...")
    df = pd.read_csv(raw_csv, comment="#")
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # ── Validate required columns ──────────────────────────────────────────────
    # NOTE: epoch column is tce_time0bt (BTJD), NOT tce_time0bk (doesn't exist)
    required = ["ticid", "tce_period", "tce_time0bt", "tce_model_snr", "tce_num_transits"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from CSV: {missing}\n"
            "Check the TESS TCE CSV schema — column names may vary by sector batch."
        )

    # ── Clean & filter ─────────────────────────────────────────────────────────
    df = clean_and_filter(df, cfg_data)

    # ── Assign labels ──────────────────────────────────────────────────────────
    confirmed_ids = fetch_confirmed_tic_ids()
    df["label"] = assign_labels(df, confirmed_ids)

    # ── Download, preprocess, and save ────────────────────────────────────────
    global_views: list[np.ndarray] = []
    local_views:  list[np.ndarray] = []
    labels:       list[int]        = []
    tic_ids:      list[int]        = []
    skipped = 0

    logger.info("Downloading and preprocessing light curves...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="TCEs"):
        tic_id = int(row["ticid"])
        period = float(row["tce_period"])
        # Use tce_time0bt (BTJD) — consistent with lightkurve's time system
        t0     = float(row["tce_time0bt"])

        if period <= 0:
            skipped += 1
            continue

        result = _download_light_curve(tic_id)
        if result is None:
            skipped += 1
            continue

        time, flux = result

        try:
            gv, lv = extract_dual_views(
                time, flux, period, t0,
                global_length=cfg_data["global_view_length"],
                local_length=cfg_data["local_view_length"],
                local_fraction=cfg_data["local_view_fraction"],
            )
        except Exception as exc:
            print(f"\n[PROCESSING ERROR] TIC {tic_id} extraction failed: {type(exc).__name__} - {exc}")
            skipped += 1
            continue

        global_views.append(gv)
        local_views.append(lv)
        labels.append(int(row["label"]))
        tic_ids.append(tic_id)

    if not global_views:
        raise RuntimeError(
            "No light curves were successfully processed. "
            "Check your internet connection and the CSV path."
        )

    global_arr = np.stack(global_views, axis=0)  # (N, global_length)
    local_arr  = np.stack(local_views,  axis=0)  # (N, local_length)
    labels_arr = np.array(labels, dtype=np.int8)  # (N,)

    np.save(out_dir / "global_views.npy", global_arr)
    np.save(out_dir / "local_views.npy",  local_arr)
    np.save(out_dir / "labels.npy",       labels_arr)

    # Save metadata for traceability and reproducibility
    meta = pd.DataFrame({"tic_id": tic_ids, "label": labels})
    meta.to_csv(out_dir / "metadata.csv", index=False)

    logger.info(
        f"\nDataset saved to {out_dir}/"
        f"\n  Processed successfully : {len(labels_arr):,}"
        f"\n  Skipped (no download)  : {skipped:,}"
        f"\n  Positives              : {labels_arr.sum():,} ({labels_arr.mean()*100:.1f}%)"
        f"\n  Negatives              : {(1 - labels_arr).sum():,}"
        f"\n  Global view shape      : {global_arr.shape}"
        f"\n  Local  view shape      : {local_arr.shape}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the TESS dual-view dataset.")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    build_dataset(config)