from __future__ import annotations
import argparse
import logging
import sys
import time
import gc
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import psutil

# --- AWS SPEED BOOST ---
from astroquery.mast import Catalogs, Observations
Observations.enable_cloud_dataset(provider='AWS')
import lightkurve as lk

# --- START OF MONKEY PATCH ---
import astroquery.mast.cloud
class DummyProgressBar:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args): pass
astroquery.mast.cloud.ProgressBarOrSpinner = DummyProgressBar
# --- END OF MONKEY PATCH ---

# --- COLAB PERSISTENCE SETUP ---
BASE_DIR = Path("/content/drive/MyDrive/TESS_Project")
sys.path.insert(0, str(BASE_DIR))
from src.utils import load_config

log_dir = BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "data_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

_EXOPLANET_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+tic_id+from+ps+where+tic_id+is+not+null"
    "+and+default_flag=1&format=csv"
)

def fetch_confirmed_tic_ids() -> set[int]:
    try:
        import requests
        response = requests.get(_EXOPLANET_ARCHIVE_URL, timeout=180)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        confirmed_ids = set()
        for line in lines[1:]:
            clean_line = line.replace('"', '').replace('TIC', '').strip()
            if clean_line.isdigit():
                confirmed_ids.add(int(clean_line))
        return confirmed_ids
    except Exception as exc:
        logger.warning(f"Exoplanet archive fetch failed: {exc}")
        return set()

def fetch_batch_tic_metadata(tic_ids: list[int]) -> dict:
    logger.info(f"Querying MAST TIC Catalog for {len(tic_ids)} stars...")
    try:
        results = Catalogs.query_criteria(catalog="Tic", ID=tic_ids)
        meta_dict = {}
        for row in results:
            tid = int(row['ID'])
            meta_dict[tid] = {
                'teff': row['Teff'] if not np.ma.is_masked(row['Teff']) and not np.isnan(row['Teff']) else 5778.0,
                'logg': row['logg'] if not np.ma.is_masked(row['logg']) and not np.isnan(row['logg']) else 4.44,
                'rad':  row['rad']  if not np.ma.is_masked(row['rad'])  and not np.isnan(row['rad'])  else 1.0
            }
        logger.info(f"Successfully retrieved metadata for {len(meta_dict)} stars.")
        return meta_dict
    except Exception as e:
        logger.error(f"MAST Query failed: {e}")
        return {}

def apply_scientific_balancing(gv, lv, meta, labels, target_ratio=0.5):
    planet_idx = np.where(labels == 1)[0]
    non_planet_count = np.where(labels == 0)[0].shape[0]
    
    needed = int(non_planet_count * target_ratio) - len(planet_idx)
    if needed <= 0 or len(planet_idx) < 2: 
        return gv, lv, meta, labels

    logger.info(f"Balancing: Creating {needed} synthetic transits from {len(planet_idx)} real planets...")
    
    new_gv, new_lv, new_meta = [], [], []
    for _ in range(needed):
        idx1, idx2 = np.random.choice(planet_idx, 2, replace=True)
        lam = np.random.uniform(0.1, 0.9)
        
        synth_gv = (1 - lam) * gv[idx1] + lam * gv[idx2]
        synth_lv = (1 - lam) * lv[idx1] + lam * lv[idx2]
        synth_mt = (1 - lam) * meta[idx1] + lam * meta[idx2]
        
        new_gv.append(synth_gv + np.random.normal(0, 0.01, synth_gv.shape))
        new_lv.append(synth_lv + np.random.normal(0, 0.01, synth_lv.shape))
        new_meta.append(synth_mt)

    final_gv = np.concatenate([gv, np.array(new_gv)], axis=0)
    final_lv = np.concatenate([lv, np.array(new_lv)], axis=0)
    final_mt = np.concatenate([meta, np.array(new_meta)], axis=0)
    final_lb = np.concatenate([labels, np.ones(needed, dtype=np.int8)], axis=0)
    
    return final_gv, final_lv, final_mt, final_lb

def _bin_to_fixed_length(phase: np.ndarray, flux: np.ndarray, n_bins: int, phase_min: float = -0.5, phase_max: float = 0.5) -> np.ndarray:
    bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        vals = flux[bin_idx == i]
        if len(vals) > 0: binned[i] = np.median(vals)

    nan_mask = np.isnan(binned)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        interp = interp1d(bin_centres[~nan_mask], binned[~nan_mask], kind="linear", fill_value="extrapolate")
        binned[nan_mask] = interp(bin_centres[nan_mask])
    elif nan_mask.all():
        binned[:] = 0.0

    return binned.astype(np.float32)

def extract_dual_views(time_arr: np.ndarray, flux: np.ndarray, period: float, t0: float, global_length: int = 2048, local_length: int = 201, local_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    phase = ((time_arr - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  

    median = np.median(flux)
    std = np.std(flux)
    flux_norm = (flux - median) / (std + 1e-8)

    half_local = local_fraction / 2.0
    global_view = _bin_to_fixed_length(phase, flux_norm, global_length, -0.5, 0.5)
    local_view = _bin_to_fixed_length(phase, flux_norm, local_length, -half_local, half_local)
    return global_view, local_view

def _download_light_curve(tic_id: int, max_retries: int = 3) -> tuple[np.ndarray, np.ndarray] | None:
    for attempt in range(max_retries):
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime="short")
            if len(search) == 0:
                search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime="long")
            if len(search) == 0: return None

            cache_path = BASE_DIR / "data" / "raw_cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            
            collection = search.download_all(download_dir=str(cache_path.resolve()))
            if collection is None or len(collection) == 0: return None
                
            lc = collection.stitch(corrector_func=lambda x: x)
            lc = lc.remove_nans().remove_outliers(sigma=5.0)

            flux = lc.pdcsap_flux.value if hasattr(lc, "pdcsap_flux") else lc.flux.value
            
            time_arr = np.array(lc.time.value, copy=True)
            flux_arr = np.array(flux, copy=True)
            
            del search, collection, lc
            return time_arr, flux_arr
        except Exception:
            if attempt < max_retries - 1: time.sleep(2) 
            else: return None

def build_dataset(config: dict) -> None:
    logger.info("=== COLAB CLOUD PIPELINE INITIATED ===")
    
    # --- AUTO-CLEANER FOR RAW FITS ---
    #cache_dir = BASE_DIR / "data" / "raw_cache"
    #if cache_dir.exists():
     #   logger.info("Sweeping raw cache for corrupted files...")
     #   corrupt_count = 0
     #   for fits_file in cache_dir.rglob("*.fits"):
      #      if fits_file.stat().st_size == 65536:
       #         fits_file.unlink()
        #        corrupt_count += 1
        #if corrupt_count > 0:
         #   logger.info(f"Deleted {corrupt_count} corrupted 64KB files.")*/

    # --- TRUE CHECKPOINT DIRECTORY ---
    processed_cache_dir = BASE_DIR / "data" / "processed_cache"
    processed_cache_dir.mkdir(parents=True, exist_ok=True)
    
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gc_interval = 50 if total_ram_gb < 20 else 100
        
    logger.info(f"Colab Hardware Detected: {total_ram_gb:.1f} GB RAM")
    logger.info(f"RAM Protection: Flushing memory every {gc_interval} stars.")

    cfg_data, cfg_paths = config["data"], config["paths"]
    out_dir = Path(cfg_paths["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(cfg_paths["raw_csv"], comment="#")
    except FileNotFoundError:
        logger.error(f"CRITICAL ERROR: Could not find raw CSV at '{cfg_paths['raw_csv']}'.")
        sys.exit(1)
        
    df = df[(df["tce_model_snr"] >= cfg_data["snr_threshold"]) & (df["tce_model_snr"] != -1.0)]
    df = df[df["tce_num_transits"] >= cfg_data.get("min_transits", 2)]
    df = df.sort_values("tce_model_snr", ascending=False).drop_duplicates(subset="ticid").reset_index(drop=True)

    confirmed_ids = fetch_confirmed_tic_ids()
    df["label"] = 0
    df.loc[df["ticid"].isin(confirmed_ids), "label"] = 1

    tic_list = df['ticid'].unique().tolist()
    tic_metadata = fetch_batch_tic_metadata(tic_list)

    global_views, local_views, labels, tic_ids, stellar_meta = [], [], [], [], []

    # --- BULLETPROOF SEQUENTIAL LOOP ---
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Stars"):
        tic_id = int(row["ticid"])
        period = float(row["tce_period"])
        t0     = float(row["tce_time0bt"])
        label  = int(row["label"])

        meta = tic_metadata.get(tic_id, {'teff': 5778.0, 'logg': 4.44, 'rad': 1.0})
        meta_vec = np.array([meta['teff']/5778.0, meta['logg']/4.44, meta['rad']/1.0], dtype=np.float32)

        if period <= 0: continue

        # --- TRUE CHECKPOINT BYPASS ---
        npz_path = processed_cache_dir / f"{tic_id}.npz"
        res = None
        
        if npz_path.exists():
            try:
                cached = np.load(npz_path)
                res = (tic_id, label, cached['gv'], cached['lv'], cached['meta'])
            except Exception:
                npz_path.unlink() 

        if res is None:
            result = _download_light_curve(tic_id)
            if result is not None:
                time_arr, flux_arr = result
                try:
                    gv, lv = extract_dual_views(time_arr, flux_arr, period, t0, cfg_data["global_view_length"], cfg_data["local_view_length"], cfg_data["local_view_fraction"])
                    del time_arr, flux_arr
                    np.savez(npz_path, gv=gv, lv=lv, meta=meta_vec) 
                    res = (tic_id, label, gv, lv, meta_vec)
                except Exception:
                    pass

        if res is not None:
            t_id, lbl, gv, lv, meta = res
            tic_ids.append(t_id); labels.append(lbl); global_views.append(gv); local_views.append(lv); stellar_meta.append(meta)
        
        if i > 0 and i % gc_interval == 0:
            gc.collect()

    if not global_views:
        logger.error("FATAL: No stars were successfully processed.")
        return

    gv_arr, lv_arr, mt_arr, lb_arr = apply_scientific_balancing(
        np.stack(global_views), np.stack(local_views), np.stack(stellar_meta), np.array(labels), target_ratio=cfg_data.get("smote_ratio", 0.5)
    )

    np.save(out_dir / "global_views.npy", gv_arr)
    np.save(out_dir / "local_views.npy",  lv_arr)
    np.save(out_dir / "stellar_meta.npy", mt_arr)
    np.save(out_dir / "labels.npy",       lb_arr)

    logger.info(f"=== SUCCESS ===")
    logger.info(f"Final dataset saved to Google Drive at {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/content/drive/MyDrive/TESS_Project/configs/config.yaml")
    args = parser.parse_args()
    build_dataset(load_config(args.config))