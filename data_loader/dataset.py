
from __future__ import annotations
import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import random
import monai
import h5py
import data_loader.custom_transform.augmentors as aug
from collections import defaultdict
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional, Dict, Any, List, Tuple
from tempfile import NamedTemporaryFile


class Dataset_slide_lvl(monai.data.Dataset):
    GRID = 32  # spatial uniform-sampling grid, adjust based on instances size

    def __init__(self, params: Dict[str, Any], data_frame=None, device=None, fold: int=None,):
        self.df               = data_frame.reset_index(drop=True)
        self.device           = device
        self.shuffle          = True

        # config
        self.label_name       = params['inputs']['label_name']
        self.feature_dir      = Path(params['files']['data_location'])
        self.hand_feature_dir = Path(params['files']['hand_data_location'])
        self.max_bag_size     = int(params['inputs']['max_bag_size'])
        self.cancer_filter    = bool(params['inputs']['cancer_filter'])
        self.cancer_thr       = float(params['inputs']['cancer_filter_threshold'])
        self.concat_hand      = bool(params['inputs']['concat_hand_features'])

        # caching (simple)
        self.enable_cache     = bool(params['inputs'].get('enable_cache', False))
        fold_id               = fold
        base_cache            = Path(params['files'].get('cache_dir', self.feature_dir / "cache"))
        # namespace per fold to avoid conflicts across parallel sessions
        self.cache_root       = base_cache
        self.cache_dir        = base_cache / f"fold_{fold_id}"
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # slide paths
        self._fpath = {r['slide_id_ne']: self.feature_dir / f"{r['slide_id_ne']}.h5"
                       for _, r in self.df.iterrows()}
        self._hpath = {r['slide_id_ne']: self.hand_feature_dir / f"{r['slide_id_ne']}.h5"
                       for _, r in self.df.iterrows()} if self.concat_hand else {}

        # per-worker handles
        self._h5:  Dict[str, h5py.File] = {}
        self._h5h: Dict[str, h5py.File] = {}

        super().__init__(self.df)

    def __len__(self): return len(self.df)

    # ---------- HDF5 helpers ----------
    def _open(self, slide: str) -> h5py.File:
        h = self._h5.get(slide)
        if h is None:
            h = h5py.File(self._fpath[slide], 'r', libver='latest', swmr=True)
            self._h5[slide] = h
        return h

    def _open_hand(self, slide: str) -> h5py.File:
        h = self._h5h.get(slide)
        if h is None:
            h = h5py.File(self._hpath[slide], 'r', libver='latest', swmr=True)
            self._h5h[slide] = h
        return h

    @staticmethod
    def _safe_take(ds: h5py.Dataset, idx: np.ndarray) -> np.ndarray:
        """h5py fancy indexing requires increasing indices; sort then restore."""
        if idx is None or idx.size == 0:
            return ds[[]]
        idx = np.asarray(idx, dtype=np.int64)
        order = np.argsort(idx)
        out_sorted = ds[idx[order]]
        inv = np.empty_like(order); inv[order] = np.arange(order.size)
        return out_sorted[inv]

    # ---------- sampling ----------
    @staticmethod
    def _uniform_indices(coords: np.ndarray, grid: int, max_bag: int) -> np.ndarray:
        mins, maxs = coords.min(0), coords.max(0)
        norm = (coords - mins) / (maxs - mins + 1e-8)
        bins = np.floor(norm * grid).astype(np.int16)
        keys = bins[:, 0] * grid + bins[:, 1]
        order = np.random.permutation(len(keys))
        chosen, seen = [], set()
        for i in order:
            k = int(keys[i])
            if k not in seen:
                seen.add(k); chosen.append(i)
                if len(chosen) == max_bag:
                    break
        if len(chosen) < max_bag:
            rest = np.setdiff1d(np.arange(len(keys)), chosen, assume_unique=True)
            if rest.size > 0:
                extra = np.random.choice(rest, max_bag - len(chosen), replace=False)
                chosen = np.concatenate([np.asarray(chosen, dtype=int), extra])
        return np.asarray(chosen, dtype=int)

    # ---------- simple cache ----------
    def _key(self, sid: str) -> str:
        return f"{sid}__grid{self.GRID}__bag{self.max_bag_size}__cf{int(self.cancer_filter)}__thr{self.cancer_thr:.4f}"

    def _cache_file(self, sid: str) -> Path:
        return self.cache_dir / f"{self._key(sid)}.npz"

    def _load_cache(self, sid: str) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        if not self.enable_cache:
            return None
        p = self._cache_file(sid)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as data:
                feats  = data["feats"]
                coords = data["coords"]
                n_total= int(data["n_total"])
            if feats.shape[0] != coords.shape[0]:
                return None
            return feats, coords, n_total
        except Exception:
            return None

    def _save_cache(self, sid: str, feats: np.ndarray, coords: np.ndarray, n_total: int):
        if not self.enable_cache:
            return
        final = self._cache_file(sid)
        if final.exists():
            return
        final.parent.mkdir(parents=True, exist_ok=True)

        # per-slide lock (handles multi-workers within the same fold)
        lock = final.with_suffix(final.suffix + ".lock")
        lock_fd = None
        try:
            try:
                lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_RDONLY)
            except FileExistsError:
                for _ in range(200):
                    if final.exists():
                        return
                    time.sleep(0.05)
                return

            if final.exists():
                return

            # write temp in the SAME dir; then atomic replace
            with NamedTemporaryFile(prefix=final.name + ".", dir=str(final.parent), delete=False) as tmp:
                np.savez(tmp,
                         feats=feats.astype(np.float32, copy=False),
                         coords=coords.astype(np.float32, copy=False),
                         n_total=np.int64(n_total))
                tmp_path = Path(tmp.name)

            os.replace(str(tmp_path), str(final))

        finally:
            try:
                if lock_fd is not None: os.close(lock_fd)
            except Exception:
                pass
            try:
                if lock.exists(): os.remove(lock)
            except Exception:
                pass
            # NamedTemporaryFile cleanup if replace failed
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    # ---------- selection ----------
    def _select_indices(self, h5f: h5py.File) -> Tuple[np.ndarray, int]:
        n_total = int(h5f['features'].shape[0])
        if self.cancer_filter:
            if self.cancer_thr > 0:
                probs = h5f['cd_probs'][:, 1]
                idx_all = np.flatnonzero(probs >= self.cancer_thr)
            else:
                preds = h5f['preds'][...]
                idx_all = np.flatnonzero(preds == 1)
        else:
            idx_all = np.arange(n_total, dtype=np.int64)

        if idx_all.size > self.max_bag_size:
            coords_all = self._safe_take(h5f['coords'], idx_all)
            chosen_rel = self._uniform_indices(coords_all, self.GRID, self.max_bag_size)
            idx_final = idx_all[chosen_rel]
        else:
            idx_final = idx_all
        return idx_final, n_total

    # ---------- main ----------
    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        sid   = row['slide_id_ne']
        pid   = row.get('patient_id', -1)
        label = torch.as_tensor(row[self.label_name]).long()

        cached = self._load_cache(sid)
        if cached is not None:
            feats, coords, n_total = cached
        else:
            h5f = self._open(sid)
            idx_final, n_total = self._select_indices(h5f)
            coords = self._safe_take(h5f['coords'], idx_final).astype(np.float32, copy=False)
            feats  = self._safe_take(h5f['features'], idx_final).astype(np.float32, copy=False)
            self._save_cache(sid, feats, coords, n_total)

        if self.shuffle and feats.shape[0] > 1:
            p = np.random.permutation(feats.shape[0])
            feats, coords = feats[p], coords[p]

        if self.concat_hand:
            h5h     = self._open_hand(sid)
            feats_h = h5h['features'][...]
            coords_h= h5h['coords'][...]
            key_to_row = { (int(c[0]), int(c[1])): i for i, c in enumerate(coords_h) }
            gather = np.array([key_to_row.get((int(c[0]), int(c[1])), -1) for c in coords], dtype=np.int64)
            has = gather >= 0
            if np.any(has):
                joined = np.concatenate([feats[has], feats_h[gather[has]]], axis=1)
                if not np.all(has):
                    pad = np.zeros((np.sum(~has), feats_h.shape[1]), dtype=feats.dtype)
                    feats = np.concatenate([joined, np.concatenate([feats[~has], pad], axis=1)], axis=0)
                else:
                    feats = joined

        tumor_ratio = feats.shape[0] / max(1, n_total)
        return {
            "patient_id":        pid,
            "slide_id":          sid,
            "image":             torch.from_numpy(feats),
            "coords":            torch.from_numpy(coords),
            "label":             label,
            "tumor_ratio":       tumor_ratio,
            "tumor_patch_count": feats.shape[0],
        }

    # ---------- cleanup ----------
    def purge_cache(self):
        """Delete this fold's cache dir (or base cache if no fold_id). Call at the end of this session."""
        if not self.enable_cache:
            return
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"[purge_cache] warn: {e}")

    @staticmethod
    def purge_all_cache_dir(cache_root: str):
        """Delete the entire cache root (all folds). Call ONCE after all folds finish."""
        p = Path(cache_root)
        try:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            print(f"[purge_all_cache_dir] warn: {e}")



class Dataset_slide_CSSL(monai.data.PersistentDataset):
    def __init__(self, data_path=None, device=None, num_patches=5000, grid_size=32):
        # Set all input args as attributes
        self.__dict__.update(locals()) 
        # ---->data
        self.data_path = data_path
        files = [{"filename": os.path.abspath(os.path.join(self.data_path, f))}for f in os.listdir(self.data_path) if f.endswith('.h5')]
        self.data_frame = pd.DataFrame(files)
        self.num_patches = num_patches
        self.grid_size = grid_size
        self.device = device

        self.transform = aug.Compose([
                                      aug.InstanceMasking(pf=0.8), 
                                      aug.InstanceFeatureReplace(pf=0.8), 
                                      aug.InstanceReplace(pf=0.8), 
                                      aug.InstanceFeatureNoise(pf=0.8),
                                      aug.InstanceFeatureDrop(pf=0.8),
                                      aug.InstanceFeatureDropout(pf=0.8)
                                    ])

        # ---->order
        self.shuffle = True
        self.data_frame = self.data_frame.reset_index(drop=True)

    def __len__(self):
        return len(self.data_frame)

    def _uniform_spatial_sample(self, coords, features):
        # Normalize coords to [0, 1]
        norm_coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)
        grid_coords = (norm_coords * self.grid_size).astype(int)
        grid_coords = np.clip(grid_coords, 0, self.grid_size - 1)

        cell_to_indices = defaultdict(list)
        for i, cell in enumerate(grid_coords):
            cell_to_indices[tuple(cell)].append(i)

        selected_indices = []
        cells = list(cell_to_indices.keys())
        random.shuffle(cells)

        for cell in cells:
            idx = random.choice(cell_to_indices[cell])
            selected_indices.append(idx)
            if len(selected_indices) >= self.num_patches:
                break

        # Fill in if fewer than requested
        if len(selected_indices) < self.num_patches:
            remaining = list(set(range(len(coords))) - set(selected_indices))
            extra = random.sample(remaining, min(self.num_patches - len(selected_indices), len(remaining)))
            selected_indices.extend(extra)

        selected_indices = np.array(selected_indices[:self.num_patches])
        return features[selected_indices], coords[selected_indices]
    
    def __getitem__(self, idx):
        full_path = self.data_frame['filename'][idx]
        with h5py.File(full_path, 'r') as file:
            features = file['features'][:]
            coords = file['coords'][:]
            try:
                preds = file['preds'][:]
                # Filter by prediction and remove NaNs
                valid = (preds == 1)
                features = features[valid]
                coords = coords[valid]
            except:
                pass
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]
            coords = coords[mask]

        # Uniform sampling if needed
        if len(features) > self.num_patches:
            features, coords = self._uniform_spatial_sample(coords, features)    
        # -----> Padding if needed
        pad_len = self.num_patches - len(features)
        if pad_len > 0:
            feature_dim = features.shape[1]
            coord_dim = coords.shape[1]

            pad_feat = np.zeros((pad_len, feature_dim), dtype=np.float32)
            pad_coord = np.zeros((pad_len, coord_dim), dtype=np.float32)

            features = np.concatenate([features, pad_feat], axis=0)
            coords = np.concatenate([coords, pad_coord], axis=0)
        # ----> shuffle
        if self.shuffle:
            indices = np.random.permutation(len(features))
            features = features[indices]
            coords = coords[indices]

        q = self.transform(torch.from_numpy(features).to(self.device))
        k = self.transform(torch.from_numpy(features).to(self.device))
        return q, k, torch.from_numpy(coords),full_path