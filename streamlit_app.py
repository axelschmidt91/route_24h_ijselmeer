# streamlit_app_refactored.py
# -*- coding: utf-8 -*-
"""
Segel-Routenplaner (Browser, Python/Streamlit) — Refactored Single-File Version

Refactoring-Ziele:
- Gleiche Funktionalität wie Original beibehalten.
- Saubere Schichten/Objekte:
  * State/Cache (AppStateManager, FileCache)
  * Loader (DataLoader)
  * Geo/Math (Geo, BilinearTable)
  * Domain (RouteEngine)
  * Plot/Map (PlotEngine, MapEngine)
  * UI (UIApp) -> orchestriert Streamlit-Flow
- Code-Doppelungen entfernt (Uploader/Persistenz zentralisiert).
- Streamlit-UI übersichtlicher.

Hinweise:
- Verhalten/Outputs wurden bewusst an mehreren Stellen 1:1 erhalten
  (z.B. Zeit_h im Rückweg wie im Original), um Kompatibilität zu wahren.
"""

import os
import io
import json
import math
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from PIL import Image
import gpxpy
# import pygrib  # im Original auskommentiert, hier belassen zur Funktionsgleichheit

# ------------------------------- Constants -------------------------------------------

APP_STATE_PATH = "./app_state.json"
CACHE_DIR = "./data_cache"
ICON_ARROW_PATH = "icons/arrow_left.png"
EARTH_RADIUS_NM = 3440.065  # Nautical miles

os.makedirs(CACHE_DIR, exist_ok=True)


# =============================== Layer: State & Cache ================================

class FileCache:
    """Dateicache basierend auf Content-Hash (verhindert Dubletten)."""
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c for c in name if c.isalnum() or c in ("_", "-", "."))[:60] or "file"

    def save_bytes(self, name_hint: str, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        fname = f"{digest[:16]}_{self._safe_name(name_hint)}"
        path = os.path.join(self.cache_dir, fname)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(content)
        return path


class AppStateManager:
    """Laden/Speichern app_state.json und Session-Bridge."""
    def __init__(self, state_path: str = APP_STATE_PATH):
        self.state_path = state_path

    def load(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save(self, state: dict) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def ensure_session_defaults(self) -> None:
        loaded = self.load()
        if "visit_log" not in st.session_state:
            st.session_state["visit_log"] = loaded.get("visit_log", [])
        if "next_target" not in st.session_state:
            st.session_state["next_target"] = loaded.get("next_target", None)

    def set_app_state_in_session(self) -> dict:
        return st.session_state.setdefault("app_state", self.load())

    def get_key(self, key: str) -> Optional[dict]:
        app_state = self.set_app_state_in_session()
        return app_state.get(key)

    def put_key(self, key: str, info: dict) -> None:
        app_state = self.set_app_state_in_session()
        app_state[key] = info
        self.save(app_state)

    def clear_key(self, key: str) -> None:
        app_state = self.set_app_state_in_session()
        app_state.pop(key, None)
        self.save(app_state)


# =============================== Layer: Loaders ======================================

class DataLoader:
    """Einheitliche Upload-/Persistenz-Logik + Parser für CSV/GPX/XLSX/GRIB."""
    def __init__(self, state: AppStateManager, cache: FileCache):
        self.state = state
        self.cache = cache

    # ---------- Generic Upload Helper ----------
    def _uploader(self, label: str, key: str, types: List[str]) -> Optional[dict]:
        col1, col2 = st.columns([3, 1])
        with col1:
            file = st.file_uploader(label, type=types, key=key)
        with col2:
            st.write("")
            if st.button("⎚ Reset", key=f"reset_{key}"):
                self.state.clear_key(key)
                st.rerun()

        if file is not None:
            content = file.getvalue()
            cached_path = self.cache.save_bytes(file.name, content)
            info = {"path": cached_path, "name": file.name}
            self.state.put_key(key, info)
            return info

        info = self.state.get_key(key)
        if info and os.path.exists(info["path"]):
            return info
        return None

    # ---------- Specific Loaders ----------
    @staticmethod
    def _read_csv_flexible(path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep=",", encoding="utf-8", engine="python")

    def load_polar_csv(self, label: str, key: str, expected_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        info = self._uploader(label, key, ["csv"])
        if not info:
            return None
        try:
            df = self._read_csv_flexible(info["path"])
            if expected_cols and not set(expected_cols).issubset(df.columns):
                st.warning(f"{label}: Erwartete Spalten fehlen. Erwartet: {expected_cols}, gefunden: {list(df.columns)}")
            else:
                st.caption(f"Geladen: {info['name']} (persistiert)")
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
            return None

    def load_gpx_buoys(self, label: str, key: str) -> Optional[pd.DataFrame]:
        info = self._uploader(label, key, ["gpx"])
        if not info:
            return None
        try:
            with open(info["path"], "r", encoding="utf-8") as f:
                gpx = gpxpy.parse(f)
                data = {"Name": [], "LAT": [], "LON": []}
                for w in gpx.waypoints:
                    data["Name"].append(w.name)
                    data["LAT"].append(w.latitude)
                    data["LON"].append(w.longitude)
                df = pd.DataFrame(data)
            st.caption(f"Geladen: {info['name']} (persistiert)")
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
            return None

    def load_routes_xlsx(self, label: str, key: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        info = self._uploader(label, key, ["xlsx"])
        if not info:
            return None, None
        try:
            ext = os.path.splitext(info["name"])[1].lower()
            if ext == ".xlsx":
                df = pd.read_excel(info["path"])
            else:
                st.warning(f"{label}: Unbekanntes Dateiformat {ext}. Bitte XLSX.")
                return None, None
            if df.shape[1] < 2:
                st.warning(f"{label}: Erwartet mindestens 4 Spalten (Boje1, Boje2, Distanz, Anzahl_passieren).")
            else:
                st.caption(f"Geladen: {info['name']} (persistiert)")

            # Startbojen extrahieren (wie im Original)
            start_bojes = []
            for i, row in df.iterrows():
                if pd.isna(row[0]) or pd.isna(row[1]):
                    break
                if i == 0:
                    continue  # skip header
                start_bojes.append(str(row[0]))

            # Clean & Normalize
            df = df.dropna(subset=[df.columns[0], df.columns[1]])
            cols = ["Start", "End", "Distance", "MaxAmount"]
            df = df.iloc[:, :4]
            df.columns = cols[:df.shape[1]]

            # WV19 -> FINISH Ergänzung (wie Original)
            if "WV19" in df["Start"].values and "FINISH" not in df["End"].values:
                df = pd.concat(
                    [df, pd.DataFrame({"Start": ["WV19"], "End": ["FINISH"], "Distance": [4], "MaxAmount": [1]})],
                    ignore_index=True,
                )

            # Reorder
            if "MaxAmount" in df.columns and "Distance" in df.columns:
                df = df[["Start", "End", "MaxAmount", "Distance"]]

            # MaxAmount -> int, invalid droppen
            if "MaxAmount" in df.columns:
                df = df[pd.to_numeric(df["MaxAmount"], errors='coerce').notnull()]
                df["MaxAmount"] = df["MaxAmount"].astype(int)

            # Distance: Komma zu Punkt, numeric
            if "Distance" in df.columns:
                df["Distance"] = df["Distance"].astype(str).str.replace(",", ".")
                df["Distance"] = pd.to_numeric(df["Distance"], errors='coerce')
                df = df[pd.to_numeric(df["Distance"], errors='coerce').notnull()]

            return df.reset_index(drop=True), start_bojes
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
            return None, None

    def load_grib(self, label: str, key: str) -> Optional[pd.DataFrame]:
        info = self._uploader(label, key, ["grib", "grb", "bz2"])
        if not info:
            return None
        try:
            import pygrib  # lazy import, falls verfügbar
            grbs = pygrib.open(info["path"])
            records = []
            for grb in grbs:
                if grb.name in ["10 metre U wind component", "10 metre V wind component"]:
                    data = grb.data()
                    lats, lons = data[1], data[2]
                    values = data[0]
                    for i in range(lats.shape[0]):
                        for j in range(lats.shape[1]):
                            records.append({
                                "lat": float(lats[i, j]),
                                "lon": float(lons[i, j]),
                                "type": grb.name,
                                "value": float(values[i, j]),
                                "date": grb.validDate.strftime("%Y-%m-%d %H:%M")
                            })
            df = pd.DataFrame(records)
            st.caption(f"Geladen: {info['name']} (persistiert)")
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
            return None


# =============================== Layer: Geo & Interpolation ==========================

class Geo:
    @staticmethod
    def haversine_nm(lat1, lon1, lat2, lon2) -> float:
        phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dphi = phi2 - phi1
        dlam = lam2 - lam1
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return EARTH_RADIUS_NM * c

    @staticmethod
    def initial_bearing_deg(lat1, lon1, lat2, lon2) -> float:
        phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlam = lam2 - lam1
        x = math.sin(dlam) * math.cos(phi2)
        y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
        brng = math.degrees(math.atan2(x, y))
        return (brng + 360) % 360

    @staticmethod
    def angle_diff(a: float, b: float) -> float:
        """Smallest signed difference a-b in [-180,180)."""
        return (a - b + 180) % 360 - 180


class BilinearTable:
    """Bilineare Interpolation über (TWA, TWS) -> Wert.
    Unterstützt zyklische Winkel (0..360).
    """
    def __init__(self, df: pd.DataFrame, tws_col: str, twa_col: str, value_col: str):
        d = df.copy()
        d[twa_col] = d[twa_col] % 360
        self.angles = np.sort(d[twa_col].unique())
        self.tws = np.sort(d[tws_col].unique())
        pivot = d.pivot_table(index=twa_col, columns=tws_col, values=value_col, aggfunc='mean')
        pivot = pivot.reindex(index=self.angles, columns=self.tws)
        self.grid = pivot.values.astype(float)
        self._ffill_nan()

    def _ffill_nan(self):
        # Zeilen
        for i in range(self.grid.shape[0]):
            s = pd.Series(self.grid[i, :]).ffill().bfill()
            self.grid[i, :] = s.values
        # Spalten
        for j in range(self.grid.shape[1]):
            s = pd.Series(self.grid[:, j]).ffill().bfill()
            self.grid[:, j] = s.values

    @staticmethod
    def _interp1(x: np.ndarray, xi: float) -> Tuple[int, int, float]:
        if xi <= x[0]:
            return 0, 0, 0.0
        if xi >= x[-1]:
            return len(x)-1, len(x)-1, 0.0
        j = np.searchsorted(x, xi)
        x0, x1 = x[j-1], x[j]
        t = (xi - x0) / (x1 - x0) if x1 != x0 else 0.0
        return j-1, j, t

    def value(self, tws: float, twa_deg: float) -> float:
        a = twa_deg % 360
        angles = self.angles
        grid = self.grid
        if angles[0] != 0 or angles[-1] != 360:
            angles_ext = np.concatenate([angles, [angles[0]+360]])
            grid_ext = np.vstack([grid, grid[0:1, :]])
        else:
            angles_ext = angles
            grid_ext = grid
        i0, i1, ta = self._interp1(angles_ext, a)
        j0, j1, tt = self._interp1(self.tws, tws)
        v00 = grid_ext[i0, j0]; v01 = grid_ext[i0, j1]
        v10 = grid_ext[i1, j0]; v11 = grid_ext[i1, j1]
        v0 = v00*(1-tt) + v01*tt
        v1 = v10*(1-tt) + v11*tt
        return float(v0*(1-ta) + v1*ta)


# =============================== Layer: Domain (Routen/Optimierung) ==================

class RouteEngine:
    def __init__(self):
        pass

    @staticmethod
    def compute_route_table(
        buoys: pd.DataFrame, routes: pd.DataFrame,
        polar: Optional[BilinearTable], leeway: Optional[BilinearTable],
        twd_deg: float, tws_kn: float
    ) -> pd.DataFrame:
        bmap = {row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in buoys.iterrows()}

        rows = []
        for _, r in routes.iterrows():
            b1, b2 = str(r[0]), str(r[1])
            passes = int(r[2]) if len(r) > 2 and not pd.isna(r[2]) else 999999
            if b1 not in bmap or b2 not in bmap:
                continue
            lat1, lon1 = bmap[b1]; lat2, lon2 = bmap[b2]
            dist = round(Geo.haversine_nm(lat1, lon1, lat2, lon2), 2)
            crs_fwd = int(round(Geo.initial_bearing_deg(lat1, lon1, lat2, lon2), 0))
            crs_rev = (crs_fwd + 180) % 360

            twa_fwd = abs(Geo.angle_diff(twd_deg, crs_fwd))
            twa_rev = abs(Geo.angle_diff(twd_deg, crs_rev))

            bs_fwd = round(polar.value(tws_kn, twa_fwd), 2) if polar else np.nan
            bs_rev = round(polar.value(tws_kn, twa_rev), 2) if polar else np.nan

            lw_fwd = leeway.value(twa_fwd, tws_kn) if leeway else 0.0
            lw_rev = leeway.value(twa_rev, tws_kn) if leeway else 0.0

            sign_fwd = -1 if Geo.angle_diff(twd_deg, crs_fwd) < 0 else 1
            sign_rev = -1 if Geo.angle_diff(twd_deg, crs_rev) < 0 else 1
            steer_fwd = (crs_fwd + sign_fwd * lw_fwd) % 360
            steer_rev = (crs_rev + sign_rev * lw_rev) % 360

            t_fwd_h = round(dist / bs_fwd, 2) if bs_fwd and bs_fwd > 0 else np.nan
            t_rev_h = round(dist / bs_rev, 2) if bs_rev and bs_rev > 0 else np.nan

            # Vorwärts-Zeile
            rows.append({
                'Boje1': b1, 'Boje2': b2, 'Distanz_NM': dist,
                'COG': crs_fwd,
                'TWA_kt': twa_fwd,
                'HEADING': steer_fwd,
                'BS_kt': bs_fwd,
                'Zeit_h': t_fwd_h,
                'Max_Passieren': passes
            })
            # Rückwärts-Zeile (wie im Original: Zeit_h = t_fwd_h belassen)
            rows.append({
                'Boje1': b2, 'Boje2': b1, 'Distanz_NM': dist,
                'COG': crs_rev,
                'TWA_kt': twa_rev,
                'HEADING': steer_rev,
                'BS_kt': bs_rev,
                'Zeit_h': t_fwd_h,
                'Max_Passieren': passes
            })

        df = pd.DataFrame(rows)

        # Limits anhand Log/Nächster Boje verringern (Originalverhalten)
        log = st.session_state.get('visit_log', [])
        route_took = [l[0] for l in log]
        route_took.append(st.session_state.get('next_target', None))

        for i in range(1, len(route_took)):
            a = route_took[i-1]; b = route_took[i]
            mask1 = (df['Boje1'] == a) & (df['Boje2'] == b)
            mask2 = (df['Boje1'] == b) & (df['Boje2'] == a)
            df.loc[mask1 | mask2, 'Max_Passieren'] = df.loc[mask1 | mask2, 'Max_Passieren'] - 1

        return df

    @staticmethod
    def route_graph(routes: pd.DataFrame) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {}
        for _, r in routes.iterrows():
            a, b = str(r[0]), str(r[1])
            g.setdefault(a, []).append(b)
            g.setdefault(b, []).append(a)
        return g

    @staticmethod
    def interpolate_speed(polar: BilinearTable, tws: float, twa: float) -> float:
        return max(0.0, float(polar.value(tws, twa % 360)))

    @staticmethod
    def interpolate_leeway(leeway: BilinearTable, tws: float, twa: float) -> float:
        return max(0.0, float(leeway.value(tws, twa % 360)))

    @staticmethod
    def simulate_leg(
        bmap: Dict[str, Tuple[float, float]], polar: BilinearTable, leeway: BilinearTable,
        twd: float, tws: float, a: str, b: str
    ) -> Tuple[float, float, float, float]:
        lat1, lon1 = bmap[a]; lat2, lon2 = bmap[b]
        dist = Geo.haversine_nm(lat1, lon1, lat2, lon2)
        cog = Geo.initial_bearing_deg(lat1, lon1, lat2, lon2)
        leeway_deg = (-1 if Geo.angle_diff(twd, cog) < 0 else 1) * RouteEngine.interpolate_leeway(leeway, tws, abs(Geo.angle_diff(twd, cog)))
        heading = cog + leeway_deg
        twa = abs(Geo.angle_diff(twd, heading))
        bs = RouteEngine.interpolate_speed(polar, tws, twa) * math.cos(math.radians(leeway_deg))
        hours = dist / bs if bs > 0 else float('inf')
        return dist, heading, bs, hours

    @staticmethod
    def optimize_variants(
        buoys: pd.DataFrame, routes: pd.DataFrame, polar: BilinearTable, leeway: BilinearTable,
        start_b: str, goal_b: str, start_dt: datetime, deadline_dt: datetime,
        twd: float, tws: float, passed: List[Tuple[str, datetime]], next_target: Optional[str],
        n_variants: int = 10, max_steps: int = 30, restarts: int = 5
    ) -> List[dict]:
        bmap = {row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in buoys.iterrows()}

        # Limits
        limit_map: Dict[Tuple[str, str], int] = {}
        for _, r in routes.iterrows():
            a, b = str(r[0]), str(r[1])
            lim = int(r[2]) if len(r) > 2 and not pd.isna(r[2]) else 999999
            key = tuple(sorted([a, b]))
            limit_map[key] = lim

        # Bereits genutzte Kanten abziehen
        for i in range(1, len(passed)):
            a = passed[i-1][0]; b = passed[i][0]
            key = tuple(sorted([a, b]))
            if key in limit_map:
                limit_map[key] = max(0, limit_map[key] - 1)

        g = RouteEngine.route_graph(routes)

        def penalty(distance_nm: float, arrival: datetime) -> float:
            if arrival <= deadline_dt:
                return 0.0
            delta_min = (arrival - deadline_dt).total_seconds() / 60.0
            return distance_nm * (delta_min / 750.0)

        start_node = start_b
        current_time0 = start_dt
        variants: List[dict] = []

        for _v in range(n_variants):
            best_plan = None
            for _ in range(restarts):
                node = start_node
                tcur = current_time0
                plan = [node]
                times = [tcur]
                dist_sum = 0.0
                limits_left = dict(limit_map)
                bias_target = next_target

                for step in range(max_steps):
                    nbrs = [n for n in g.get(node, []) if limits_left.get(tuple(sorted([node, n])), 0) > 0]
                    if not nbrs:
                        break
                    scores = []
                    for nb in nbrs:
                        dist, course, bs, hours = RouteEngine.simulate_leg(bmap, polar, leeway, twd, tws, node, nb)
                        if not math.isfinite(hours):
                            sc = -1e9
                        else:
                            sc = dist
                            if nb == goal_b:
                                sc += 0.2 * dist
                            if bias_target and nb == bias_target:
                                sc += 0.1 * dist
                            sc += max(0.0, bs - 3.0)
                            sc *= (0.9 + 0.2 * random.random())
                        scores.append((sc, nb, dist, hours))
                    scores.sort(reverse=True)
                    topk = scores[:min(3, len(scores))]
                    _, nb, dist, hours = random.choice(topk)

                    plan.append(nb)
                    tcur = tcur + timedelta(hours=hours)
                    times.append(tcur)
                    dist_sum += dist
                    key = tuple(sorted([node, nb]))
                    limits_left[key] = limits_left.get(key, 0) - 1
                    node = nb
                    if node == goal_b and random.random() < 0.6:
                        break

                arr = times[-1]
                dist_after_penalty = dist_sum - penalty(dist_sum, arr)
                cand = {
                    'plan': plan, 'times': times,
                    'distance_nm': dist_sum,
                    'arrival': arr,
                    'score_nm': dist_after_penalty
                }
                if (best_plan is None) or (cand['score_nm'] > best_plan['score_nm']):
                    best_plan = cand
            variants.append(best_plan)

        variants.sort(key=lambda x: x['score_nm'], reverse=True)
        return variants


# =============================== Layer: Plot & Map ===================================

class PlotEngine:
    @staticmethod
    def plot_polar(polar_df: pd.DataFrame, tws_select: float) -> go.Figure:
        d = polar_df.copy()
        d.rename(columns={d.columns[0]: 'tws', d.columns[1]: 'twa', d.columns[2]: 'bsp'}, inplace=True)
        d['twa'] = d['twa'] % 360
        bl = BilinearTable(d, 'tws', 'twa', 'bsp')
        angles = np.linspace(0, 179, 180)
        speeds = [bl.value(tws_select, a) for a in angles]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=speeds, theta=angles, mode='lines', name=f'TWS {tws_select:.1f} kn'))
        fig.update_layout(
            polar={
                "radialaxis": {"title": 'Bootsgeschwindigkeit [kn]', "range": [0, 10]},
                "angularaxis": {"rotation": 90, "direction": "clockwise"},
                "sector": [-90, 90]
            },
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return fig

    @staticmethod
    def plot_leeway(leeway_df: pd.DataFrame, tws_select: float) -> go.Figure:
        d = leeway_df.copy()
        d.rename(columns={d.columns[0]: 'tws', d.columns[1]: 'twa', d.columns[2]: 'lw'}, inplace=True)
        d['twa'] = d['twa'] % 360
        bl = BilinearTable(d, 'tws', 'twa', 'lw')
        angles = np.linspace(0, 359, 360)
        vals = [bl.value(tws_select, a) for a in angles]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=vals, theta=angles, mode='lines', name=f'TWS {tws_select:.1f} kn'))
        fig.update_layout(
            polar={
                "radialaxis": {"title": 'Abdrift [°]', "range": [0, 10]},
                "angularaxis": {"rotation": 90, "direction": "clockwise"},
                "sector": [-90, 90]
            },
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        return fig


class MapEngine:
    def __init__(self, start_bojes: Optional[List[str]] = None):
        self.start_bojes = set(start_bojes or [])

    def make_map(self, buoys: pd.DataFrame, routes: pd.DataFrame, highlight_pairs: Optional[List[Tuple[str, str]]] = None) -> folium.Map:
        highlight_pairs = set(tuple(p) for p in (highlight_pairs or []))
        coords = [(float(r['LAT']), float(r['LON'])) for _, r in buoys.iterrows()]
        if not coords:
            return folium.Map(location=[54.5, 10.0], zoom_start=11, control_scale=True)

        lat_mean = np.mean([c[0] for c in coords]); lon_mean = np.mean([c[1] for c in coords])
        m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11, control_scale=True)

        # Marker
        for _, r in buoys.iterrows():
            name = str(r['Name'])
            if name.upper() == "FINISH":
                color = "red"
            elif name in self.start_bojes:
                color = "green"
            elif name == st.session_state.get('next_target', None):
                color = "orange"
            else:
                color = "blue"
            folium.Marker(
                location=[float(r['LAT']), float(r['LON'])],
                tooltip=name,
                icon=folium.Icon(color=color, icon="")
            ).add_to(m)

        # Linien
        bmap = {row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in buoys.iterrows()}
        for _, rr in routes.iterrows():
            b1, b2 = str(rr[0]), str(rr[1])
            if b1 not in bmap or b2 not in bmap:
                continue
            p1 = bmap[b1]; p2 = bmap[b2]
            dist = Geo.haversine_nm(p1[0], p1[1], p2[0], p2[1])
            crs = Geo.initial_bearing_deg(p1[0], p1[1], p2[0], p2[1])
            crs_back = (crs + 180) % 360
            txt = f"{b1} ⇄ {b2}\nDistanz: {dist:.2f} NM\nHin: {int(crs)}°, Zurück: {int(crs_back)}°"
            color = "red" if (b1, b2) in highlight_pairs or (b2, b1) in highlight_pairs else "blue"
            folium.PolyLine([p1, p2], tooltip=txt, color=color, weight=4 if color == "red" else 2, opacity=0.8).add_to(m)

        # Nächster Schlag
        log = st.session_state.get('visit_log', [])
        if log and st.session_state.get('next_target', None):
            last_b = log[-1][0]; next_b = st.session_state.get('next_target', None)
            if last_b in bmap and next_b in bmap:
                p1 = bmap[last_b]; p2 = bmap[next_b]
                folium.PolyLine([p1, p2], tooltip=f"Nächste Etappe: {last_b} → {next_b}",
                                color="orange", weight=4, opacity=0.8, dash_array='5,10').add_to(m)
                mid_lat = (p1[0] + p2[0]) / 2; mid_lon = (p1[1] + p2[1]) / 2
                m.location = [mid_lat, mid_lon]; m.zoom_start = 12
        return m


# =============================== Layer: UI Orchestrierung ============================

class UIApp:
    def __init__(self):
        self.state = AppStateManager()
        self.cache = FileCache()
        self.loader = DataLoader(self.state, self.cache)
        self.route_engine = RouteEngine()
        self.start_bojes: List[str] = []

        st.set_page_config(page_title="Segel-Routenplaner", layout="wide")
        self.state.ensure_session_defaults()

        # Platzhalter für geladene Daten
        self.buoys_df: Optional[pd.DataFrame] = None
        self.routes_df: Optional[pd.DataFrame] = None
        self.polar_df: Optional[pd.DataFrame] = None
        self.leeway_df: Optional[pd.DataFrame] = None
        self.forecast_df: Optional[pd.DataFrame] = None

        # Interpolatoren
        self.polar_interp: Optional[BilinearTable] = None
        self.leeway_interp: Optional[BilinearTable] = None

    # ---------- Helpers ----------
    def _update_next_targets(self):
        possible_targets = set()
        log = st.session_state.get('visit_log', [])
        if self.routes_df is None:
            st.session_state['next_possible_targets'] = []
            return
        if log:
            last_b = log[-1][0]
            for _, r in self.routes_df.iterrows():
                if r[0] == last_b:
                    possible_targets.add(r[1])
                elif r[1] == last_b:
                    possible_targets.add(r[0])
        else:
            possible_targets = set(self.start_bojes.copy())
        st.session_state['next_possible_targets'] = sorted(possible_targets)

    def _build_interpolators(self):
        self.polar_interp = None
        self.leeway_interp = None
        if self.polar_df is not None and len(self.polar_df.columns) >= 3:
            self.polar_interp = BilinearTable(self.polar_df, self.polar_df.columns[0], self.polar_df.columns[1], self.polar_df.columns[2])
        if self.leeway_df is not None and len(self.leeway_df.columns) >= 3:
            self.leeway_interp = BilinearTable(self.leeway_df, self.leeway_df.columns[0], self.leeway_df.columns[1], self.leeway_df.columns[2])

    # ---------- UI Sections ----------
    def section_uploads(self):
        with st.expander("ℹ️ Einlesen von Dateien"):
            st.write("Die folgenden Spalten werden in den jeweiligen CSV-Dateien erwartet:")
            st.markdown(
                """  
                **1. Polardaten (CSV):** `Windwinkel_deg, TWS_kn, BootsSpeed_kn`  
                **2. Abdrift (CSV):** `Windwinkel_deg, TWS_kn, Abdrift_deg`  
                """
            )
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                self.buoys_df = self.loader.load_gpx_buoys("1) Bojen GPX", key="buoys")
                self.routes_df, self.start_bojes = self.loader.load_routes_xlsx("2) Routen & Limits XLSX", key="routes")
                self.polar_df = self.loader.load_polar_csv("3) Polardaten CSV", key="polar")
            with col_u2:
                self.leeway_df = self.loader.load_polar_csv("4) Abdrift CSV", key="leeway")
                self.forecast_df = self.loader.load_grib("5) Windvorhersage GRIB (optional)", key="forecast")

            # Interpolatoren nach Uploads
            self._build_interpolators()

    def section_sidebar(self) -> Tuple[float, float, float]:
        sidebar = st.sidebar
        sidebar.header("Aktuelle Bedingungen & Optionen")
        cur_twd = sidebar.slider("TWD (°)", min_value=0, max_value=359, value=180, step=1)
        cur_tws = sidebar.slider("TWS (kn)", min_value=0, max_value=40, value=12, step=1)

        # Icon rotieren
        if os.path.exists(ICON_ARROW_PATH):
            img = Image.open(ICON_ARROW_PATH).convert("RGBA")
            rotated = img.rotate(-cur_twd + 90, resample=Image.BICUBIC, expand=True)
            canvas_size = max(rotated.width, rotated.height)
            canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
            x = (canvas_size - rotated.width) // 2
            y = (canvas_size - rotated.height) // 2
            canvas.paste(rotated, (x, y), rotated)
            sidebar.image(canvas, use_container_width=False)
        else:
            st.error(f"⚠️ Icon nicht gefunden unter {ICON_ARROW_PATH}")

        # Forecast -> TWD/TWS übernehmen (gleiche Logik wie im Original)
        if self.forecast_df is not None and 'Datum' in self.forecast_df.columns[0]:
            try:
                df_fc = self.forecast_df.copy()
                df_fc.columns = ['Datum_Uhrzeit', 'Windrichtung', 'Windgeschwindigkeit']
                df_fc['Datum_Uhrzeit'] = pd.to_datetime(df_fc['Datum_Uhrzeit'])
                t_pick = sidebar.datetime_input("Vorhersage-Zeit wählen", value=df_fc['Datum_Uhrzeit'].min())
                if t_pick is not None:
                    idx = (df_fc['Datum_Uhrzeit'] - t_pick).abs().idxmin()
                    cur_twd = float(df_fc.loc[idx, 'Windrichtung'])
                    cur_tws = float(df_fc.loc[idx, 'Windgeschwindigkeit'])
                    sidebar.info(f"Aus Vorhersage: TWD={cur_twd:.1f}°, TWS={cur_tws:.1f} kn")
            except Exception:
                pass

        # Diagramme
        right = st.sidebar.container()
        right.subheader("Polar & Abdrift Diagramme")
        tws_slider = right.slider("TWS für Diagramme [kn]", min_value=0.0, max_value=40.0, value=float(cur_tws), step=0.5)
        if self.polar_df is not None:
            right.plotly_chart(PlotEngine.plot_polar(self.polar_df, tws_slider), use_container_width=True)
        if self.leeway_df is not None:
            right.plotly_chart(PlotEngine.plot_leeway(self.leeway_df, tws_slider), use_container_width=True)

        return cur_twd, cur_tws, tws_slider

    def section_left_log(self, cur_twd: float, cur_tws: float):
        col_left = st.container()
        with col_left:
            with st.expander("📝 Log & Nächste Boje"):
                if self.buoys_df is not None and self.routes_df is not None:
                    log: List[Tuple[str, str]] = st.session_state.get('visit_log', [])
                    self._update_next_targets()
                    st.markdown("**Bereits passierte Bojen & Zeiten**")
                    with st.form("logform", clear_on_submit=False):
                        # Index für Next-Target
                        next_opts = st.session_state.get("next_possible_targets", [])
                        cur_next = st.session_state.get("next_target")
                        idx = next_opts.index(cur_next) if cur_next in next_opts and next_opts else 0
                        bsel = st.selectbox("Boje", options=next_opts, index=idx if next_opts else 0)
                        tsel = st.text_input("Zeit (YYYY-MM-DD HH:MM:SS)", value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        c1, c2 = st.columns(2)
                        with c1:
                            add = st.form_submit_button("➕ Log-Eintrag hinzufügen")
                        with c2:
                            remove_last = st.form_submit_button("↩️ Letzten Eintrag entfernen")
                    if add:
                        log.append((bsel, tsel))
                        st.session_state['visit_log'] = log
                    if remove_last and log:
                        log.pop()
                        st.session_state['visit_log'] = log
                    if log:
                        st.table(pd.DataFrame(log, columns=['Boje', 'Zeit']))

            # Nächste Boje Auswahl
            self._update_next_targets()
            next_target = st.selectbox("Nächste anzulaufende Boje", options=["(keine)"] + st.session_state.get("next_possible_targets", []))
            st.session_state['next_target'] = next_target if next_target != "(keine)" else None

            st.subheader(f"Route von {log[-1][0] if log else '(Start)'} → {st.session_state['next_target'] if st.session_state.get('next_target') else '(keine)'}")

            # Kennzahlen zur nächsten Boje
            log = st.session_state.get('visit_log', [])
            c1, c2 = st.columns(2)
            if log and st.session_state.get('next_target') is not None and st.session_state.get("route_table") is not None:
                last_b, last_t = log[-1]
                try:
                    route_table = st.session_state["route_table"]
                    row = route_table[(route_table['Boje1'] == last_b) & (route_table['Boje2'] == st.session_state['next_target'])]
                    if not row.empty:
                        dist = float(row.iloc[0]['Distanz_NM'])
                        crs = float(row.iloc[0]['COG'])
                        bs = float(row.iloc[0]['BS_kt'])
                        t_h = float(row.iloc[0]['Zeit_h'])
                        with c1:
                            if Geo.angle_diff(cur_twd, crs) < 0:
                                st.metric("True Wind Angle (TWA)", f"{chr(8594)} {int(abs(Geo.angle_diff(cur_twd, crs))):03d}°")
                            else:
                                st.metric("True Wind Angle (TWA)", f"{int(abs(Geo.angle_diff(cur_twd, crs))):03d}° {chr(8592)}")
                            st.metric(f"Kurs zu {st.session_state['next_target']}", f"{int(crs):03d}°")
                        with c2:
                            st.metric(f"Distanz zu {st.session_state['next_target']}", f"{dist:.2f} nm")
                    elif st.session_state['next_target'] != last_b:
                        st.warning(f"Keine Route von {last_b} zu {st.session_state['next_target']} in Routentabelle gefunden.")
                except Exception as e:
                    st.warning(f"Konnte Distanz/Kurs zur nächsten Boje nicht berechnen: {e}")

            # ETA
            if log and st.session_state.get('next_target') is not None and self.polar_interp is not None and self.leeway_interp is not None:
                last_b, last_t = log[-1]
                try:
                    t0 = datetime.strptime(last_t, "%Y-%m-%d %H:%M:%S")
                    bmap = {row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in self.buoys_df.iterrows()}
                    dist, heading, bs, hours = RouteEngine.simulate_leg(bmap, self.polar_interp, self.leeway_interp, cur_twd, cur_tws, last_b, st.session_state['next_target'])
                    eta = t0 + timedelta(hours=hours)
                    with c1:
                        st.metric("theoretische Speed", f"{bs:.2f} kn")
                        st.metric("benötigte Zeit", f"{int(hours)}:{int((hours-int(hours))*60):02d} h:min")
                    with c2:
                        total_dist = 0.0
                        if len(log) >= 2:
                            for i in range(1, len(log)):
                                a = log[i-1][0]; b = log[i][0]
                                lat1, lon1 = bmap[a]; lat2, lon2 = bmap[b]
                                total_dist += Geo.haversine_nm(lat1, lon1, lat2, lon2)
                        st.metric("Heading zu nächster Boje", f"{int(heading):03d}°")
                        st.metric("Bisher gesegelte Distanz", f"{total_dist:.2f} nm")
                        st.metric("Distanz inkl. nächste Boje", f"{(total_dist+dist):.2f} nm")
                        st.metric("ETA an nächster Boje", eta.strftime("%H:%M:%S"))
                except Exception as e:
                    st.warning(f"ETA konnte nicht berechnet werden: {e}")

    def section_center_map_table(self, cur_twd: float, cur_tws: float):
        st.subheader("Karte & Routentabelle")
        if self.buoys_df is not None and self.routes_df is not None:
            # Route-Table berechnen
            st.session_state["route_table"] = self.route_engine.compute_route_table(
                self.buoys_df, self.routes_df, self.polar_interp, self.leeway_interp, cur_twd, cur_tws
            )
            # Karte
            fmap = MapEngine(self.start_bojes).make_map(
                self.buoys_df, self.routes_df, highlight_pairs=st.session_state.get('highlight_pairs', [])
            )
            st_folium(fmap, width=None, height=500)

            # Tabelle nach Kontext filtern
            route_table = st.session_state.get("route_table")
            log = st.session_state.get('visit_log', [])
            if st.session_state.get('next_target'):
                rt = route_table[(route_table['Boje1'] == st.session_state['next_target']) & (route_table['Max_Passieren'] > 0)]
                st.markdown(f"**Routentabelle (nur Routen ab nächster Boje {st.session_state['next_target']})**")
            elif log:
                last_b = log[-1][0]
                rt = route_table[(route_table['Boje1'] == last_b) & (route_table['Max_Passieren'] > 0)]
                st.markdown(f"**Routentabelle (nur Routen ab letzter Boje {last_b})**")
            else:
                rt = route_table
                st.markdown("**Routentabelle (alle Routen)**")

            # sortiere das DataFrame nach "BS_kt" absteigend (dort sind die besten Routen oben)
            st.dataframe(rt.sort_values(by='BS_kt', ascending=False).reset_index(drop=True), use_container_width=True)
            if (rt['Max_Passieren'] <= 0).any():
                st.warning("Einige Routen können nicht mehr passiert werden (Max_Passieren ≤ 0). Bitte Log prüfen.")
        else:
            st.info("Bitte Bojen- und Routen-Dateien laden, um Karte und Tabelle zu sehen.")

    def section_optimization(self, cur_twd: float, cur_tws: float):
        st.markdown("---")
        st.subheader("Optimierung: 10 Varianten erzeugen")
        if self.buoys_df is not None and self.routes_df is not None and self.polar_interp is not None and self.leeway_interp is not None:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                start_b = st.selectbox("Start-Boje", options=self.start_bojes if self.start_bojes else list(self.buoys_df['Name'].astype(str).values), index=0)
                start_date = st.date_input("Start-Datum", value=datetime.now())
                start_dt = datetime.combine(start_date, st.time_input("Start-Zeit", value=datetime.now().time(), step=timedelta(minutes=5)))
            with c2:
                goal_b = "FINISH"
                deadline_date = st.date_input("Ziel-Datum", value=(start_date + timedelta(hours=24)))
                deadline_dt = datetime.combine(deadline_date, st.time_input("Ziel-Zeit", value=(start_dt + timedelta(hours=24)).time(), step=timedelta(minutes=5)))
                _hard_deadline_dt = datetime.combine(deadline_date, st.time_input("Harte Deadline", value=(deadline_dt + timedelta(hours=1)).time(), step=timedelta(minutes=5)))
            with c3:
                max_steps = st.number_input("Max. Schritte pro Variante", min_value=3, max_value=200, value=30)
                restarts = st.number_input("Restarts/Variante", min_value=1, max_value=20, value=5)
            with c4:
                n_variants = st.number_input("Anzahl Varianten", min_value=1, max_value=10, value=10)

            run_opt = st.button("🔁 Optimieren (lädt CSVs neu)")
            if run_opt:
                # Im Original werden die aktuell geladenen DataFrames verwendet (erneutes Einlesen aus Cache ist sinngleich)
                pol_interp = BilinearTable(self.polar_df, self.polar_df.columns[0], self.polar_df.columns[1], self.polar_df.columns[2])
                lw_interp = BilinearTable(self.leeway_df, self.leeway_df.columns[0], self.leeway_df.columns[1], self.leeway_df.columns[2])

                # Log aufbereiten
                passed_log: List[Tuple[str, datetime]] = []
                for b, t in st.session_state.get('visit_log', []):
                    try:
                        passed_log.append((b, datetime.strptime(t, "%Y-%m-%d %H:%M")))
                    except Exception:
                        pass

                variants = RouteEngine.optimize_variants(
                    self.buoys_df, self.routes_df, pol_interp, lw_interp,
                    start_b, goal_b, start_dt, deadline_dt,
                    cur_twd, cur_tws,
                    passed_log, st.session_state.get('next_target'),
                    n_variants=int(n_variants), max_steps=int(max_steps), restarts=int(restarts)
                )
                st.session_state['variants'] = variants

        # Variantenanzeige
        variants = st.session_state.get('variants', [])
        if variants:
            lc, rc = st.columns([1.0, 2.0])
            with lc:
                st.markdown("**Varianten (anklickbar)**")
                labels = []
                for i, v in enumerate(variants, 1):
                    score = v['score_nm']
                    arr = v['arrival'].strftime('%Y-%m-%d %H:%M')
                    labels.append(f"Variante {i}: {score:.2f} NM (ETA {arr})")
                selected = st.radio("", options=list(range(len(variants))), format_func=lambda i: labels[i])
                st.session_state['selected_variant'] = int(selected)
            with rc:
                sel = variants[st.session_state.get('selected_variant', 0)]
                pairs = [(sel['plan'][i], sel['plan'][i+1]) for i in range(len(sel['plan']) - 1)]
                st.session_state['highlight_pairs'] = pairs
                if self.buoys_df is not None and self.routes_df is not None:
                    fmap2 = MapEngine(self.start_bojes).make_map(self.buoys_df, self.routes_df, highlight_pairs=pairs)
                    st_folium(fmap2, width=None, height=500, returned_objects=[], key="map2", zoom=12)

            with st.expander("Übersicht aller Varianten (Top-Distanzen nach Abzug)"):
                rows = []
                for i, v in enumerate(variants, 1):
                    nxt1 = v['plan'][1] if len(v['plan']) > 1 else None
                    nxt2 = v['plan'][2] if len(v['plan']) > 2 else None
                    rows.append({
                        'Variante': i,
                        'Distanz_NM_nach_Abzug': round(v['score_nm'], 2),
                        'Ankunftszeit': v['arrival'].strftime('%Y-%m-%d %H:%M'),
                        'Nächste_1': nxt1, 'Nächste_2': nxt2,
                        'Gesamt_Distanz_NM': round(v['distance_nm'], 2)
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("Noch keine Varianten berechnet.")

    def section_selfcheck_and_persist(self):
        # Selbst-Check (wie Original, nur strukturiert)
        with st.expander("✅ Selbst-Check (automatisch)"):
            msgs = []
            if self.buoys_df is None:
                msgs.append("Bojen GPX fehlt -> Karte/Tabelle/Optimierung teilweise deaktiviert.")
            if self.routes_df is None:
                msgs.append("Routen CSV fehlt -> Karte/Tabelle/Optimierung deaktiviert.")
            if self.polar_df is None:
                msgs.append("Polar CSV fehlt -> Geschwindigkeiten/Zeiten/Diagramme/Optimierung eingeschränkt.")
            if self.leeway_df is None:
                msgs.append("Abdrift CSV fehlt -> Steuerkurs/Optimierung ohne Abdrift.")
            if not msgs:
                msgs.append("Alle Kerndateien vorhanden.")
            if self.routes_df is not None and len(self.routes_df.columns) < 2:
                msgs.append("Routen: Mindestens Spalten Boje1, Boje2 erforderlich.")
            if self.polar_df is not None and len(self.polar_df.columns) < 3:
                msgs.append("Polar: 3 Spalten benötigt (Winkel, TWS, Speed).")
            if self.leeway_df is not None and len(self.leeway_df.columns) < 3:
                msgs.append("Abdrift: 3 Spalten benötigt (Winkel, TWS, Abdrift).")
            st.write("\n".join(f"- {m}" for m in msgs))

        st.markdown("---")
        st.caption("© 2025 – Prototyp. Hinweise/Fehler bitte melden – wir iterieren weiter.")

        # Persistente Felder speichern (wie Original)
        save_dict = st.session_state.get("app_state", {})
        save_dict.update({
            "visit_log": st.session_state.get("visit_log", []),
            "next_target": st.session_state.get("next_target", None),
            "selected_variant": st.session_state.get("selected_variant", 0),
            "highlight_pairs": st.session_state.get("highlight_pairs", []),
            "next_possible_targets": st.session_state.get("next_possible_targets", []),
        })
        self.state.save(save_dict)

    # ---------- Main ----------
    def run(self):
        # Uploads/Inputs
        self.section_uploads()
        # Sidebar (TWD/TWS + Plots)
        cur_twd, cur_tws, _ = self.section_sidebar()

        # RouteTable vor linken Kennzahlen berechnen (entspricht Originalfluss)
        if self.buoys_df is not None and self.routes_df is not None:
            st.session_state["route_table"] = self.route_engine.compute_route_table(
                self.buoys_df, self.routes_df, self.polar_interp, self.leeway_interp, cur_twd, cur_tws
            )

        # Layout: links (Log & Kennzahlen), Mitte (Karte & Tabelle)
        left_col, center_col = st.columns([1.1, 2.0])
        with left_col:
            self.section_left_log(cur_twd, cur_tws)
        with center_col:
            self.section_center_map_table(cur_twd, cur_tws)

        # Optimierung
        self.section_optimization(cur_twd, cur_tws)

        # Self-Check + Persist
        self.section_selfcheck_and_persist()


# =============================== App Entry ===========================================

if __name__ == "__main__":
    UIApp().run()