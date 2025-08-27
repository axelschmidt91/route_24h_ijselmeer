# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Segel-Routenplaner (Browser, Python/Streamlit)

Features:
- CSV-Uploads für: Bojen, Routenlimits, Polar, Abdrift, Windvorhersage (optional)
- Persistenz: gewählte/hochgeladene Dateien werden lokal in ./data_cache/ gespeichert
  und Referenzen in ./app_state.json, sodass sie beim nächsten Start automatisch geladen werden
- Rechte Spalte: Polardiagramm (Plotly) mit Slider für TWS; darunter Abdriftdiagramm
- Eingaben: aktuelle Windrichtung/-geschwindigkeit -> Ausgabe theoretische Bootsgeschwindigkeit
- Mitte: interaktive Karte (Folium) mit Bojen & Routen, Zoom/Pan, Ausschnitt umfasst alle Bojen
  - Entfernungen (NM) und Richtungen (hin/zurück) als Tooltip
- Tabelle aller Routen mit: Boje1, Boje2, Distanz, Richtung hin/zurück,
  aktueller Windwinkel (hin/zurück), Kurs zu steuern (mit Abdrift),
  theoretische Fahrtzeit (hin/zurück) aus Polar
- Linke Spalte: Log der bereits passierten Bojen (mit Zeit); Auswahl der nächsten Boje
  -> Vorhersage ETA; unten Summen-Distanzen
- Optimierung: Eingabe Start/Ziel/Startzeit/Zeitfenster; generiere 10 Varianten (randomisierte Heuristik)
  unter Beachtung der Routen-Passierlimits und bereits passierter Bojen/Zeiten; 
  Ziel kann früh erreicht werden; verspätete Ankunft wird mit Distanzabzug bestraft:
  Abzug = gesegelte Distanz * (Minuten über Ziel) / 750.
- Reiter: Übersicht der 10 Varianten mit (Distanz nach Abzug), Ankunftszeit, nächste zwei Bojen.

Hinweis: Dies ist ein funktionsfähiger Prototyp mit solider Basis; für große Graphen
kann die Heuristik weiter verbessert werden (z.B. genetische Verfahren/ILP). 
"""

import os
import io
import json
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import gpxpy

APP_STATE_PATH = "./app_state.json"
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------- Utility: Persistence ---------------------------------

import hashlib

def _save_bytes_to_cache(name_hint: str, content: bytes) -> str:
    """Speichert Datei nur, wenn sie noch nicht existiert (per Hash)."""
    # Hash berechnen
    digest = hashlib.sha256(content).hexdigest()
    safe = "".join(c for c in name_hint if c.isalnum() or c in ("_", "-", "."))[:60]
    fname = f"{digest[:16]}_{safe or 'file'}"
    path = os.path.join(CACHE_DIR, fname)

    # Datei nur anlegen, wenn noch nicht vorhanden
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)
    return path


def _load_app_state_to_session():
    state = _load_app_state()
    for k, v in state.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_app_state() -> dict:
    if os.path.exists(APP_STATE_PATH):
        try:
            with open(APP_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_app_state(state: dict):
    with open(APP_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def next_targets(routes_df: pd.DataFrame):
    # next target. show only targets that are possible due to a route row entry.
    possible_targets = set()
    log = st.session_state.get('visit_log', [])
    if log:
        last_b = log[-1][0]
        for _, r in routes_df.iterrows():
            if r[0] == last_b:
                possible_targets.add(r[1])
            elif r[1] == last_b:
                possible_targets.add(r[0])
    else:
        # if no log, all bojes that are start_bojes are possible
        possible_targets = set(start_bojes.copy())

    # update session state "next_possible_targets"
    st.session_state['next_possible_targets'] = sorted(possible_targets)

# ------------------------------- CSV Loaders ------------------------------------------

def read_csv_flexible(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", encoding="utf-8", engine="python")


def handle_csv_uploader(label: str, key: str, expected_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """File uploader with persistence to cache; returns DataFrame or cached one."""
    col1, col2 = st.columns([3,1])
    with col1:
        file = st.file_uploader(label, type=["csv"], key=key)
    with col2:
        st.write("")
        clear = st.button("⎚ Reset", key=f"reset_{key}")
    app_state = st.session_state.setdefault("app_state", _load_app_state())

    if clear:
        app_state.pop(key, None)
        _save_app_state(app_state)
        st.rerun()

    # If new upload, cache it
    if file is not None:
        content = file.getvalue()
        cached_path = _save_bytes_to_cache(file.name, content)
        app_state[key] = {"path": cached_path, "name": file.name}
        _save_app_state(app_state)

    # Try to load from cache/state
    info = app_state.get(key)
    if info and os.path.exists(info["path"]):
        try:
            df = read_csv_flexible(info["path"])
            if expected_cols and not set(expected_cols).issubset(df.columns):
                st.warning(f"{label}: Erwartete Spalten fehlen. Erwartet: {expected_cols}, gefunden: {list(df.columns)}")
            else:
                st.caption(f"Geladen: {info['name']} (persistiert)")
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
    return None

def handle_gpx_uploader(label: str, key: str) -> Optional[pd.DataFrame]:
    col1, col2 = st.columns([3,1])
    with col1:
        file = st.file_uploader(label, type=["gpx"], key=key)
    with col2:
        st.write("")
        clear = st.button("⎚ Reset", key=f"reset_{key}")
    app_state = st.session_state.setdefault("app_state", _load_app_state())

    if clear:
        app_state.pop(key, None)
        _save_app_state(app_state)
        st.rerun()

    # If new upload, cache it
    if file is not None:
        content = file.getvalue()
        cached_path = _save_bytes_to_cache(file.name, content)
        app_state[key] = {"path": cached_path, "name": file.name}
        _save_app_state(app_state)

    # Try to load from cache/state
    info = app_state.get(key)
    if info and os.path.exists(info["path"]):
        try:
            with open(info["path"], "r", encoding="utf-8") as f:
                gpx = gpxpy.parse(f)
                bojes = {"Name": [], "LAT": [], "LON": []}
                for waypoint in gpx.waypoints:
                    bojes["Name"].append(waypoint.name)
                    bojes["LAT"].append(waypoint.latitude)
                    bojes["LON"].append(waypoint.longitude)
                df = pd.DataFrame(bojes)
            st.caption(f"Geladen: {info['name']} (persistiert)")
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
    return None

def handle_routes_uploader(label: str, key: str) -> Optional[pd.DataFrame]:
    col1, col2 = st.columns([3,1])
    with col1:
        file = st.file_uploader(label, type=["xlsx"], key=key)
    with col2:
        st.write("")
        clear = st.button("⎚ Reset", key=f"reset_{key}")
    app_state = st.session_state.setdefault("app_state", _load_app_state())

    if clear:
        app_state.pop(key, None)
        _save_app_state(app_state)
        st.rerun()


    # If new upload, cache it
    if file is not None:
        content = file.getvalue()
        cached_path = _save_bytes_to_cache(file.name, content)
        app_state[key] = {"path": cached_path, "name": file.name}
        _save_app_state(app_state)


    # Try to load from cache/state
    info = app_state.get(key)
    if info and os.path.exists(info["path"]):
        try:
            ext = os.path.splitext(info["name"])[1].lower()
            if ext == ".xlsx":
                df = pd.read_excel(info["path"])
            else:
                st.warning(f"{label}: Unbekanntes Dateiformat {ext}. Bitte XLSX.")
                return None
            if df.shape[1] < 2:
                st.warning(f"{label}: Erwartet mindestens 4 Spalten (Boje1, Boje2, Distanz, Anzahl_passieren).")
            else:
                st.caption(f"Geladen: {info['name']} (persistiert)")

            # Extract start bojes (those that appear in the first rows of df untill a empty row is found). start_bojes is a list
            start_bojes = []
            for i, row in df.iterrows():
                if pd.isna(row[0]) or pd.isna(row[1]):
                    break
                if i == 0:
                    continue  # skip header
                start_bojes.append(str(row[0]))

            # Clean df. Rename columns to "Start", "End", "Distance", "MaxAmount"
            df = df.dropna(subset=[df.columns[0], df.columns[1]])
            cols = ["Start", "End", "Distance", "MaxAmount"]
            df = df.iloc[:, :4]
            df.columns = cols[:df.shape[1]]

            # add a route from WV19 to FINISH with distance 4 and max amount 1
            if "WV19" in df["Start"].values and "FINISH" not in df["End"].values:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "Start": ["WV19"],
                                "End": ["FINISH"],
                                "Distance": [4],
                                "MaxAmount": [1],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            # reorder columns: Start, End, MaxAmount, Distance
            if "MaxAmount" in df.columns and "Distance" in df.columns:
                df = df[["Start", "End", "MaxAmount", "Distance"]]
            
            #remove rows where any value in "MaxAmount" is not a number
            if "MaxAmount" in df.columns:
                df = df[pd.to_numeric(df["MaxAmount"], errors='coerce').notnull()]
                df["MaxAmount"] = df["MaxAmount"].astype(int)

            # Distance has a "," as decimal separator -> convert to "."
            if "Distance" in df.columns:
                df["Distance"] = df["Distance"].astype(str).str.replace(",", ".")
                df["Distance"] = pd.to_numeric(df["Distance"], errors='coerce')
                df = df[pd.to_numeric(df["Distance"], errors='coerce').notnull()]
            
            # index reset
            df = df.reset_index(drop=True)
                
            return df, start_bojes
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
    return None, None

def handle_grib_uploader(label: str, key: str) -> Optional[pd.DataFrame]:
    col1, col2 = st.columns([3,1])
    with col1:
        file = st.file_uploader(label, type=["grib", "grb", "bz2"], key=key)
    with col2:
        st.write("")
        clear = st.button("⎚ Reset", key=f"reset_{key}")
    app_state = st.session_state.setdefault("app_state", _load_app_state())

    if clear:
        app_state.pop(key, None)
        _save_app_state(app_state)
        st.rerun()

    # If new upload, cache it
    if file is not None:
        content = file.getvalue()
        cached_path = _save_bytes_to_cache(file.name, content)
        app_state[key] = {"path": cached_path, "name": file.name}
        _save_app_state(app_state)

    # Try to load from cache/state
    info = app_state.get(key)
    if info and os.path.exists(info["path"]):
        try:
            import pygrib
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
                                "lat": float(lats[i,j]),
                                "lon": float(lons[i,j]),
                                "type": grb.name,
                                "value": float(values[i,j]),
                                "date": grb.validDate.strftime("%Y-%m-%d %H:%M")
                            })
            df = pd.DataFrame(records)
            st.caption(f"Geladen: {info['name']} (persistiert)")
            return df
        except Exception as e:
            st.error(f"Fehler beim Laden {info['name']}: {e}")
    return None


# ------------------------------- Geo helpers -----------------------------------------

EARTH_RADIUS_NM = 3440.065  # Nautical miles


def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    # Convert to radians
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return EARTH_RADIUS_NM * c


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlam = lam2 - lam1
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360) % 360


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b in [-180,180)."""
    d = (a - b + 180) % 360 - 180
    return d


# ------------------------------- Interpolation (Polar & Abdrift) ---------------------

class BilinearTable:
    """Simple bilinear interpolator for values on regular (x,y) grid.
    x: angle in deg (0..360 cyclic supported) or wind angle degrees, y: TWS (kn).
    """
    def __init__(self, df: pd.DataFrame, angle_col: str, tws_col: str, value_col: str):
        # Normalize angles to [0,360)
        d = df.copy()
        d[angle_col] = d[angle_col] % 360
        # Build pivot table
        self.angles = np.sort(d[angle_col].unique())
        self.tws = np.sort(d[tws_col].unique())
        pivot = d.pivot_table(index=angle_col, columns=tws_col, values=value_col, aggfunc='mean')
        # Ensure full grid
        pivot = pivot.reindex(index=self.angles, columns=self.tws)
        self.grid = pivot.values.astype(float)
        # Forward fill along axes for small gaps
        self._ffill_nan()

    def _ffill_nan(self):
        # fill along rows
        mask = np.isnan(self.grid)
        # forward/backward along tws axis
        for i in range(self.grid.shape[0]):
            row = self.grid[i, :]
            s = pd.Series(row)
            s = s.ffill().bfill()
            self.grid[i, :] = s.values
        # forward/backward along angle axis
        for j in range(self.grid.shape[1]):
            col = self.grid[:, j]
            s = pd.Series(col)
            s = s.ffill().bfill()
            self.grid[:, j] = s.values

    def _interp1(self, arr: np.ndarray, x: np.ndarray, xi: float) -> Tuple[int, int, float]:
        # find bracketing indices and fraction
        if xi <= x[0]:
            return 0, 0, 0.0
        if xi >= x[-1]:
            return len(x)-1, len(x)-1, 0.0
        j = np.searchsorted(x, xi)
        x0, x1 = x[j-1], x[j]
        t = (xi - x0) / (x1 - x0) if x1 != x0 else 0.0
        return j-1, j, t

    def value(self, angle_deg: float, tws: float) -> float:
        # cyclic angle: map to [0,360) and enable wrap by duplicating first row at +360
        a = angle_deg % 360
        # If grid spans 0..360 but may not include 360, extend for wrap
        angles = self.angles
        grid = self.grid
        if angles[0] != 0 or angles[-1] != 360:
            # Create wrapped arrays for interpolation near 0 boundary
            angles_ext = np.concatenate([angles, [angles[0]+360]])
            grid_ext = np.vstack([grid, grid[0:1, :]])
        else:
            angles_ext = angles
            grid_ext = grid
        i0, i1, ta = self._interp1(grid_ext[:,0], angles_ext, a)
        j0, j1, tt = self._interp1(grid_ext[0,:], self.tws, tws)
        v00 = grid_ext[i0, j0]
        v01 = grid_ext[i0, j1]
        v10 = grid_ext[i1, j0]
        v11 = grid_ext[i1, j1]
        v0 = v00*(1-tt) + v01*tt
        v1 = v10*(1-tt) + v11*tt
        return float(v0*(1-ta) + v1*ta)


# ------------------------------- Domain Logic ----------------------------------------

def compute_route_table(buoys: pd.DataFrame, routes: pd.DataFrame,
                        polar: Optional[BilinearTable], leeway: Optional[BilinearTable],
                        twd_deg: float, tws_kn: float) -> pd.DataFrame:
    """Return DataFrame with computed columns for each route edge (both directions)."""
    # Expect buoys columns: Name, Lat, Lon
    bmap ={row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in buoys.iterrows()}

    rows = []
    for _, r in routes.iterrows():
        b1, b2 = str(r[0]), str(r[1])
        passes = int(r[2]) if len(r) > 2 and not pd.isna(r[2]) else 999999
        if b1 not in bmap or b2 not in bmap:
            continue
        lat1, lon1 = bmap[b1]
        lat2, lon2 = bmap[b2]
        dist = round(haversine_nm(lat1, lon1, lat2, lon2),2)
        crs_fwd = int(round(initial_bearing_deg(lat1, lon1, lat2, lon2),0))
        crs_rev = (crs_fwd + 180) % 360

        # wind angles TWA (true wind angle) relative to course
        twa_fwd = abs(angle_diff(twd_deg, crs_fwd))
        twa_rev = abs(angle_diff(twd_deg, crs_rev))

        # boat speeds via polar
        bs_fwd = polar.value(twa_fwd, tws_kn) if polar else np.nan
        bs_rev = polar.value(twa_rev, tws_kn) if polar else np.nan

        # leeway (deg) positive to leeward; we'll add/subtract based on wind coming from left/right
        lw_fwd = leeway.value(twa_fwd, tws_kn) if leeway else 0.0
        lw_rev = leeway.value(twa_rev, tws_kn) if leeway else 0.0

        # Determine sign: if wind is to the right of course (negative angle_diff), steer left (add negative)
        sign_fwd = -1 if angle_diff(twd_deg, crs_fwd) < 0 else 1
        sign_rev = -1 if angle_diff(twd_deg, crs_rev) < 0 else 1
        steer_fwd = (crs_fwd + sign_fwd * lw_fwd) % 360
        steer_rev = (crs_rev + sign_rev * lw_rev) % 360

        t_fwd_h = dist / bs_fwd if bs_fwd and bs_fwd > 0 else np.nan
        t_rev_h = dist / bs_rev if bs_rev and bs_rev > 0 else np.nan

        rows.append({
            'Boje1': b1, 'Boje2': b2, 'Distanz_NM': dist,
            'Richtung_hin_deg': crs_fwd, 'Richtung_zurück_deg': crs_rev,
            'Akt_TWA_hin_deg': twa_fwd, 'Akt_TWA_zurück_deg': twa_rev,
            'Kurs_steuern_hin_deg': steer_fwd, 'Kurs_steuern_zurück_deg': steer_rev,
            'BS_hin_kn': bs_fwd, 'BS_zurück_kn': bs_rev,
            'Zeit_hin_h': t_fwd_h, 'Zeit_zurück_h': t_rev_h,
            'Max_Passieren': passes
        })

    df = pd.DataFrame(rows)

    # update with already passed legs from log and next target to decrease remaining limits. 
    log = st.session_state.get('visit_log', [])
    route_took = []
    for l in log:
        route_took.append(l[0])
    route_took.append(st.session_state.get('next_target', None))

    for i in range(1, len(route_took)):
        a = route_took[i-1]
        b = route_took[i]
        mask1 = (df['Boje1'] == a) & (df['Boje2'] == b)
        mask2 = (df['Boje1'] == b) & (df['Boje2'] == a)
        df.loc[mask1 | mask2, 'Max_Passieren'] = df.loc[mask1 | mask2, 'Max_Passieren'] - 1

    return df


# ------------------------------- Visualization: Polar & Leeway -----------------------

def plot_polar(polar_df: pd.DataFrame, tws_select: float) -> go.Figure:
    # polar_df columns expected: Windwinkel (deg), TWS (kn), Bootsgeschwindigkeit (kn)
    d = polar_df.copy()
    d.rename(columns={
        d.columns[0]: 'angle', d.columns[1]: 'tws', d.columns[2]: 'bs'
    }, inplace=True)
    d['angle'] = d['angle'] % 360
    # Filter nearest TWS values and interpolate along angle
    # Build Bilinear to interpolate a full 0..360 curve at selected TWS
    bl = BilinearTable(d, 'angle', 'tws', 'bs')
    angles = np.linspace(0, 359, 360)
    speeds = [bl.value(a, tws_select) for a in angles]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=speeds, theta=angles, mode='lines', name=f'TWS {tws_select:.1f} kn'))
    fig.update_layout(polar=dict(radialaxis=dict(title='Bootsgeschwindigkeit [kn]')), showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
    return fig


def plot_leeway(leeway_df: pd.DataFrame, tws_select: float) -> go.Figure:
    d = leeway_df.copy()
    d.rename(columns={
        d.columns[0]: 'angle', d.columns[1]: 'tws', d.columns[2]: 'lw'
    }, inplace=True)
    d['angle'] = d['angle'] % 360
    bl = BilinearTable(d, 'angle', 'tws', 'lw')
    angles = np.linspace(0, 359, 360)
    vals = [bl.value(a, tws_select) for a in angles]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=angles, mode='lines', name=f'TWS {tws_select:.1f} kn'))
    fig.update_layout(polar=dict(radialaxis=dict(title='Abdrift [°]')), showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
    return fig


# ------------------------------- Map -------------------------------------------------

def make_map(buoys: pd.DataFrame, routes: pd.DataFrame, highlight_pairs: Optional[List[Tuple[str,str]]] = None) -> folium.Map:
    highlight_pairs = set(tuple(p) for p in (highlight_pairs or []))
    coords = [(float(r['LAT']), float(r['LON'])) for _, r in buoys.iterrows()]
    if not coords:
        # default center
        m = folium.Map(location=[54.5, 10.0], zoom_start=11, control_scale=True)
        return m
    lat_mean = np.mean([c[0] for c in coords])
    lon_mean = np.mean([c[1] for c in coords])
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11, control_scale=True)

    # Add buoy markers
    for _, r in buoys.iterrows():
        name = str(r['Name'])
        if name.upper() == "FINISH":
            color = "red"
        elif name in start_bojes:
            color = "green"
        elif name == st.session_state.get('next_target', None):
            color = "orange"
        else:
            color = "blue"
        
        folium.Marker(location=[float(r['LAT']), float(r['LON'])], tooltip=str(r['Name']),icon=folium.Icon(color=color, icon="")).add_to(m) # icon=folium.Icon(color="blue", icon="info-sign")

    # Draw routes
    bmap = {row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in buoys.iterrows()}
    for _, rr in routes.iterrows():
        b1, b2 = str(rr[0]), str(rr[1])
        if b1 not in bmap or b2 not in bmap:
            continue
        p1 = bmap[b1]
        p2 = bmap[b2]
        dist = haversine_nm(p1[0], p1[1], p2[0], p2[1])
        crs = initial_bearing_deg(p1[0], p1[1], p2[0], p2[1])
        crs_back = (crs + 180) % 360
        txt = f"{b1} ⇄ {b2}\nDistanz: {dist:.2f} NM\nHin: {int(crs)}°, Zurück: {int(crs_back)}°"
        color = "red" if (b1,b2) in highlight_pairs or (b2,b1) in highlight_pairs else "blue"
        folium.PolyLine([p1, p2], tooltip=txt, color=color, weight=4 if color=="red" else 2, opacity=0.8).add_to(m)

    # if next_target is set, draw a line from last visited to next_target and center map on it
    log = st.session_state.get('visit_log', [])
    if log and st.session_state.get('next_target', None):
        last_b = log[-1][0]
        next_b = st.session_state.get('next_target', None)
        if last_b in bmap and next_b in bmap:
            p1 = bmap[last_b]
            p2 = bmap[next_b]
            folium.PolyLine([p1, p2], tooltip=f"Nächste Etappe: {last_b} → {next_b}", color="orange", weight=4, opacity=0.8, dash_array='5,10').add_to(m)
            # center map on midpoint of this line
            mid_lat = (p1[0] + p2[0]) / 2
            mid_lon = (p1[1] + p2[1]) / 2
            m.location = [mid_lat, mid_lon]
            m.zoom_start = 12

    # # Fit bounds
    # sw = [min(c[0] for c in coords), min(c[1] for c in coords)]
    # ne = [max(c[0] for c in coords), max(c[1] for c in coords)]
    # m.fit_bounds([sw, ne])
    return m


# ------------------------------- Optimization (Heuristic) ----------------------------

def interpolate_speed(polar: BilinearTable, twa: float, tws: float) -> float:
    return max(0.0, float(polar.value(twa % 360, tws)))


def interpolate_leeway(leeway: BilinearTable, twa: float, tws: float) -> float:
    return max(0.0, float(leeway.value(twa % 360, tws)))


def route_graph(routes: pd.DataFrame) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for _, r in routes.iterrows():
        a, b = str(r[0]), str(r[1])
        g.setdefault(a, []).append(b)
        g.setdefault(b, []).append(a)
    return g


def simulate_leg(bmap: Dict[str, Tuple[float,float]], polar: BilinearTable, leeway: BilinearTable,
                 twd: float, tws: float, a: str, b: str) -> Tuple[float, float, float, float]:
    """Return distance NM, course_deg, speed_kn, hours."""
    lat1, lon1 = bmap[a]
    lat2, lon2 = bmap[b]
    dist = haversine_nm(lat1, lon1, lat2, lon2)
    course = initial_bearing_deg(lat1, lon1, lat2, lon2)
    twa = abs(angle_diff(twd, course))
    bs = interpolate_speed(polar, twa, tws)
    hours = dist / bs if bs > 0 else float('inf')
    return dist, course, bs, hours


def optimize_variants(buoys: pd.DataFrame, routes: pd.DataFrame, polar: BilinearTable, leeway: BilinearTable,
                      start_b: str, goal_b: str, start_dt: datetime, deadline_dt: datetime,
                      twd: float, tws: float,
                      passed: List[Tuple[str, datetime]], next_target: Optional[str],
                      n_variants: int = 10, max_steps: int = 30, restarts: int = 5) -> List[dict]:
    bmap = {row['Name']: (float(row['LAT']), float(row['LON'])) for _, row in buoys.iterrows()}
    # route pass limits
    limit_map: Dict[Tuple[str,str], int] = {}
    for _, r in routes.iterrows():
        a, b, lim = str(r[0]), str(r[1]), int(r[2]) if len(r) > 2 and not pd.isna(r[2]) else 999999
        key = tuple(sorted([a,b]))
        limit_map[key] = lim
    g = route_graph(routes)

    # Incorporate already passed legs from log to decrease remaining limits
    for i in range(1, len(passed)):
        a = passed[i-1][0]
        b = passed[i][0]
        key = tuple(sorted([a,b]))
        if key in limit_map:
            limit_map[key] = max(0, limit_map[key]-1)

    variants: List[dict] = []

    def penalty(distance_nm: float, arrival: datetime) -> float:
        if arrival <= deadline_dt:
            return 0.0
        delta_min = (arrival - deadline_dt).total_seconds() / 60.0
        return distance_nm * (delta_min / 750.0)

    # Starting node/time
    start_node = start_b
    current_time0 = start_dt

    for v in range(n_variants):
        best_plan = None
        # multiple restarts to diversify
        for _ in range(restarts):
            node = start_node
            tcur = current_time0
            plan = [node]
            times = [tcur]
            dist_sum = 0.0
            limits_left = dict(limit_map)

            # if next_target is set, bias towards hitting it soon
            bias_target = next_target

            for step in range(max_steps):
                nbrs = [n for n in g.get(node, []) if limits_left.get(tuple(sorted([node,n])), 0) > 0]
                if not nbrs:
                    break
                # Score neighbors by projected speed/time and soft bias to goal/bias_target
                scores = []
                for nb in nbrs:
                    dist, course, bs, hours = simulate_leg(bmap, polar, leeway, twd, tws, node, nb)
                    if not math.isfinite(hours):
                        sc = -1e9
                    else:
                        sc = dist  # base on distance to encourage long sailing
                        # bias toward goal
                        if nb == goal_b:
                            sc += 0.2 * dist
                        if bias_target and nb == bias_target:
                            sc += 0.1 * dist
                        # discourage sharp beating (very low bs)
                        sc += max(0.0, bs - 3.0)
                        # tiny randomness
                        sc *= (0.9 + 0.2*random.random())
                    scores.append((sc, nb, dist, hours))
                scores.sort(reverse=True)
                # pick top-k randomly to diversify
                topk = scores[: min(3, len(scores))]
                _, nb, dist, hours = random.choice(topk)

                # move
                plan.append(nb)
                tcur = tcur + timedelta(hours=hours)
                times.append(tcur)
                dist_sum += dist
                key = tuple(sorted([node, nb]))
                limits_left[key] = limits_left.get(key, 0) - 1
                node = nb
                # stop if reached goal and still can continue but we may stop to avoid penalty
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

    # sort by score desc
    variants.sort(key=lambda x: x['score_nm'], reverse=True)
    return variants


# ------------------------------- Streamlit UI ----------------------------------------

st.set_page_config(page_title="Segel-Routenplaner", layout="wide")

# load visit log and next target from data in APP_STATE_PATH
loaded_state = _load_app_state()
if "visit_log" not in st.session_state:
    st.session_state["visit_log"] = loaded_state.get("visit_log", [])
if "next_target" not in st.session_state:
    st.session_state["next_target"] = loaded_state.get("next_target", None)



with st.expander("ℹ️ Einlesen von Dateien"):
    st.write("Die folgenden Spalten werden in den jeweiligen CSV-Dateien erwartet:")
    st.markdown(
        """  
        **1. Polardaten (CSV):** `Windwinkel_deg, TWS_kn, BootsSpeed_kn`  
        **2. Abdrift (CSV):** `Windwinkel_deg, TWS_kn, Abdrift_deg`  
        """
    )

    # --- Uploaders (persisted) ---
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        buoys_df = handle_gpx_uploader("1) Bojen GPX", key="buoys")
        routes_df, start_bojes = handle_routes_uploader("2) Routen & Limits XLSX", key="routes")
        polar_df = handle_csv_uploader("3) Polardaten CSV", key="polar")
    with col_u2:
        leeway_df = handle_csv_uploader("4) Abdrift CSV", key="leeway")
        forecast_df = handle_grib_uploader("5) Windvorhersage GRIB (optional)", key="forecast")

# Build interpolators if possible
polar_interp = None
leeway_interp = None
if polar_df is not None and len(polar_df.columns) >= 3:
    polar_interp = BilinearTable(polar_df, polar_df.columns[0], polar_df.columns[1], polar_df.columns[2])
if leeway_df is not None and len(leeway_df.columns) >= 3:
    leeway_interp = BilinearTable(leeway_df, leeway_df.columns[0], leeway_df.columns[1], leeway_df.columns[2])

# --- Controls: current wind ---
sidebar = st.sidebar
sidebar.header("Aktuelle Bedingungen & Optionen")
cur_twd = sidebar.number_input("Aktuelle Windrichtung (°)", min_value=0.0, max_value=359.9, value=180.0, step=1.0)
cur_tws = sidebar.number_input("Aktuelle Windgeschwindigkeit (kn)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)

# If forecast provided, allow picking a time to auto-fill twd/tws
if forecast_df is not None and 'Datum' in forecast_df.columns[0]:
    try:
        df_fc = forecast_df.copy()
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

# Theoretical speed at given TWD/TWS, at best course (max over angles)
if polar_df is not None and polar_interp is not None:
    # approximate by sampling angle every 1°
    angles = np.arange(0, 360)
    # Find course with TWA close to angles -> we need BS for given TWA directly
    speeds = np.array([polar_interp.value(a, cur_tws) for a in angles])
    max_bs = float(np.nanmax(speeds)) if speeds.size else float('nan')
    sidebar.metric("Theoretische aktuelle Geschwindigkeit (max über Kurse)", f"{max_bs:.2f} kn")

# Right column: Polar and Leeway plots
right = st.sidebar.container()
right.subheader("Polar & Abdrift Diagramme")
tws_slider = right.slider("TWS für Diagramme [kn]", min_value=0.0, max_value=40.0, value=float(cur_tws), step=0.5)
if polar_df is not None:
    fig_p = plot_polar(polar_df, tws_slider)
    right.plotly_chart(fig_p, use_container_width=True)
if leeway_df is not None:
    fig_l = plot_leeway(leeway_df, tws_slider)
    right.plotly_chart(fig_l, use_container_width=True)

# --- Compute route table ---
if buoys_df is not None and routes_df is not None:
    st.session_state["route_table"] = compute_route_table(buoys_df, routes_df, polar_interp, leeway_interp, cur_twd, cur_tws)

# Layout: Left (log), Center (map+table), Right (handled above)
col_left, col_center = st.columns([1.1, 2.0])

with col_left:
    with st.expander("📝 Log & Nächste Boje"):
        # st.subheader("Log & Nächste Boje")
        if buoys_df is not None:
            # Log state
            log: List[Tuple[str,str]] = st.session_state.get('visit_log', [])
            next_targets(routes_df)
            st.markdown("**Bereits passierte Bojen & Zeiten**")
            with st.form("logform", clear_on_submit=False):
                bsel = st.selectbox("Boje", options=st.session_state["next_possible_targets"], index=0 if st.session_state["next_possible_targets"] else None)
                tsel = st.text_input("Zeit (YYYY-MM-DD HH:MM)", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
                col_left, col_right = st.columns(2)
                with col_left:
                    add = st.form_submit_button("➕ Log-Eintrag hinzufügen")
                with col_right:
                    remove_last = st.form_submit_button("↩️ Letzten Eintrag entfernen")
            if add:
                log.append((bsel, tsel))
                st.session_state['visit_log'] = log
            if remove_last and log:
                log.pop()
                st.session_state['visit_log'] = log
            # show log
            if log:
                st.table(pd.DataFrame(log, columns=['Boje', 'Zeit']))

    next_targets(routes_df)
    next_target = st.selectbox("Nächste anzulaufende Boje", options=["(keine)"]+st.session_state["next_possible_targets"])
    st.session_state['next_target'] = None if next_target == "(keine)" else next_target

    # Give the course and distance to next target
    if log and st.session_state.get('next_target') and st.session_state["route_table"] is not None:
        last_b, last_t = log[-1]
        try:
            route_table = st.session_state["route_table"]
            row = route_table[((route_table['Boje1'] == last_b) & (route_table['Boje2'] == st.session_state['next_target'])) |
                            ((route_table['Boje2'] == last_b) & (route_table['Boje1'] == st.session_state['next_target']))]
            if not row.empty:
                dist = float(row.iloc[0]['Distanz_NM'])
                crs = float(row.iloc[0]['Richtung_hin_deg']) if row.iloc[0]['Boje1'] == last_b else float(row.iloc[0]['Richtung_zurück_deg'])
                bs = float(row.iloc[0]['BS_hin_kn']) if row.iloc[0]['Boje1'] == last_b else float(row.iloc[0]['BS_zurück_kn'])
                t_h = float(row.iloc[0]['Zeit_hin_h']) if row.iloc[0]['Boje1'] == last_b else float(row.iloc[0]['Zeit_zurück_h'])
                # Tabelle in Streamlit mit Metriken
                st.metric(f"Distanz zu {st.session_state['next_target']}", f"{dist:.2f} nm")
                st.metric(f"Kurs zu {st.session_state['next_target']}", f"{int(crs):03d}°")
                st.metric(f"True Wind Angle (TWA)", f"{int(abs(angle_diff(cur_twd, crs))):03d}°, {chr(8594) if angle_diff(cur_twd, crs)<0 else chr(8592)}")
                st.metric(f"Theoretische Geschwindigkeit dorthin", f"{bs:.2f} kn")
                try:
                    st.metric(f"Theoretische Zeit dorthin", f"{int(t_h):02d}:{int((t_h*60)%60):02d} h")
                except:
                    st.metric(f"Theoretische Zeit dorthin", "nan h nan min")

            else:
                st.warning(f"Keine Route von {last_b} zu {st.session_state['next_target']} in Routentabelle gefunden.")
        except Exception as e:
            st.warning(f"Konnte Distanz/Kurs zur nächsten Boje nicht berechnen: {e}")

    # ETA prediction from last log -> next_target
    if log and st.session_state.get('next_target') and polar_interp is not None and leeway_interp is not None:
        last_b, last_t = log[-1]
        try:
            t0 = datetime.strptime(last_t, "%Y-%m-%d %H:%M")
            bmap = {row['Name']: (float(row['Breitengrad']), float(row['Längengrad'])) for _, row in buoys_df.iterrows()}
            dist, course, bs, hours = simulate_leg(bmap, polar_interp, leeway_interp, cur_twd, cur_tws, last_b, st.session_state['next_target'])
            eta = t0 + timedelta(hours=hours)
            st.info(f"Prognose ETA an {st.session_state['next_target']}: {eta.strftime('%Y-%m-%d %H:%M')}  (Distanz {dist:.2f} NM, Speed {bs:.2f} kn)")
            # Totals
            total_dist = 0.0
            if len(log) >= 2:
                for i in range(1, len(log)):
                    a = log[i-1][0]; b = log[i][0]
                    lat1, lon1 = bmap[a]; lat2, lon2 = bmap[b]
                    total_dist += haversine_nm(lat1, lon1, lat2, lon2)
            st.metric("Bisher gesegelte Distanz", f"{total_dist:.2f} NM")
            st.metric("Distanz inkl. nächste Boje", f"{(total_dist+dist):.2f} NM")
        except Exception as e:
            st.warning(f"ETA konnte nicht berechnet werden: {e}")

with col_center:
    st.subheader("Karte & Routentabelle")
    if buoys_df is not None and routes_df is not None:
        # Highlight according to selected variant (set later)
        highlight = st.session_state.get('highlight_pairs', [])
        fmap = make_map(buoys_df, routes_df, highlight_pairs=highlight)
        st_data = st_folium(fmap, width=None, height=500)

        # Show route table if computed with the possible Routes from next selected boje as dataframe
        if buoys_df is not None and routes_df is not None:
            st.session_state["route_table"] = compute_route_table(buoys_df, routes_df, polar_interp, leeway_interp, cur_twd, cur_tws)
            route_table = st.session_state.get("route_table")
            log = st.session_state.get('visit_log', [])
            # select all rows of routetable that contain in Boje1 or Boje2 the next_target and max_Passieren > 0
            if st.session_state.get('next_target'):
                rt = route_table[(route_table['Boje1'] == st.session_state['next_target']) | (route_table['Boje2'] == st.session_state['next_target']) & (route_table['Max_Passieren'] > 0)]
                st.markdown(f"**Routentabelle (nur Routen ab nächster Boje {st.session_state['next_target']})**")

            # letze Logeintrag nehmen als Boje
            elif log:
                last_b = log[-1][0]
                rt = route_table[(route_table['Boje1'] == last_b) | (route_table['Boje2'] == last_b) & (route_table['Max_Passieren'] > 0)]
                st.markdown(f"**Routentabelle (nur Routen ab letzter Boje {last_b})**")

            else:
                rt = route_table
                st.markdown("**Routentabelle (alle Routen)**")
            st.dataframe(rt)

            # if any row has Max_Passieren <= 0, show a warning
            if (rt['Max_Passieren'] <= 0).any():
                st.warning("Einige Routen können nicht mehr passiert werden (Max_Passieren ≤ 0). Bitte Log prüfen.")
    else:
        st.info("Bitte Bojen- und Routen-Dateien laden, um Karte und Tabelle zu sehen.")

# ------------------------------- Optimierung UI --------------------------------------
st.markdown("---")
st.subheader("Optimierung: 10 Varianten erzeugen")
if buoys_df is not None and routes_df is not None and polar_interp is not None and leeway_interp is not None:
    bojes = list(buoys_df['Name'].astype(str).values)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_b = st.selectbox("Start-Boje", options=bojes)
        start_dt = st.datetime_input("Start-Datum & Zeit", value=datetime.now())
    with c2:
        goal_b = st.selectbox("Ziel-Boje", options=bojes, index=min(1, len(bojes)-1))
        deadline_dt = st.datetime_input("Ziel-Datum & Zeit (Deadline)", value=datetime.now()+timedelta(hours=12))
    with c3:
        max_steps = st.number_input("Max. Schritte pro Variante", min_value=3, max_value=200, value=30)
        restarts = st.number_input("Restarts/Variante", min_value=1, max_value=20, value=5)
    with c4:
        n_variants = st.number_input("Anzahl Varianten", min_value=1, max_value=10, value=10)

    run_opt = st.button("🔁 Optimieren (lädt CSVs neu)")

    if run_opt:
        # Reload from cache to honor "immer neu einladen"
        app_state = st.session_state.get("app_state", _load_app_state())
        def load_key(k):
            info = app_state.get(k)
            return read_csv_flexible(info['path']) if info and os.path.exists(info['path']) else None
        buoys_latest = load_key('buoys')
        routes_latest = load_key('routes')
        polar_latest = load_key('polar')
        leeway_latest = load_key('leeway')
        if buoys_latest is None or routes_latest is None or polar_latest is None or leeway_latest is None:
            st.error("Bitte alle benötigten CSVs hochladen (Bojen, Routen, Polar, Abdrift).")
        else:
            pol_interp = BilinearTable(polar_latest, polar_latest.columns[0], polar_latest.columns[1], polar_latest.columns[2])
            lw_interp = BilinearTable(leeway_latest, leeway_latest.columns[0], leeway_latest.columns[1], leeway_latest.columns[2])

            # Prepare passed log
            passed_log = []
            for b, t in st.session_state.get('visit_log', []):
                try:
                    passed_log.append((b, datetime.strptime(t, "%Y-%m-%d %H:%M")))
                except Exception:
                    pass
            variants = optimize_variants(buoys_latest, routes_latest, pol_interp, lw_interp,
                                         start_b, goal_b, start_dt, deadline_dt,
                                         cur_twd, cur_tws,
                                         passed_log, st.session_state.get('next_target'),
                                         n_variants=int(n_variants), max_steps=int(max_steps), restarts=int(restarts))
            st.session_state['variants'] = variants

# Show variants (left list) and overview tab
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
        # highlight on map
        sel = variants[st.session_state.get('selected_variant', 0)]
        pairs = [(sel['plan'][i], sel['plan'][i+1]) for i in range(len(sel['plan'])-1)]
        st.session_state['highlight_pairs'] = pairs
        if buoys_df is not None and routes_df is not None:
            fmap2 = make_map(buoys_df, routes_df, highlight_pairs=pairs)
            st_folium(fmap2, width=None, height=500)

    with st.expander("Übersicht aller Varianten (Top-Distanzen nach Abzug)"):
        rows = []
        for i, v in enumerate(variants, 1):
            nxt1 = v['plan'][1] if len(v['plan'])>1 else None
            nxt2 = v['plan'][2] if len(v['plan'])>2 else None
            rows.append({
                'Variante': i,
                'Distanz_NM_nach_Abzug': round(v['score_nm'],2),
                'Ankunftszeit': v['arrival'].strftime('%Y-%m-%d %H:%M'),
                'Nächste_1': nxt1, 'Nächste_2': nxt2,
                'Gesamt_Distanz_NM': round(v['distance_nm'],2)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.caption("Noch keine Varianten berechnet.")

# ------------------------------- Self-Review Checks ----------------------------------
with st.expander("✅ Selbst-Check (automatisch)"):
    msgs = []
    # Check required dataframes for core features
    if buoys_df is None:
        msgs.append("Bojen GPX fehlt -> Karte/Tabelle/Optimierung teilweise deaktiviert.")
    if routes_df is None:
        msgs.append("Routen CSV fehlt -> Karte/Tabelle/Optimierung deaktiviert.")
    if polar_df is None:
        msgs.append("Polar CSV fehlt -> Geschwindigkeiten/Zeiten/Diagramme/Optimierung eingeschränkt.")
    if leeway_df is None:
        msgs.append("Abdrift CSV fehlt -> Steuerkurs/Optimierung ohne Abdrift.")
    if not msgs:
        msgs.append("Alle Kerndateien vorhanden.")
    if routes_df is not None and len(routes_df.columns) < 2:
        msgs.append("Routen: Mindestens Spalten Boje1, Boje2 erforderlich.")
    # Polar/leeway 3 columns
    if polar_df is not None and len(polar_df.columns) < 3:
        msgs.append("Polar: 3 Spalten benötigt (Winkel, TWS, Speed).")
    if leeway_df is not None and len(leeway_df.columns) < 3:
        msgs.append("Abdrift: 3 Spalten benötigt (Winkel, TWS, Abdrift).")

    st.write("\n".join(f"- {m}" for m in msgs))

st.markdown("---")
st.caption("© 2025 – Prototyp. Hinweise/Fehler bitte melden – wir iterieren weiter.")


# Save log to cache for next run
save_dict = st.session_state.get("app_state", {})
save_dict.update({
    "visit_log": st.session_state.get("visit_log", []),
    "next_target": st.session_state.get("next_target", None),
    "variants": st.session_state.get("variants", []),
    "selected_variant": st.session_state.get("selected_variant", 0),
    "highlight_pairs": st.session_state.get("highlight_pairs", []),
    "next_possible_targets": st.session_state.get("next_possible_targets", []),
})
st.write(save_dict)
_save_app_state(save_dict)

# write all session state at the bottom for debugging
st.write(st.session_state)