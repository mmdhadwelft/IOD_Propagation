import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sgp4.api import Satrec, jday
import datetime
import numpy as np
import itertools
from math import acos, degrees
import time
from numba import njit
import threading
import queue
import itertools
from multiprocessing import Pool, cpu_count
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


matplotlib.use("TkAgg")
mu_earth = 398600.4418  # km^3/s^2

# ----------------- TLE (HIDDEN IN CODE) -----------------
# CARTOSAT-3 TLE (kept in code, not exposed in GUI)
TLE_LINE1 = "1 44804U 19081A   25254.28859446  .00003156  00000+0  15280-3 0  9996"
TLE_LINE2 = "2 44804  97.4717 314.9649 0013827  50.9413 309.3049 15.19227258321125"
try:
    SAT = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
except Exception:
    SAT = None

#------------------------------------------------------------
MU = 398600.8       # km^3/s^2
RE = 6378.135       # km
J2 = 1.082616e-3
J3 = -2.53881e-6
J4 = -1.65597e-6

# ----------------------------
# Sun & Moon helpers (kept)
# ----------------------------
def norm(v):
    return np.linalg.norm(v)

def angle_between_deg(u, v):
    nu = norm(u); nv = norm(v)
    if nu == 0 or nv == 0:
        return 180.0
    c = np.clip(np.dot(u, v) / (nu*nv), -1.0, 1.0)
    return degrees(acos(c))

def orbital_elements_from_rv(r, v, mu=MU):
    r = np.array(r, dtype=float)
    v = np.array(v, dtype=float)
    rmag = norm(r)
    vmag = norm(v)
    h = np.cross(r, v)
    hmag = norm(h)
    energy = vmag**2/2.0 - mu/rmag
    a = -mu/(2.0*energy) if energy != 0 else np.inf
    e_vec = (1.0/mu) * ((vmag**2 - mu/rmag)*r - np.dot(r, v)*v)
    e = norm(e_vec)
    return {"a": a, "e": e, "energy": energy, "hmag": hmag, "e_vec": e_vec, "rmag": rmag, "vmag": vmag}

# def herrick_gibbs(r1, r2, r3, t1, t2, t3,mu=398600.4418):
#     dt21 = t2 - t1
#     dt32 = t3 - t2
#     dt31 = t3 - t1
#     if abs(dt21) < 1e-9 or abs(dt32) < 1e-9 or abs(dt31) < 1e-9:
#         return np.zeros(3)
#       # Magnitudes
#     r1_norm = np.linalg.norm(r1)
#     r2_norm = np.linalg.norm(r2)
#     r3_norm = np.linalg.norm(r3)

#     # # Correction factors due to orbital motion
#     # f1 = 1 + mu * dt21**2 / (12 * r1_norm**3)
#     # f2 = 1 + mu * dt31**2 / (12 * r2_norm**3)
#     # f3 = 1 + mu * dt32**2 / (12 * r3_norm**3)

    
#     # term1 = -dt32 / (dt21 * dt31) * f1*np.array(r1, dtype=float)
#     # term2 = (dt32 - dt21) / (dt21 * dt32) *f2* np.array(r2, dtype=float)
#     # term3 = dt21 / (dt32 * dt31) *f3* np.array(r3, dtype=float)

#     term1 = -dt32 * ((1 / (dt21 * dt31)) + (mu / (12 * r1**3))) * np.array(r1, dtype=float)
#     term2 = (dt32 - dt21) * ((1 / (dt21 * dt32)) + (mu / (12 * r2**3))) *np.array(r2, dtype=float)
#     term3 = dt21 * ((1 / (dt32 * dt31)) + (mu / (12 * r3**3))) * np.array(r3, dtype=float)
#     # term1 = -dt32 / (dt21 * dt31) * f1 * r1
#     # term2 = (dt32 - dt21) / (dt21 * dt32) * f2 * r2
#     # term3 = dt21 / (dt32 * dt31) * f3 * r3

#     v2 = term1 + term2 + term3
#     return v2

# ------------------ Utility functions ------------------
def circular_mean_deg(angles_deg):
    a = np.radians(angles_deg)
    s = np.mean(np.sin(a))
    c = np.mean(np.cos(a))
    return np.degrees(np.arctan2(s, c)) % 360

def angle_difference_deg(a_deg, b_deg):
    """Smallest signed difference (a - b) in [-180, 180]"""
    d = (a_deg - b_deg + 180) % 360 - 180
    return d

def fit_signal_savgol(y, window_length=51, polyorder=3):
    n = len(y)
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window_length < 3:
        return y.copy()
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(y, window_length, polyorder)

def estimate_noise_std_from_residuals(residuals):
    return np.std(residuals, ddof=1)

def estimate_noise_from_spherical(df):
    r = df['range_km'].values.astype(float)
    az = df['azimuth_deg'].values.astype(float)
    el = df['elevation_deg'].values.astype(float)

    # Smooth the signals using Savitzky-Golay filter
    r_fit = fit_signal_savgol(r, window_length=51, polyorder=3)
    el_fit = fit_signal_savgol(el, window_length=51, polyorder=3)
    
    # For azimuth, unwrap first to handle circularity
    az_unwrapped = np.degrees(np.unwrap(np.radians(az)))
    az_fit_unwrapped = fit_signal_savgol(az_unwrapped, window_length=51, polyorder=3)
    az_fit = (az_fit_unwrapped + 360) % 360

    # Residuals
    resid_r = r - r_fit
    resid_el = el - el_fit
    resid_az = angle_difference_deg(az, az_fit)

    # Noise std
    sigma_r = estimate_noise_std_from_residuals(resid_r)
    sigma_el = estimate_noise_std_from_residuals(resid_el)
    sigma_az = estimate_noise_std_from_residuals(resid_az)

    return sigma_r, sigma_az, sigma_el

def herrick_gibbs(r1, r2, r3, t1, t2, t3, mu=398600.4418):
    dt21 = t2 - t1
    dt32 = t3 - t2
    dt31 = t3 - t1

    if abs(dt21) < 1e-9 or abs(dt32) < 1e-9 or abs(dt31) < 1e-9:
        return np.zeros(3)

    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    r3 = np.array(r3, dtype=float)

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    r3_norm = np.linalg.norm(r3)

    # Correction factors (Vallado)
    f1 = 1 + mu * dt21**2 / (12 * r1_norm**3)
    f2 = 1 + mu * dt31**2 / (12 * r2_norm**3)
    f3 = 1 + mu * dt32**2 / (12 * r3_norm**3)

    term1 = -dt32 / (dt21 * dt31) * f1 * r1
    term2 = (dt32 - dt21) / (dt21 * dt32) * f2 * r2
    term3 = dt21 / (dt32 * dt31) * f3 * r3

    v2 = term1 + term2 + term3
    return v2

def select_triplet_combined(df_noisy, verbose=True):
    vectors = df_noisy[["px_km","py_km","pz_km"]].values.astype(float)
    t_abs = pd.to_datetime(df_noisy["timestamp"]).astype("int64").values / 1e9
    n = len(vectors)
    if verbose:
        print("select_triplet_combined: noisy samples:", n)

    if n < 5:
        raise RuntimeError("Not enough noisy samples to select triplet.")

    n_j = min(60, max(10, n//5))
    j_candidates = np.unique(np.round(np.linspace(2, n-3, n_j)).astype(int))
    offsets = [5, 10, 15, 20, 30]

    best_score = 1e12
    best_trip = None

    def coplanarity_measure(v1, v2, v3):
        cross = np.cross(v1, v2)
        dot = np.dot(v3, cross)
        nc = np.linalg.norm(cross)
        nv3 = np.linalg.norm(v3)
        if nc == 0 or nv3 == 0:
            return None
        return abs(dot / (nc * nv3))

    for j in j_candidates:
        for oi in offsets:
            for ok in offsets:
                i = j - oi
                k = j + ok
                if i < 0 or k >= n:
                    continue
                r1 = vectors[i]; r2 = vectors[j]; r3 = vectors[k]
                a12 = angle_between_deg(r1, r2)
                a23 = angle_between_deg(r2, r3)
                a13 = angle_between_deg(r1, r3)
                if a12 > 10.0 or a23 > 10.0 or a13 > 20.0:
                    continue
                C = coplanarity_measure(r1, r2, r3)
                if C is None:
                    continue
                theta = degrees(acos(np.clip(C, -1.0, 1.0)))
                alpha = abs(90.0 - theta)
                t1 = float(t_abs[i]); t2 = float(t_abs[j]); t3 = float(t_abs[k])
                v_hg = herrick_gibbs(r1, r2, r3, t1, t2, t3)
                el = orbital_elements_from_rv(r2, v_hg)
                if not (6300 < el["a"] < 7500):
                    continue
                if el["e"] > 0.2:
                    continue
                score = alpha * 1.0 + (a12 + a23 + a13) * 0.2 + el["e"] * 50.0 + abs(el["vmag"] - 7.6) * 5.0
                if score < best_score:
                    best_score = score
                    best_trip = {
                        "i": int(i), "j": int(j), "k": int(k),
                        "t1": t1, "t2": t2, "t3": t3,
                        "r1": r1, "r2": r2, "r3": r3,
                        "v_hg": v_hg, "el": el, "alpha": alpha, "a12": a12, "a23": a23, "a13": a13, "score": score
                    }

    if best_trip is None:
        print("select_triplet_combined: fallback to brute force.")
        vectors = df_noisy[["px_km","py_km","pz_km"]].values.astype(float)
        timestamps = pd.to_datetime(df_noisy["timestamp"]).astype("int64").values / 1e9
        best_alpha = float('inf'); best_triplet = None
        for (i, j, k) in itertools.combinations(range(n), 3):
            v1, v2, v3 = vectors[i], vectors[j], vectors[k]
            a12 = angle_between_deg(v1, v2)
            a23 = angle_between_deg(v2, v3)
            a13 = angle_between_deg(v1, v3)
            if a12 > 10.0 or a23 > 10.0 or a13 > 20.0:
                continue
            C = coplanarity_measure(v1, v2, v3)
            if C is None:
                continue
            theta = degrees(acos(np.clip(C, -1.0, 1.0)))
            alpha = abs(90.0 - theta)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triplet = (i, j, k)
        if best_triplet is None:
            raise RuntimeError("select_triplet_combined: fallback failed.")
        i,j,k = best_triplet
        t1 = float(timestamps[i]); t2 = float(timestamps[j]); t3 = float(timestamps[k])
        v_hg = herrick_gibbs(vectors[i], vectors[j], vectors[k], t1, t2, t3)
        return (i,j,k), (t1,t2,t3)

    print("SELECTED TRIPLET (combined):", (best_trip["i"], best_trip["j"], best_trip["k"]))
    print("  abs times (s):", (best_trip["t1"], best_trip["t2"], best_trip["t3"]))
    print("  HG |v|:", best_trip["el"]["vmag"], " a:", best_trip["el"]["a"], " e:", best_trip["el"]["e"])
    print("  pairwise angles (deg):", best_trip["a12"], best_trip["a23"], best_trip["a13"])
    print("  coplanarity alpha (deg):", best_trip["alpha"])
    print("  score:", best_trip["score"])
    return (best_trip["i"], best_trip["j"], best_trip["k"]), (best_trip["t1"], best_trip["t2"], best_trip["t3"])

@njit(cache=True)
def get_sun_position_numba(jd):
    n = jd - 2451545.0
    L = 280.460 + 0.9856474 * n
    g = 357.528 + 0.9856003 * n
    L = np.radians(L % 360.0)
    g = np.radians(g % 360.0)

    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2.0 * g)
    epsilon = np.radians(23.439 - 0.0000004 * n)

    r = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
    r_km = r * 149597870.7

    x = r_km * np.cos(lam)
    y = r_km * np.cos(epsilon) * np.sin(lam)
    z = r_km * np.sin(epsilon) * np.sin(lam)

    return x, y, z

@njit(cache=True)
def get_moon_position_numba(jd):
    T = (jd - 2451545.0) / 36525.0

    Lp = np.radians(218.3164477 + 481267.88123421 * T)
    Mp = np.radians(134.9633964 + 477198.8675055 * T)
    F = np.radians(93.2720950 + 483202.0175233 * T)

    lon = Lp + np.radians(6.289) * np.sin(Mp)
    lat = np.radians(5.128) * np.sin(F)

    r = 385000.56 - 20905.0 * np.cos(Mp)
    eps = np.radians(23.439)

    x_ecl = r * np.cos(lat) * np.cos(lon)
    y_ecl = r * np.cos(lat) * np.sin(lon)
    z_ecl = r * np.sin(lat)

    x = x_ecl
    y = y_ecl * np.cos(eps) - z_ecl * np.sin(eps)
    z = y_ecl * np.sin(eps) + z_ecl * np.cos(eps)

    return x, y, z

# ----------------------------
# Extended acceleration (kept, J2/J3/J4 supported)
# ----------------------------
@njit(cache=True)
def acceleration_extended_numba(t, state, epoch_jd,
                                H_SCALE, RHO_REF, H_REF, OMEGA_E, CD_A_M,
                                use_j2, use_j3, use_j4,
                                use_drag, use_sun, use_moon):

    x, y, z, vx, vy, vz = state

    # --- position norm ---
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2) + 1e-12

    # 2-body
    r3 = r2 * r
    a2 = -MU / (r3 + 1e-20)
    ax = a2 * x
    ay = a2 * y
    az = a2 * z

    # ratio
    ratio2 = (z/r)**2

    # ------------ J2 ------------
    if use_j2:
        c2 = 1.5 * J2 * MU * RE*RE / (r*r2*r2 + 1e-20)
        f = (1.0 - 5.0*ratio2)
        ax += -c2 * x * f
        ay += -c2 * y * f
        az += -c2 * z * (3.0 - 5.0*ratio2)

    # ------------ J3 ------------
    if use_j3:
        r7 = r2 * r2 * r3
        c3 = 0.5 * J3 * MU * RE*RE*RE / (r7 + 1e-20)
        ax += -5*c3*x*z*(3 - 7*ratio2)
        ay += -5*c3*y*z*(3 - 7*ratio2)
        az += -c3*(30*z*z - 35*(z*z*z*z)/(r2+1e-20) - 3*r2)

    # ------------ J4 ------------
    if use_j4:
        r7 = r2 * r2 * r3
        c4 = 0.125 * J4 * MU * RE**4 / (r7 + 1e-20)
        tx = 1 - 14*ratio2 + 21*ratio2*ratio2
        tz = 15 - 70*ratio2 + 63*ratio2*ratio2
        ax += 15*c4*x*tx
        ay += 15*c4*y*tx
        az += 5*c4*z*tz

    # ------------ Drag ------------
    if use_drag:
        h = r - RE
        rho = RHO_REF * np.exp(-(h - H_REF) / H_SCALE) * 1e9

        v_rx = vx + OMEGA_E*y
        v_ry = vy - OMEGA_E*x
        v_rz = vz
        v_rel = np.sqrt(v_rx*v_rx + v_ry*v_ry + v_rz*v_rz) + 1e-12

        drag = 0.5 * rho * (CD_A_M * 1e-6) * v_rel
        ax += -drag * v_rx
        ay += -drag * v_ry
        az += -drag * v_rz

    # ------------ Sun ------------
    if use_sun:
        jd = epoch_jd + t/86400.0
        xs, ys, zs = get_sun_position_numba(jd)
        dx = xs - x
        dy = ys - y
        dz = zs - z
        d2 = dx*dx + dy*dy + dz*dz
        d = np.sqrt(d2) + 1e-12
        mu_s = 1.32712440018e11
        ax += mu_s*(dx/(d*d*d) - xs/( (xs*xs+ys*ys+zs*zs)**1.5 ))
        ay += mu_s*(dy/(d*d*d) - ys/( (xs*xs+ys*ys+zs*zs)**1.5 ))
        az += mu_s*(dz/(d*d*d) - zs/( (xs*xs+ys*ys+zs*zs)**1.5 ))

    # ------------ Moon ------------
    if use_moon:
        jd = epoch_jd + t/86400.0
        xm, ym, zm = get_moon_position_numba(jd)
        dx = xm - x
        dy = ym - y
        dz = zm - z
        d2 = dx*dx + dy*dy + dz*dz
        d = np.sqrt(d2) + 1e-12
        mu_m = 4902.800066
        ax += mu_m*(dx/(d*d*d) - xm/( (xm*xm+ym*ym+zm*zm)**1.5 ))
        ay += mu_m*(dy/(d*d*d) - ym/( (xm*xm+ym*ym+zm*zm)**1.5 ))
        az += mu_m*(dz/(d*d*d) - zm/( (xm*xm+ym*ym+zm*zm)**1.5 ))

    return np.array([vx, vy, vz, ax, ay, az])

def propagate_cowell(initial_r, initial_v, t_eval, epoch_jd,use_H_SCALE,use_RHO_REF,use_H_REF,use_OMEGA_E,use_CD_AM,
                     use_j2=True, use_j3=True, use_j4=True,
                     use_drag=False, use_sun=False, use_moon=False):
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1:
        raise ValueError("t_eval must be 1-D")
    y0 = np.concatenate([np.asarray(initial_r, dtype=float), np.asarray(initial_v, dtype=float)])
    t_span = (float(t_eval[0]), float(t_eval[-1]))
    print(f"[Cowell] t_span={t_span}, samples={len(t_eval)}")
    sol = solve_ivp(lambda t, y: acceleration_extended_numba(t, y, epoch_jd,use_H_SCALE,use_RHO_REF,use_H_REF,use_OMEGA_E,use_CD_AM, use_j2, use_j3, use_j4, use_drag, use_sun, use_moon),
                    t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9, method="RK45")
    print("[Cowell] solve_ivp success:", sol.success, "status:", sol.status, "message:", sol.message)
    Y = np.asarray(sol.y, dtype=float)
    print("[Cowell] raw sol.y shape:", Y.shape)
    expected = (6, len(t_eval))
    if Y.ndim != 2 or Y.shape != expected:
        raise RuntimeError(f"Cowell solve_ivp returned invalid shape: {None if sol.y is None else sol.y.shape}. message: {sol.message}")
    return Y.T

def get_best_iod_triplet(df):
    vectors = df[["px_km", "py_km", "pz_km"]].values
    timestamps = pd.to_datetime(df["timestamp"]).astype("int64").values / 1e9  # convert to seconds

    n = len(vectors)

    ALPHA_THRESH = 0.001
    A12_THRESH = 1.0
    A23_THRESH = 1.0
    A13_THRESH = 3.0

    best_alpha = float('inf')
    best_triplet = None

    # Helper functions
    def angle(v1, v2):
        dot = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return None
        c = np.clip(dot / (n1 * n2), -1.0, 1.0)
        return degrees(acos(c))

    def coplanarity(v1, v2, v3):
        cross = np.cross(v1, v2)
        dot = np.dot(v3, cross)
        nc = np.linalg.norm(cross)
        nv3 = np.linalg.norm(v3)
        if nc == 0 or nv3 == 0:
            return None
        return abs(dot / (nc * nv3))

    # Main search
    for (i, j, k) in itertools.combinations(range(n), 3):
        v1, v2, v3 = vectors[i], vectors[j], vectors[k]

        A12 = angle(v1, v2)
        if A12 is None or A12 > A12_THRESH:
            continue

        A23 = angle(v2, v3)
        if A23 is None or A23 > A23_THRESH:
            continue

        A13 = angle(v1, v3)
        if A13 is None or A13 > A13_THRESH:
            continue

        C = coplanarity(v1, v2, v3)
        if C is None:
            continue

        theta = degrees(acos(np.clip(C, -1.0, 1.0)))
        alpha = abs(90.0 - theta)

        # Early exit
        if alpha <= ALPHA_THRESH:
            return (i, j, k), (timestamps[i], timestamps[j], timestamps[k])

        # Track minimum
        if alpha < best_alpha:
            best_alpha = alpha
            best_triplet = (i, j, k)

    # Return best if none satisfy perfect threshold
    return best_triplet, (timestamps[best_triplet[0]],
                          timestamps[best_triplet[1]],
                          timestamps[best_triplet[2]])

def get_best_iod_triplet_fast(df):
    vectors = df[["px_km", "py_km", "pz_km"]].values.astype(float)
    timestamps = pd.to_datetime(df["timestamp"]).astype("int64").values / 1e9

    n = len(vectors)

    ALPHA_THRESH = 0.001
    A12_THRESH = 1.0
    A23_THRESH = 1.0
    A13_THRESH = 3.0

    # Precompute norms and unit vectors
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / norms[:, None]

    best_alpha = float('inf')
    best_triplet = None

    # Precompute frequently used dot matrix
    dot_mat = np.einsum('ij,kj->ik', unit, unit)

    # Precompute cross product matrix magnitude (|v1 × v2|)
    cross_mat = np.linalg.norm(
        np.cross(vectors[:, None, :], vectors[None, :, :], axis=-1), axis=-1
    )

    # Main optimized loop
    for i, j, k in itertools.combinations(range(n), 3):

        # Angles via dot products (faster)
        A12 = np.degrees(np.arccos(np.clip(dot_mat[i, j], -1, 1)))
        if A12 > A12_THRESH:
            continue

        A23 = np.degrees(np.arccos(np.clip(dot_mat[j, k], -1, 1)))
        if A23 > A23_THRESH:
            continue

        A13 = np.degrees(np.arccos(np.clip(dot_mat[i, k], -1, 1)))
        if A13 > A13_THRESH:
            continue

        # Coplanarity value C = |dot(v3, (v1×v2))| / (|v3|*|v1×v2|)
        cp = cross_mat[i, j]
        if cp == 0:
            continue

        C = abs(np.dot(vectors[k], np.cross(vectors[i], vectors[j]))) / (norms[k] * cp)

        # theta = arccos(C)
        theta = degrees(acos(np.clip(C, -1, 1)))
        alpha = abs(90.0 - theta)

        # Early exit
        if alpha <= ALPHA_THRESH:
            return (i, j, k), (timestamps[i], timestamps[j], timestamps[k])

        # Track best
        if alpha < best_alpha:
            best_alpha = alpha
            best_triplet = (i, j, k)

    return best_triplet, (timestamps[best_triplet[0]],
                          timestamps[best_triplet[1]],
                          timestamps[best_triplet[2]])

@njit(fastmath=True)
def compute_triplet(dot_mat, norms, vectors, indices,first_best):

    ALPHA_THRESH = 0.001
    A12_THRESH = 1.0
    A23_THRESH = 1.0
    A13_THRESH = 3.0

    best_alpha = 1e9
    best_i = -1
    best_j = -1
    best_k = -1

    for idx in range(indices.shape[0]):
        i = indices[idx, 0]
        j = indices[idx, 1]
        k = indices[idx, 2]

        # --- ANGLE 12 ---
        x = dot_mat[i, j]
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        A12 = degrees(acos(x))
        if A12 > A12_THRESH:
            continue

        # --- ANGLE 23 ---
        x = dot_mat[j, k]
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        A23 = degrees(acos(x))
        if A23 > A23_THRESH:
            continue

        # --- ANGLE 13 ---
        x = dot_mat[i, k]
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        A13 = degrees(acos(x))
        if A13 > A13_THRESH:
            continue

        # Coplanarity
        c12 = np.cross(vectors[i], vectors[j])
        cp = np.sqrt(c12[0]**2 + c12[1]**2 + c12[2]**2)
        if cp == 0:
            continue

        dotv = vectors[k, 0] * c12[0] + vectors[k, 1] * c12[1] + vectors[k, 2] * c12[2]
        C = abs(dotv) / (norms[k] * cp)

        # Clip C
        if C < -1.0:
            C = -1.0
        elif C > 1.0:
            C = 1.0

        theta = degrees(acos(C))
        alpha = abs(90.0 - theta)

        if first_best and alpha <= ALPHA_THRESH:
            return i, j, k

        if alpha < best_alpha:
            best_alpha = alpha
            best_i, best_j, best_k = i, j, k

    return best_i, best_j, best_k

@njit(fastmath=True)
def compute_triplet_new(dot_mat, norms, vectors, indices, first_best):

    #ALPHA_THRESH = 0.001
    ALPHA_THRESH = 0.001
    A12_THRESH = 1.0
    A23_THRESH = 1.0
    A13_THRESH = 3.0

    N = indices.shape[0]

    # Arrays to store valid triplets with coplanarity alpha
    alphas = np.empty(N, dtype=np.float64)
    trip_i = np.empty(N, dtype=np.int64)
    trip_j = np.empty(N, dtype=np.int64)
    trip_k = np.empty(N, dtype=np.int64)

    count = 0
    # NEW: iteration counter
    iter_count = 0

    # ---------------------------------------------------------
    #   1) COPLANARITY FIRST, THEN ANGLE CHECK
    # ---------------------------------------------------------
    for idx in range(N):

        iter_count += 1   # <-- COUNT EACH ITERATION

        i = indices[idx, 0]
        j = indices[idx, 1]
        k = indices[idx, 2]

        # ---- Coplanarity ----
        c12 = np.cross(vectors[i], vectors[j])
        cp = np.sqrt(c12[0]**2 + c12[1]**2 + c12[2]**2)
        if cp == 0:
            continue

        dotv = vectors[k, 0]*c12[0] + vectors[k, 1]*c12[1] + vectors[k, 2]*c12[2]
        C = abs(dotv) / (norms[k] * cp)

        if C < -1.0: C = -1.0
        elif C > 1.0: C = 1.0

        theta = degrees(acos(C))
        alpha = abs(90.0 - theta)

        # ---- Now check vector ANGLES ----
        # ANGLE 12
        x = dot_mat[i, j]
        if x < -1.0: x = -1.0
        elif x > 1.0: x = 1.0
        A12 = degrees(acos(x))
        if A12 > A12_THRESH:
            continue

        # ANGLE 23
        x = dot_mat[j, k]
        if x < -1.0: x = -1.0
        elif x > 1.0: x = 1.0
        A23 = degrees(acos(x))
        if A23 > A23_THRESH:
            continue

        # ANGLE 13
        x = dot_mat[i, k]
        if x < -1.0: x = -1.0
        elif x > 1.0: x = 1.0
        A13 = degrees(acos(x))
        if A13 > A13_THRESH:
            continue

        # -------------------------------------------------
        # FIRST_BEST LOGIC — NOW CORRECT
        # -------------------------------------------------
        if first_best and alpha <= ALPHA_THRESH:
            print("TOTAL ITERATIONS:", iter_count)
            # Return ONLY after angle AND alpha check
            return i, j, k

        # -------------------------------------------------
        # Store for later sorting (if first_best = False)
        # -------------------------------------------------
        alphas[count] = alpha
        trip_i[count] = i
        trip_j[count] = j
        trip_k[count] = k
        count += 1

    # If finished loop without first_best
    print("TOTAL ITERATIONS:", iter_count)

    # ---------------------------------------------------------
    # 2) If NOT first_best, pick best alpha among angle-valid set
    # ---------------------------------------------------------
    if count == 0:
        return -1, -1, -1

    # Simple Numba-safe selection sort by alpha
    for a in range(count):
        min_idx = a
        for b in range(a+1, count):
            if alphas[b] < alphas[min_idx]:
                min_idx = b

        # swap alpha and triplet
        tmp = alphas[a]
        alphas[a] = alphas[min_idx]
        alphas[min_idx] = tmp

        ti = trip_i[a];  trip_i[a] = trip_i[min_idx];  trip_i[min_idx] = ti
        tj = trip_j[a];  trip_j[a] = trip_j[min_idx];  trip_j[min_idx] = tj
        tk = trip_k[a];  trip_k[a] = trip_k[min_idx];  trip_k[min_idx] = tk

    # Smallest alpha with all angle constraints
    return trip_i[0], trip_j[0], trip_k[0]

def get_best_iod_triplet_numba(df,first_best):
    vectors = df[["px_km", "py_km", "pz_km"]].values.astype(np.float64)
    timestamps = pd.to_datetime(df["timestamp"]).astype("int64").values / 1e9

    n = len(vectors)

    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / norms[:, None]

    dot_mat = (unit @ unit.T).astype(np.float64)

    # Pre-generate combinations once (Numba doesn't support itertools)
    combos = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)

    i, j, k = compute_triplet_new(dot_mat, norms, vectors, combos,first_best)

    return (i, j, k), (timestamps[i], timestamps[j], timestamps[k])
# ----------------- IOD METHODS -----------------


def gibbs_method(r1, r2, r3, mu=mu_earth):
    z12 = np.cross(r1, r2)
    z23 = np.cross(r2, r3)
    z31 = np.cross(r3, r1)
    N = np.linalg.norm(r1)*z23 + np.linalg.norm(r2)*z31 + np.linalg.norm(r3)*z12
    D = z12 + z23 + z31
    S = (np.linalg.norm(r2) - np.linalg.norm(r3))*r1 + \
        (np.linalg.norm(r3) - np.linalg.norm(r1))*r2 + \
        (np.linalg.norm(r1) - np.linalg.norm(r2))*r3
    # protect division by zero
    if np.linalg.norm(N) < 1e-12 or np.linalg.norm(D) < 1e-12:
        return np.zeros(3)
    v2 = np.sqrt(mu / (np.linalg.norm(N) * np.linalg.norm(D))) * (np.cross(D, r2)/np.linalg.norm(r2) + S)
    return v2

def herrick_gibbs_method(r1, r2, r3, t1, t2, t3, mu=mu_earth):
    """
    Herrick-Gibbs formula to compute velocity at r2.
    t1,t2,t3 are seconds (float) since some epoch (they can be absolute seconds).
    """
    dt21 = float(t2 - t1)
    dt31 = float(t3 - t1)
    dt32 = float(t3 - t2)
    # Avoid division by zero
    if abs(dt21) < 1e-9 or abs(dt31) < 1e-9 or abs(dt32) < 1e-9:
        return np.zeros(3)
    term1 = -dt32 / (dt21 * dt31) * np.array(r1, dtype=float)
    term2 = (dt32 - dt21) / (dt21 * dt32) * np.array(r2, dtype=float)
    term3 = dt21 / (dt32 * dt31) * np.array(r3, dtype=float)
    v2 = (term1 + term2 + term3) * np.sqrt(mu)
    return v2

# ----------------- TWO-BODY DYNAMICS -----------------
def propagate_two_body(r, v, dt, mu=mu_earth):
    r = np.array(r)
    v = np.array(v)
    r_norm = np.linalg.norm(r) + 1e-12
    a = -mu * r / r_norm**3
    r_new = r + v * dt
    v_new = v + a * dt
    return r_new, v_new

# ----------------- EKF -----------------
def ekf_filter(r_init, v_init, r_meas, t_data, Q, R, P0):
    n = len(r_meas)
    x = np.zeros((6, n))
    x[:, 0] = np.hstack([r_init, v_init])
    P = P0.copy()
    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    for k in range(1, n):
        dt = t_data[k] - t_data[k - 1]
        r_pred, v_pred = propagate_two_body(x[0:3, k - 1], x[3:6, k - 1], dt)
        x_pred = np.hstack([r_pred, v_pred])

        r_norm = np.linalg.norm(x[0:3, k - 1]) + 1e-12
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 0:3] = -mu_earth * (np.eye(3) / r_norm**3 - 3 * np.outer(x[0:3, k - 1], x[0:3, k - 1]) / r_norm**5) * dt

        P_pred = F @ P @ F.T + Q

        z = r_meas[k]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x[:, k] = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred

    return x

# ----------------- COMPUTE RIC -----------------
def compute_ric(r_ref, r_target, v_ref):
    r_ref = np.array(r_ref, dtype=float)
    v_ref = np.array(v_ref, dtype=float)
    r_target = np.array(r_target, dtype=float)

    r_norm = np.linalg.norm(r_ref) + 1e-12
    r_hat = r_ref / r_norm

    # in-track: component of v_ref perpendicular to r_hat
    i_temp = v_ref - np.dot(v_ref, r_hat) * r_hat
    if np.linalg.norm(i_temp) < 1e-12:
        # fallback pick an arbitrary in-track
        if abs(r_ref[2]) < 0.9 * r_norm:
            i_hat = np.cross([0,0,1], r_hat)
        else:
            i_hat = np.cross([1,0,0], r_hat)
        i_hat = i_hat / (np.linalg.norm(i_hat) + 1e-12)
    else:
        i_hat = i_temp / (np.linalg.norm(i_temp) + 1e-12)

    c_hat = np.cross(r_hat, i_hat)

    delta_r = r_target - r_ref
    R = np.dot(delta_r, r_hat)
    I = np.dot(delta_r, i_hat)
    C = np.dot(delta_r, c_hat)
    return np.array([R, I, C])

def eci_to_ric(r_ref, v_ref, r_obj, v_obj):
    diff_eci = r_obj - r_ref
    ric_diffs = []
    for i in range(len(r_ref)):
        r = r_ref[i]
        v = v_ref[i]

        # Unit vectors
        u_r = r / (np.linalg.norm(r) + 1e-12)
        h = np.cross(r, v)
        u_c = h / (np.linalg.norm(h) + 1e-12)
        u_i = np.cross(u_c, u_r)

        M = np.vstack((u_r, u_i, u_c))
        d_ric = M @ diff_eci[i]
        ric_diffs.append(d_ric)

    return np.array(ric_diffs)

# ----------------- COWELL PROPAGATION (basic two-body used in original) -----------------
def cowell_equations(t, y, mu=mu_earth):
    r = y[0:3]
    v = y[3:6]
    r_norm = np.linalg.norm(r) + 1e-12
    a = -mu * r / r_norm**3
    # ---- J2 perturbation (comment out to disable) ----
    J2 = 1.08262668e-3
    Re = 6378.137  # Earth radius (km)
    x, y_, z = r
    r2 = r_norm**2
    factor = 1.5 * J2 * mu * (Re**2) / (r_norm**5)
    a_J2 = factor * np.array([
        x * (5*(z*z)/r2 - 1),
        y_ * (5*(z*z)/r2 - 1),
        z * (5*(z*z)/r2 - 3)
    ])
    a += a_J2
    return np.hstack([v, a])

# ----------------- Helper: parse ISO timestamp -> seconds since epoch and jday -----------------
def parse_iso_timestamp_to_dt(ts_str):
    """
    Parse ISO timestamp string (with timezone) to a timezone-aware datetime in UTC.
    """
    try:
        # handles formats like '2025-11-19T05:01:25.555453+00:00'
        dt = datetime.datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # assume UTC if no timezone
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        dt_utc = dt.astimezone(datetime.timezone.utc)
        return dt_utc
    except Exception:
        # fallback: try pandas
        try:
            dt = pd.to_datetime(ts_str)
            if dt.tzinfo is None:
                dt = dt.tz_localize('UTC')
            return dt.tz_convert('UTC')
        except Exception:
            raise

def datetime_to_jd_fr(dt_utc):
    """Convert timezone-aware UTC datetime to (jd, fr) for Satrec.sgp4"""
    # dt_utc must be timezone-aware in UTC
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
    dt_utc = dt_utc.astimezone(datetime.timezone.utc)
    year = dt_utc.year
    month = dt_utc.month
    day = dt_utc.day
    hour = dt_utc.hour
    minute = dt_utc.minute
    second = dt_utc.second + dt_utc.microsecond * 1e-6
    jd, fr = jday(year, month, day, hour, minute, second)
    return jd, fr

def sgp4_rv_at_datetime(dt_utc):
    """Return (r, v) from SGP4 at a timezone-aware UTC datetime. Returns (None, None) on failure."""
    if SAT is None:
        return None, None
    try:
        jd, fr = datetime_to_jd_fr(dt_utc)
        e, r, v = SAT.sgp4(jd, fr)
        if e != 0:
            return None, None
        return np.array(r, dtype=float), np.array(v, dtype=float)
    except Exception:
        return None, None
    
# ----------------------------
# Two-body for EKF prediction
# ----------------------------
def two_body_deriv(t, y):
    r = y[0:3]
    v = y[3:6]
    rnorm = np.linalg.norm(r) + 1e-12
    a = -MU * r / (rnorm**3)
    return np.hstack((v, a))

def propagate_two_body(r0, v0, dt):
    y0 = np.hstack((r0, v0))
    if abs(dt) < 1e-9:
        return np.array(r0), np.array(v0)
    max_step = 60.0
    sol = solve_ivp(two_body_deriv, (0.0, dt), y0, rtol=1e-9, atol=1e-9, method='RK45', max_step=max_step)
    yf = sol.y[:, -1]
    return yf[0:3], yf[3:6]

# ----------------------------
# EKF filter (starts at provided r_init time and filters forward)
# returns (N_filtered, 6)
# ----------------------------
def ekf_filter_starting_at(r_init, v_init, r_meas, t_data, Q, R, P0, verbose=True):
    r_meas = np.asarray(r_meas, dtype=float)
    M = len(r_meas)
    if M < 1:
        raise ValueError("ekf_filter_starting_at: no measurements")
    X = np.zeros((M, 6), dtype=float)
    X[0, 0:3] = r_init
    X[0, 3:6] = v_init
    P = P0.copy()
    H = np.hstack([np.eye(3), np.zeros((3, 3))])  # 3x6

    if verbose:
        print("[EKF] starting at provided epoch with M =", M)
        print("[EKF] r_init:", r_init, "v_init:", v_init)

    for k in range(1, M):
        dt = float(t_data[k] - t_data[k - 1])
        if dt <= 0:
            dt = 1.0
        r_prev = X[k - 1, 0:3].copy()
        v_prev = X[k - 1, 3:6].copy()
        r_pred, v_pred = propagate_two_body(r_prev, v_prev, dt)
        x_pred = np.hstack((r_pred, v_pred))

        # linearized transition
        rnorm = np.linalg.norm(r_prev) + 1e-12
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        mu = MU
        I3 = np.eye(3)
        rr = r_prev.reshape(3, 1)
        term = (mu / (rnorm**3)) * I3 - 3 * mu * (rr @ rr.T) / (rnorm**5)
        F[3:6, 0:3] = -term * dt

        P_pred = F @ P @ F.T + Q

        z = r_meas[k]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pred @ H.T @ S_inv
        X[k, :] = x_pred + (K @ y)
        P = (np.eye(6) - K @ H) @ P_pred

        if verbose and (k % max(1, M//10) == 0):
            print(f"[EKF] progress {k}/{M} dt={dt:.1f} pos_res_norm={np.linalg.norm(y):.6f}")

    if verbose:
        print("[EKF] finished. final state:", X[-1])
    return X

# ----------------- GUI -----------------
class IODApp:
    def __init__(self, master):
        self.master = master
        master.title("DPSS IOD + EKF + RIC + Cowell (Herrick-Gibbs + SGP4)")
        master.geometry("1400x900")

        # -------- Controls Frame --------
        ctrl = ttk.Frame(master)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Row 1: File selection + Run buttons
        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Select Noisy CSV", command=self.load_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(row1, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Select Clean CSV (RIC)", command=self.load_clean_file).pack(side=tk.LEFT, padx=5)
        self.clean_file_label = ttk.Label(row1, text="No clean file selected")
        self.clean_file_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Select Clean 24hr CSV", command=self.load_clean_24hr_file).pack(side=tk.LEFT, padx=5)
        self.clean_24hr_label = ttk.Label(row1, text="No 24hr file selected")
        self.clean_24hr_label.pack(side=tk.LEFT, padx=5)
        # --- NEW CHECKBUTTON HERE ---
        self.first_best_var = tk.BooleanVar(value=True)   # default ON (1)
        ttk.Checkbutton(row1, text="First Best", variable=self.first_best_var).pack(side=tk.LEFT, padx=10)
        ttk.Button(row1, text="Run IOD + EKF", command=self.run_iod).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Run Cowell Propagation", command=self.run_cowell).pack(side=tk.LEFT, padx=5)

        # --- New Cowell initial state selector ---
        self.cowell_source_var = tk.StringVar(value="EKF")
        ttk.Label(row1, text="Cowell start from:").pack(side=tk.LEFT, padx=5)

        self.cowell_source_box = ttk.Combobox(
            row1,
            textvariable=self.cowell_source_var,
            values=["EKF", "IOD"],
            state="readonly",
            width=10
        )
        self.cowell_source_box.pack(side=tk.LEFT, padx=5)

        ttk.Button(row1, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="Save File", command=self.save_states_to_csv).pack(side=tk.LEFT, padx=5)

        # Row 2: Checkboxes
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=2)
        self.show_measured_var = tk.BooleanVar(value=True)
        self.show_iod_points_var = tk.BooleanVar(value=True)
        self.show_ekf_var = tk.BooleanVar(value=True)
        self.show_ric_var = tk.BooleanVar(value=True)
        self.show_ric_error_var = tk.BooleanVar(value=True)
        self.show_cowell_var = tk.BooleanVar(value=True)
        self.ekf_run_var = tk.BooleanVar(value=True)
        self.show_clean_24hr_var = tk.BooleanVar(value=True)

        

        tk.Checkbutton(row2, text="Show Measured Orbit", variable=self.show_measured_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Show IOD Points", variable=self.show_iod_points_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Show EKF Orbit", variable=self.show_ekf_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Show Clean Orbit", variable=self.show_ric_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Show RIC Error", variable=self.show_ric_error_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Show Cowell Orbit", variable=self.show_cowell_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Show Clean 24hr Orbit", variable=self.show_clean_24hr_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row2, text="Run EKF", variable=self.ekf_run_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)

        # Row 3: Cowell Force Model Toggles
        row3 = ttk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=2)

        self.use_j2_var   = tk.BooleanVar(value=True)
        self.use_j3_var   = tk.BooleanVar(value=True)
        self.use_j4_var   = tk.BooleanVar(value=True)
        self.use_drag_var = tk.BooleanVar(value=True)
        self.use_sun_var  = tk.BooleanVar(value=True)
        self.use_moon_var = tk.BooleanVar(value=True)

        ttk.Label(row3, text="Cowell Forces:").pack(side=tk.LEFT, pady=3)

        tk.Checkbutton(row3, text="J2",   variable=self.use_j2_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row3, text="J3",   variable=self.use_j3_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row3, text="J4",   variable=self.use_j4_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row3, text="Drag", variable=self.use_drag_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row3, text="Sun",  variable=self.use_sun_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(row3, text="Moon", variable=self.use_moon_var).pack(side=tk.LEFT, padx=5)

         # --- Drag Model Parameters (Right of Cowell Forces) ---
        # --- Drag Model Parameters inline (horizontal layout) ---
        drag_row = ttk.Frame(row3)
        drag_row.pack(side=tk.LEFT, padx=20)

        # Variables
        self.H_SCALE_var = tk.DoubleVar(value=60.0)
        self.RHO_REF_var = tk.DoubleVar(value=1e-13)
        self.H_REF_var = tk.DoubleVar(value=500)
        self.OMEGA_E_var = tk.DoubleVar(value=7.2921159e-5)
        self.B_STAR_var = tk.DoubleVar(value=0.00022108)
        self.CD_A_M_var = tk.DoubleVar(value=0.012)

        

        # Helper to add label+entry inline
        def add_param(parent, text, var):
            ttk.Label(parent, text=f"{text}:").pack(side=tk.LEFT, padx=3)
            ttk.Entry(parent, textvariable=var, width=8).pack(side=tk.LEFT, padx=3)

        add_param(drag_row, "H_SCALE", self.H_SCALE_var)
        add_param(drag_row, "RHO_REF", self.RHO_REF_var)
        add_param(drag_row, "H_REF", self.H_REF_var)
        add_param(drag_row, "OMEGA_E", self.OMEGA_E_var)
        add_param(drag_row, "B_STAR", self.B_STAR_var)
        add_param(drag_row, "CD_A_M", self.CD_A_M_var)

        

         # --- EKF Parameter Row (Horizontal) ---
        ekf_row = ttk.Frame(ctrl)
        ekf_row.pack(fill=tk.X, pady=4)

        ttk.Label(ekf_row, text="EKF Parameters:").pack(side=tk.LEFT, padx=10)

        # EKF Variables
        self.fact_var = tk.DoubleVar(value=1.0)

        self.P0_r_var = tk.DoubleVar(value=25)   # position covariance
        self.P0_v_var = tk.DoubleVar(value=0.01)     # velocity covariance

        self.Q_r_var  = tk.DoubleVar(value=0.0001)   # process noise (pos)
        self.Q_v_var  = tk.DoubleVar(value=1e-6)    # process noise (vel)

        self.R_r_var    = tk.DoubleVar(value=10)     # measurement noise
        self.R_a_var    = tk.DoubleVar(value=5)     # measurement noise
        self.R_e_var    = tk.DoubleVar(value=5)     # measurement noise

        self.use_user_input_var = tk.BooleanVar(value=True)

        # Helper: inline label + entry
        def add_ekf_param(parent, text, var):
            ttk.Label(parent, text=f"{text}:").pack(side=tk.LEFT, padx=3)
            ttk.Entry(parent, textvariable=var, width=8).pack(side=tk.LEFT, padx=3)

        add_ekf_param(ekf_row, "fact", self.fact_var)
        add_ekf_param(ekf_row, "P0_r", self.P0_r_var)
        add_ekf_param(ekf_row, "P0_v", self.P0_v_var)
        add_ekf_param(ekf_row, "Q_r", self.Q_r_var)
        add_ekf_param(ekf_row, "Q_v", self.Q_v_var)
        add_ekf_param(ekf_row, "R_r", self.R_r_var)
        add_ekf_param(ekf_row, "R_a", self.R_a_var)
        add_ekf_param(ekf_row, "R_e", self.R_e_var)

        # Checkbox
        ttk.Checkbutton(
            ekf_row,
            text="User Input",
            variable=self.use_user_input_var,
            command=self.update_R_parameters
        ).pack(side=tk.LEFT, padx=10)

        


        # Main resizable panes: left = 3D orbit, right = (output box over 2D RIC error)
        self.main_paned = tk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left frame (3D orbit + toolbar)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, minsize=600)

        # Right side is a vertical paned window for output (top) and 2D plot (bottom)
        self.right_paned = tk.PanedWindow(self.main_paned, orient=tk.VERTICAL)
        self.main_paned.add(self.right_paned, minsize=300)

        # 3D Orbit Figure in left frame
        self.fig = plt.figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Orbit 3D Visualization")
        self.ax.set_xlabel("X (km)")
        self.ax.set_ylabel("Y (km)")
        self.ax.set_zlabel("Z (km)")
        self.canvas = FigureCanvasTkAgg(self.fig, self.left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.left_frame)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)
        self.canvas.get_tk_widget().bind("<Button-1>", lambda e: self._on_tk_click_3(e), add="+")

        # Top-right: Output box
        self.output_frame = ttk.Frame(self.right_paned)
        self.output_box = tk.Text(self.output_frame, height=10)
        self.output_box.pack(fill=tk.BOTH, expand=True)
        self.legend_ctrl_frame = ttk.Frame(self.output_frame)
        self.legend_ctrl_frame.pack(fill=tk.X, pady=4)
        self._checkbox_map = {}
        self._checkbox_order = []
        self.right_paned.add(self.output_frame, minsize=150)

        # Bottom-right: 2D RIC Error Figure
        self.bottom_frame = ttk.Frame(self.right_paned)
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 3))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.bottom_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.bottom_frame)
        self.toolbar2.update()
        self.toolbar2.pack(fill=tk.X)
        self.right_paned.add(self.bottom_frame, minsize=150)
        self.canvas2.get_tk_widget().bind("<Button-1>", lambda e: self._on_tk_click_2(e), add="+")

        # Bottom-right: 2D RIC Error Figure (SCROLLABLE)
        # Bottom-right: 2D RIC Error Figure (scrollable correctly)
        # -------------------------


        self.left_frame.bind("<Configure>", lambda e: self.canvas.draw())
        self.bottom_frame.bind("<Configure>", lambda e: self.canvas2.draw())

        # Data placeholders
        self.data = None
        self.clean_data = None
        self.clean_24hr_data = None
        self.r_data = None
        self.clean24_abs = None
        self.clean5_abs = None
        self.noisy_abs = None
        self.t_meas_for_ekf_abs=None
        self.j=None
        self.t_j_abs=None
        self.r2 = None
        self.v2 = None
        self.v_hg=None
        self.x_ekf = None
        self.ekf_end_abs_time=None
        self.cowell_states=None
        self.cowell_r = None
        self.cowell_v = None
        self.cowell_t = None
        self.t_data_full = None
        self._legend3 = None
        self.worker_thread = None
        self.ric_cowell=None
        self.ric_ekf=None
        self.worker_queue = queue.Queue()
        self.poll_interval = 100   # ms
        self._legend3_map = {}
        self._legend3_order = []

    def update_R_parameters(self):
        """
        Called whenever the 'User Input' checkbox is toggled or whenever
        Range/Azimuth/Elevation std values are computed.
        """

        use_user = self.use_user_input_var.get()

        if use_user:
            # User input mode: do not overwrite
            self.output_box.insert(tk.END, f"R:{self.R_r_var.get()} A:{self.R_a_var.get()} B:{self.R_e_var.get()}\n")
            print("R:",self.R_r_var.get()," A:",self.R_a_var.get()," E:",self.R_e_var.get())
            return
    
        # Auto-update using computed values IF they are not None
        if self.Range_std is not None:
            self.R_r_var.set(self.Range_std)

        if self.Azimuth_std is not None:
            self.R_a_var.set(self.Azimuth_std)

        if self.Elevation_std is not None:
            self.R_e_var.set(self.Elevation_std)

        self.output_box.insert(tk.END, f"R:{self.R_r_var.get()} A:{self.R_a_var.get()} B:{self.R_e_var.get()}\n")
        print("R:",self.R_r_var.get()," A:",self.R_a_var.get()," E:",self.R_e_var.get())
    
    #To clear output logs
    def clear_log(self):
        """Clears the output_text log window."""
        self.output_box.delete("1.0", tk.END)
    # ----------------- LOAD FILES -----------------
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.file_label.config(text=file_path.split("/")[-1])
                messagebox.showinfo("Success", "Noisy file loaded successfully!")
                # --- Proper indexing using .iloc ---
                rfirst_x = self.data["px_km"].iloc[0]
                rfirst_y = self.data["py_km"].iloc[0]
                rfirst_z = self.data["pz_km"].iloc[0]

                rlast_x = self.data["px_km"].iloc[-1]
                rlast_y = self.data["py_km"].iloc[-1]
                rlast_z = self.data["pz_km"].iloc[-1]

                # --- Convert to vectors ---
                r1 = np.array([rfirst_x, rfirst_y, rfirst_z], dtype=float)
                r2 = np.array([rlast_x, rlast_y, rlast_z], dtype=float)

                # --- Compute angle between r1 and r2 in degrees ---
                dot_val = np.dot(r1, r2)
                norms = np.linalg.norm(r1) * np.linalg.norm(r2)

                # Avoid division issues
                if norms == 0:
                    angle_deg = 0.0
                else:
                    cos_theta = np.clip(dot_val / norms, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_theta))

                print("Angle between first and last position vectors (deg):", angle_deg)
                self.output_box.insert(tk.END, f"Angle between first and last position vectors (deg): {angle_deg}\n")
                try:
                    df = self.data
                    required_cols = ["range_km", "azimuth_deg", "elevation_deg"]
                    if not all(col in df.columns for col in required_cols):
                        messagebox.showerror("Error", f"CSV missing required columns: {required_cols}")
                        return
                    sigma_r, sigma_az, sigma_el = estimate_noise_from_spherical(df)
                    # Round to scientific notation with 3 decimals
                    sigma_r = float(f"{sigma_r:.3e}")
                    sigma_az = float(f"{sigma_az:.3e}")
                    sigma_el = float(f"{sigma_el:.3e}")
                    print("R:",sigma_r," A:",sigma_az," E:",sigma_el)
                    self.Range_std=sigma_r
                    self.Azimuth_std=sigma_az
                    self.Elevation_std=sigma_el
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process CSV:\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def load_clean_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.clean_data = pd.read_csv(file_path)
                self.clean_file_label.config(text=file_path.split("/")[-1])
                messagebox.showinfo("Success", "Clean file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def load_clean_24hr_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.clean_24hr_data = pd.read_csv(file_path)
                self.clean_24hr_label.config(text=file_path.split("/")[-1])
                messagebox.showinfo("Success", "Clean 24hr file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def _poll_worker_queue(self):
        try:
            msg, data = self.worker_queue.get_nowait()
        except queue.Empty:
            if self.worker_thread and self.worker_thread.is_alive():
                self.master.after(self.poll_interval, self._poll_worker_queue)
            return

        if msg == "done":
            self.master.config(cursor="")
            self.master.update_idletasks()
            duration = time.time() - self.thread_start_time
            self.output_box.insert(tk.END, f"Time taken: {duration:.2f}s\n")
            self.output_box.insert(tk.END, f"[Cowell] Time taken to complete cowell propagation:{duration}\n")

            cowell_states = data
            self.cowell_states = cowell_states
            # self.output_box.insert(tk.END, f"Cowell propagation ended.\n")
            self.output_box.insert(tk.END, f"Cowell output shape:{cowell_states.shape}.\n")
            print("[COWELL] states shape:", cowell_states.shape)

            # -------------------------------
            # DO ALL POST-PROCESSING HERE
            # -------------------------------
            self.output_box.insert(tk.END, "Cowell propagation ended.\n")
            self.output_box.insert(tk.END, f"Cowell output shape: {cowell_states.shape}\n")

            np.savetxt("cowell_states_output_from_ekf_final1.csv",
                    cowell_states, delimiter=",",
                    header="px,py,pz,vx,vy,vz", comments="")

            print("[COWELL] saved cowell_states_output_from_ekf_final1.csv")

            self.update_plot()

        elif msg == "error":
            print("Cowell ERROR:", data)
            self.master.config(cursor="")
            self.master.update_idletasks()

        self.worker_thread = None

    def run_cowell_threaded(self, r0, v0, t_eval, epoch_jd,
                        use_H_SCALE, use_RHO_REF, use_H_REF,
                        use_OMEGA_E, use_CD_AM,
                        use_j2, use_j3, use_j4,
                        use_drag, use_sun, use_moon):

        # This is the worker executed in the background
        # Enable busy cursor
        
        def worker():
            try:
                cowell_states = propagate_cowell(
                    r0, v0, t_eval, epoch_jd,
                    use_H_SCALE, use_RHO_REF, use_H_REF, use_OMEGA_E, use_CD_AM,
                    use_j2, use_j3, use_j4,
                    use_drag, use_sun, use_moon
                )
                # Put result into queue for main thread
                self.worker_queue.put(("done", cowell_states))
            except Exception as e:
                self.worker_queue.put(("error", str(e)))

        # Start thread
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

        # Start polling
        self.master.after(self.poll_interval, self._poll_worker_queue)

    def save_states_to_csv(self):
    # Check if EKF results exist
        if self.x_ekf is None:
            messagebox.showinfo("Error", "EKF states are not available!")
            return

        if self.cowell_states is None:
            messagebox.showinfo("Error", "Cowell states are not available!")
            return
        
        if self.ric_ekf is None:
            messagebox.showinfo("Error", "RIC-EKF values are not available!")
            return
        
        if self.ric_cowell is None:
            messagebox.showinfo("Error", "RIC-Cowell values are not available!")
            return

        folder = filedialog.askdirectory(title="Select folder to save CSV files")
        if not folder:
            return

        try:
            df_ekf = pd.DataFrame(self.x_ekf, columns=["px","py","pz","vx","vy","vz"])
            df_ekf.to_csv(f"{folder}/ekf_states.csv", index=False)

            df_cowell = pd.DataFrame(self.cowell_states, columns=["px","py","pz","vx","vy","vz"])
            df_cowell.to_csv(f"{folder}/cowell_states.csv", index=False)

            df_ric_ekf = pd.DataFrame(self.ric_ekf, columns=["dR","dI","dC"])
            df_ric_ekf.to_csv(f"{folder}/RIC_EKF.csv", index=False)

            df_ric_cowell= pd.DataFrame(self.ric_cowell, columns=["dR","dI","dC"])
            df_ric_cowell.to_csv(f"{folder}/RIC_Cowell.csv", index=False)



            messagebox.showinfo("Success", "Files saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))


    # ----------------- RUN IOD -----------------
    def run_iod(self):
        self.update_R_parameters()
        
        if self.data is None:
            messagebox.showerror("Error", "Load a noisy data file first.")
            return
        df_clean24 = self.clean_24hr_data
        df_clean5 = self.clean_data
        df_noisy = self.data

        def to_unix(series):
            return pd.to_datetime(series).astype("int64").values / 1e9
        
        clean24_abs = to_unix(df_clean24["timestamp"])
        clean5_abs = to_unix(df_clean5["timestamp"])
        noisy_abs = to_unix(df_noisy["timestamp"])

        self.noisy_abs=noisy_abs
        self.clean5_abs=clean5_abs
        self.clean24_abs=clean24_abs

        print("[LOAD] timestamps parsed to unix seconds.")
        # Select triplet from noisy
        print("[IOD] Selecting IOD triplet from noisy...")
        self.output_box.insert(tk.END, f"[IOD] Selecting IOD triplet from noisy...\n")
        #(i_idx, j_idx, k_idx), (t_i_abs, t_j_abs, t_k_abs) = select_triplet_combined(df_noisy, verbose=True)
        s_time=time.time()
        first_best=bool(self.first_best_var.get())
        (i_idx, j_idx, k_idx), (t_i_abs, t_j_abs, t_k_abs) = get_best_iod_triplet_numba(df_noisy,first_best)
        if i_idx==-1 and j_idx==-1 and k_idx==-1:
            messagebox.showerror("No Triplet found with given criteria.")
        e_time=time.time()
        print(f"[IOD] Time taken to calculate the points: {e_time-s_time}\n")
        self.output_box.insert(tk.END, f"[IOD] Time taken to calculate the points: {e_time-s_time}\n")
        self.j=j_idx
        self.t_j_abs=t_j_abs

        print("[IOD] Selected indices:", (i_idx, j_idx, k_idx))
        self.output_box.insert(tk.END, f"[IOD] IOD points index i:{i_idx},j:{j_idx},k:{k_idx}\n")
        print("[IOD] Selected times (abs secs):", (t_i_abs, t_j_abs, t_k_abs))

        r1 = df_noisy.loc[i_idx, ["px_km","py_km","pz_km"]].values.astype(float)
        r2 = df_noisy.loc[j_idx, ["px_km","py_km","pz_km"]].values.astype(float)
        r3 = df_noisy.loc[k_idx, ["px_km","py_km","pz_km"]].values.astype(float)
        self.output_box.insert(tk.END, f"[IOD] IOD point vector r1:{r1}\n")
        self.output_box.insert(tk.END, f"[IOD] IOD point vector r2:{r2}\n")
        self.output_box.insert(tk.END, f"[IOD] IOD point vector r3:{r3}\n")
        print("[IOD] r1:", r1)
        print("[IOD] r2:", r2)
        print("[IOD] r3:", r3)
        # Coplanarity
        c12 = np.cross(r1, r2)
        cp = np.sqrt(c12[0]**2 + c12[1]**2 + c12[2]**2)
        dotv = r3[0] * c12[0] + r3[1] * c12[1] + r3[2] * c12[2]
        mag = np.linalg.norm(r3)
        C = abs(dotv) / (mag * cp)
        
        # Clip C
        if C < -1.0:
            C = -1.0
        elif C > 1.0:
            C = 1.0

        theta = degrees(acos(C))
        alpha = abs(90.0 - theta)
        self.output_box.insert(tk.END, f"[IOD] Coplanarity angle of triplet:{alpha}\n")
        self.r2=r2
        # HG velocity at r2
        v_hg = herrick_gibbs(r1, r2, r3, t_i_abs, t_j_abs, t_k_abs)
        self.v_hg=v_hg
        print("[HG] v_hg:", v_hg)
        self.output_box.insert(tk.END, f"Herrick Gibbs velocity v_hg:{v_hg}\n")
        el_hg = orbital_elements_from_rv(r2, v_hg)
        print("[HG] |r|={:.3f} |v|={:.6f} a={:.3f} e={:.6f}".format(el_hg["rmag"], el_hg["vmag"], el_hg["a"], el_hg["e"]))

        # EKF tuning
        # fact = 1.0
        # P0 = fact * np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
        # Q = fact * np.diag([0.001, 0.001, 0.001, 1e-7, 1e-7, 1e-7])
        # R = fact * np.diag([0.1, 0.1, 0.1])

        fact=self.fact_var.get()

        P0_r=self.P0_r_var.get()   # position covariance
        P0_v=self.P0_v_var.get()     # velocity covariance

        Q_r=self.Q_r_var.get()  # process noise (pos)
        Q_v=self.Q_v_var.get()   # process noise (vel)

        R_r=self.R_r_var.get()
        R_a=self.R_a_var.get()
        R_e=self.R_e_var.get()    # measurement noise

        #fact = fact
        P0 = fact * np.diag([P0_r, P0_r, P0_r, P0_v, P0_v, P0_v])
        Q = fact * np.diag([Q_r, Q_r, Q_r, Q_v, Q_v, Q_v])
        R = fact * np.diag([R_r, R_a, R_e])

        # Build measurement arrays and times (full noisy)
        r_meas_full = df_noisy[["px_km","py_km","pz_km"]].values.astype(float)
        noisy_times_full = noisy_abs  # absolute times array

        # EKF must start at j_idx (r2) and filter forward only
        print(f"[EKF] Starting EKF at noisy index j_idx={j_idx} (r2).")
        self.output_box.insert(tk.END, f"[EKF] Starting EKF at noisy index j_idx={j_idx} (r2).\n")
        r_meas_for_ekf = r_meas_full[j_idx:].copy()
        t_meas_for_ekf_abs = noisy_times_full[j_idx:].copy()  # absolute seconds
        self.t_meas_for_ekf_abs=t_meas_for_ekf_abs
        # make relative times starting at r2 epoch (t_rel[0] = 0)
        print("time from r2 length",len(t_meas_for_ekf_abs))
        t_meas_for_ekf_rel = t_meas_for_ekf_abs - t_meas_for_ekf_abs[0]

        print(f"[EKF] number of measurements for EKF: {len(r_meas_for_ekf)}")
        print(f"[EKF] first few rel times (s): {t_meas_for_ekf_rel[:5]}")

        # Run EKF with initial state r2, v_hg at time t_meas_for_ekf_abs[0]
        x_ekf = ekf_filter_starting_at(r2, v_hg, r_meas_for_ekf, t_meas_for_ekf_rel, Q, R, P0, verbose=True)
        print("[EKF] finished. EKF output shape:", x_ekf.shape)
        self.output_box.insert(tk.END, f"[EKF] finished. EKF output shape:{x_ekf.shape}\n")
        # --- EKF END POSITION ---
        self.output_box.insert(tk.END,
            f"[EKF] End position of ekf is:[{x_ekf[-1][0]},{x_ekf[-1][1]},{x_ekf[-1][2]}]\n"
        )
        self.output_box.insert(tk.END,
            f"[EKF] End velocity of ekf is:[{x_ekf[-1][3]},{x_ekf[-1][4]},{x_ekf[-1][5]}]\n"
        )

        # ----------------------------------------------------------
        #  COMPUTE END VELOCITY OF NOISY DATA FROM TIMESTAMPS
        # ----------------------------------------------------------

        # Extract last two timestamps and convert to datetime
        t_last = pd.to_datetime(df_noisy["timestamp"].iloc[-1])
        t_prev = pd.to_datetime(df_noisy["timestamp"].iloc[-2])

        # Time difference in seconds
        dt = (t_last - t_prev).total_seconds()

        # Last two positions
        x_last,  y_last,  z_last  = df_noisy.iloc[-1][["px_km","py_km","pz_km"]]
        x_prev,  y_prev,  z_prev  = df_noisy.iloc[-2][["px_km","py_km","pz_km"]]

        # Velocity = Δr / Δt
        vx_end_noisy = (x_last - x_prev) / dt
        vy_end_noisy = (y_last - y_prev) / dt
        vz_end_noisy = (z_last - z_prev) / dt

        # --- Noisy end position ---
        self.output_box.insert(tk.END,
            f"[EKF] End position of noisy data is:[{x_last},{y_last},{z_last}]\n"
        )

        # --- Noisy end velocity ---
        self.output_box.insert(tk.END,
            f"[EKF] End velocity of noisy data is:[{vx_end_noisy},{vy_end_noisy},{vz_end_noisy}]\n"
        )

        # ----------------------------------------------------------
        #  DIFFERENCES BETWEEN EKF AND NOISY (POSITION & VELOCITY)
        # ----------------------------------------------------------

        diff_x = abs(x_ekf[-1][0] - x_last)
        diff_y = abs(x_ekf[-1][1] - y_last)
        diff_z = abs(x_ekf[-1][2] - z_last)

        diff_vx = abs(x_ekf[-1][3] - vx_end_noisy)
        diff_vy = abs(x_ekf[-1][4] - vy_end_noisy)
        diff_vz = abs(x_ekf[-1][5] - vz_end_noisy)

        self.output_box.insert(tk.END,
            f"[EKF] Absolute difference in position:[{diff_x},{diff_y},{diff_z}]\n"
        )

        self.output_box.insert(tk.END,
            f"[EKF] Absolute difference in velocity:[{diff_vx},{diff_vy},{diff_vz}]\n"
        )

        # Magnitudes
        diff_position = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        diff_velocity = np.sqrt(diff_vx**2 + diff_vy**2 + diff_vz**2)

        self.output_box.insert(tk.END,
            f"[EKF] Magnitude of position difference:{diff_position}\n"
        )

        self.output_box.insert(tk.END,
            f"[EKF] Magnitude of velocity difference:{diff_velocity}\n"
        )

        
        self.x_ekf=x_ekf
        ekf_end_abs_time = t_meas_for_ekf_abs[-1]
        self.ekf_end_abs_time=ekf_end_abs_time

        print("DEBUG: stored self.x_ekf.shape:", self.x_ekf.shape)
        print("DEBUG: stored ekf abs times len:", len(self.t_meas_for_ekf_abs))

        #self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, f"IOD r2: {self.r2}\n")
        self.output_box.insert(tk.END, f"IOD v_hg (Herrick-Gibbs or fallback): {self.v_hg}\n")
        

        self.update_plot()

    # ----------------- RUN COWELL -----------------
    def run_cowell(self):
        # Choose which dataset determines the epoch
        self.master.config(cursor="watch")   # Windows spinning cursor
        self.master.update()       
        if self.cowell_source_var.get() == "EKF":
            if self.x_ekf is None:
                messagebox.showerror("Cowell", "EKF state unavailable, cannot start from EKF.")
                return
            self.output_box.insert(tk.END, f"Cowell input method selected:EKF end\n")
            X = self.x_ekf
            # if stored as shape (6, N) -> transpose to (N, 6) for uniform handling
            if X.ndim == 2 and X.shape[0] == 6 and X.shape[1] >= 1:
                X_proc = X.T
            else:
                X_proc = X  # assume (N,6)

            # now X_proc is (N,6) where last row is latest state
            r0 = X_proc[-1, 0:3].astype(float)
            v0 = X_proc[-1, 3:6].astype(float)

            # EKF absolute times must exist
            if getattr(self, 't_meas_for_ekf_abs', None) is not None and len(self.t_meas_for_ekf_abs) >= 1:
                self.ekf_end_abs_time = float(self.t_meas_for_ekf_abs[-1])
            else:
                # fallback: if you have stored ekf_end_abs_time before, keep it
                self.ekf_end_abs_time = getattr(self, 'ekf_end_abs_time', None)
                if self.ekf_end_abs_time is None:
                    messagebox.showerror("Cowell", "EKF end absolute time unavailable.")
                    return           
        else:  # IOD
            if self.r2 is None or self.v_hg is None:
                messagebox.showerror("Cowell", "IOD (HG) state unavailable, run IOD first.")
                return
            self.output_box.insert(tk.END, f"Cowell input method selected:IOD point\n")
            r0 = self.r2
            v0 = self.v_hg
            self.ekf_end_abs_time =self.t_j_abs
        self.output_box.insert(tk.END, f"Cowell input r0:{r0}\n")
        self.output_box.insert(tk.END, f"Cowell input v0:{v0}\n")
        
        idx_clean24_start = int(np.argmin(np.abs(self.clean24_abs - self.ekf_end_abs_time)))
        clean24_segment_abs = self.clean24_abs[idx_clean24_start:]
        # t_eval (seconds relative to ekf_end_abs_time)
        t_eval = clean24_segment_abs -self.ekf_end_abs_time

        # Ensure t_eval[0] == 0 so the solver's y0 corresponds to t_eval[0]
        t_eval = t_eval - t_eval[0]

        print("[ALIGN] nearest clean24 index to ekf end:", idx_clean24_start, " clean24 abs time at that idx:", clean24_segment_abs[0])
        print("[COWELL] Propagation length (samples):", len(t_eval), " first t_eval:", t_eval[:3], " last t_eval:", t_eval[-3:])

        # epoch_jd should correspond to ekf_end_abs_time
        epoch_dt_r2 = pd.to_datetime(self.ekf_end_abs_time, unit='s').to_pydatetime()
        sec = epoch_dt_r2.second + epoch_dt_r2.microsecond * 1e-6
        jd_epoch, fr = jday(epoch_dt_r2.year, epoch_dt_r2.month, epoch_dt_r2.day,
                            epoch_dt_r2.hour, epoch_dt_r2.minute, sec)
        epoch_jd_r2 = jd_epoch + fr
        print("[COWELL] ekf-end epoch datetime:", epoch_dt_r2, " epoch_jd_r2:", epoch_jd_r2)

        # propagate Cowell forward starting at ekf_end_abs_time using r_last,v_last
        print("Cowell INPUT r0 =", r0)
        print("Cowell INPUT v0 =", v0)
        print("Cowell input |v0| =", np.linalg.norm(v0))

        use_j2=self.use_j2_var.get()  
        use_j3=self.use_j3_var.get()     
        use_j4=self.use_j4_var.get()     
        use_drag=self.use_drag_var.get()   
        use_sun=self.use_sun_var.get()    
        use_moon=self.use_moon_var.get() 
        use_H_SCALE=self.H_SCALE_var.get()
        use_RHO_REF=self.RHO_REF_var.get()
        use_H_REF=self.H_REF_var.get()
        use_OMEGA_E=self.OMEGA_E_var.get()
        use_B_STAR=self.B_STAR_var.get()
        use_CD_AM=self.CD_A_M_var.get()

        self.output_box.insert(tk.END, f"Cowell propagation starting...\n")
        start_time=time.time()
        self.run_cowell_threaded(
            r0, v0, t_eval, epoch_jd_r2,
            use_H_SCALE, use_RHO_REF, use_H_REF,
            use_OMEGA_E, use_CD_AM,
            use_j2, use_j3, use_j4,
            use_drag, use_sun, use_moon
        )
        cowell_states = self.cowell_states
        self.thread_start_time = start_time
        #self.output_box.insert(tk.END, f"Time taken for cowell function to complete:{end_time-start_time}s.\n")
        #self.cowell_states=cowell_states
        
        # np.savetxt("cowell_states_output_from_ekf_final1.csv", cowell_states, delimiter=",", header="px,py,pz,vx,vy,vz", comments="")
        # print("[COWELL] saved cowell_states_output_from_ekf_final1.csv")

        #self.update_plot()
        

    # ----------------- UPDATE PLOT -----------------
    
    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Orbit 3D Visualization")
        self.ax.set_xlabel("X (km)")
        self.ax.set_ylabel("Y (km)")
        self.ax.set_zlabel("Z (km)")

        plot3d_lines = []

        # ----------------------------
        # 1. Measured Orbit
        # ----------------------------
        if self.data is not None and self.show_measured_var.get():
            try:
                arr = self.data[['px_km', 'py_km', 'pz_km']].values.astype(float)
                l = self.ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color='red', label='Measured')
                plot3d_lines.append(l[0])
            except Exception:
                pass

        # ----------------------------
        # 2. IOD Point
        # ----------------------------
        if self.r2 is not None and self.show_iod_points_var.get():
            try:
                s = self.ax.scatter(self.r2[0], self.r2[1], self.r2[2], color='green', label='IOD Point')
                plot3d_lines.append(s)
            except Exception:
                pass

        # ----------------------------
        # 3. EKF Orbit
        # ----------------------------
        print("[Debug] Length of ekf to be plotted in 3D",len(self.x_ekf[0,:]))
        print("[Debug] Size of ekf to be plotted in 3D",np.shape(self.x_ekf))
        if self.x_ekf is not None and self.show_ekf_var.get():
            try:
                #l = self.ax.plot(self.x_ekf[0, :], self.x_ekf[1, :], self.x_ekf[2, :], color='blue', label='EKF')
                l = self.ax.plot(self.x_ekf[:, 0], self.x_ekf[:, 1], self.x_ekf[:, 2], color='blue', label='EKF')
                plot3d_lines.append(l[0])
            except Exception:
                pass

        # ----------------------------
        # 4. Clean Full Orbit
        # ----------------------------
        if self.clean_data is not None and self.show_ric_var.get():
            try:
                r_clean_full = self.clean_data[['px_km', 'py_km', 'pz_km']].values.astype(float)
                l = self.ax.plot(r_clean_full[:, 0], r_clean_full[:, 1], r_clean_full[:, 2], color='orange', label='Clean Orbit')
                plot3d_lines.append(l[0])
            except Exception:
                pass

        # ----------------------------
        # 5. Cowell Orbit (UPDATED)
        # ----------------------------
        if self.cowell_states is not None and self.show_cowell_var.get():
            try:
                r = self.cowell_states[:, 0:3]
                l = self.ax.plot(r[:, 0], r[:, 1], r[:, 2], color='purple', label='Cowell')
                plot3d_lines.append(l[0])
            except Exception as e:
                print("[3D Cowell Plot Error]", e)

        # ----------------------------
        # 6. Clean 24hr Orbit
        # ----------------------------
        if self.clean_24hr_data is not None and self.show_clean_24hr_var.get():
            try:
                r_clean_24hr = self.clean_24hr_data[['px_km', 'py_km', 'pz_km']].values.astype(float)
                l = self.ax.plot(r_clean_24hr[:, 0], r_clean_24hr[:, 1], r_clean_24hr[:, 2], 
                                color='magenta', label='Clean 24hr Orbit')
                plot3d_lines.append(l[0])
            except Exception:
                pass

        # create legend proxies and map handles
        if len(plot3d_lines) > 0:
            labels = [h.get_label() for h in plot3d_lines]
            proxies = []
            for orig in plot3d_lines:
                kind = type(orig).__name__
                color = 'k'
                try:
                    color = orig.get_color()
                except Exception:
                    try:
                        fc = orig.get_facecolor()
                        if hasattr(fc, '__len__') and len(fc) > 0:
                            color = fc[0]
                        else:
                            color = fc
                    except Exception:
                        pass
                if kind == 'Path3DCollection':
                    proxy = Line2D([0], [0], marker='o', color=color, linestyle='None')
                else:
                    proxy = Line2D([0], [0], color=color, linestyle='-')
                proxies.append(proxy)

            try:
                legend3 = self.ax.legend(handles=proxies, labels=labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
                self._legend3 = legend3
                self.fig.subplots_adjust(right=0.72)
            except Exception:
                legend3 = self.ax.legend(loc='upper right')

            # prepare mapping and pick handlers
            try:
                self.canvas.draw()
                renderer3 = self.fig.canvas.get_renderer()
                legend_texts = legend3.get_texts()
                legend_labels = [t.get_text() for t in legend_texts]
                try:
                    leg_handles = legend3.legendHandles
                except Exception:
                    try:
                        leg_handles = legend3.get_lines()
                    except Exception:
                        leg_handles = []

                orig_by_label = {artist.get_label(): artist for artist in plot3d_lines}
                self._legend3_map = {}
                self._legend3_order = []
                for i, leg_label in enumerate(legend_labels):
                    orig_artist = orig_by_label.get(leg_label)
                    if orig_artist is not None:
                        self._legend3_order.append(orig_artist)
                        if i < len(leg_handles):
                            leg_handle = leg_handles[i]
                            if hasattr(leg_handle, 'set_picker'):
                                leg_handle.set_picker(5)
                            self._legend3_map[leg_handle] = orig_artist

                # compute bboxes for click handling
                try:
                    self._legend3_bboxes = [t.get_window_extent(renderer3) for t in legend_texts]
                except Exception:
                    self._legend3_bboxes = []
            except Exception:
                self._legend3_map = {}
                self._legend3_bboxes = []
                self._legend3_order = []

            def _on_legend3_pick(event):
                legline = event.artist
                orig = self._legend3_map.get(legline)
                if orig is None:
                    return
                vis = not orig.get_visible()
                orig.set_visible(vis)
                legline.set_alpha(1.0 if vis else 0.2)
                self.canvas.draw()

            try:
                if hasattr(self, '_legend3_cid') and self._legend3_cid is not None:
                    self.canvas.mpl_disconnect(self._legend3_cid)
            except Exception:
                pass
            try:
                self._legend3_cid = self.canvas.mpl_connect('pick_event', _on_legend3_pick)
            except Exception:
                self._legend3_cid = None

            def _on_legend3_click(event):
                try:
                    renderer3 = self.fig.canvas.get_renderer()
                    self._legend3_bboxes = [t.get_window_extent(renderer3) for t in legend3.get_texts()]
                except Exception:
                    pass
                if not hasattr(self, '_legend3_bboxes') or len(self._legend3_bboxes) == 0:
                    return
                x, y = event.x, event.y
                for idx, bbox in enumerate(self._legend3_bboxes):
                    if bbox.contains(x, y):
                        try:
                            orig = self._legend3_order[idx]
                        except Exception:
                            return
                        vis = not orig.get_visible()
                        orig.set_visible(vis)
                        try:
                            for leg_handle, leg_orig in self._legend3_map.items():
                                if leg_orig is orig:
                                    try:
                                        leg_handle.set_alpha(1.0 if vis else 0.2)
                                    except Exception:
                                        pass
                                    break
                        except Exception:
                            pass
                        self.canvas.draw()
                        return

            try:
                if hasattr(self, '_legend3_click_cid') and self._legend3_click_cid is not None:
                    self.canvas.mpl_disconnect(self._legend3_click_cid)
            except Exception:
                pass
            try:
                self._legend3_click_cid = self.canvas.mpl_connect('button_press_event', _on_legend3_click)
            except Exception:
                self._legend3_click_cid = None
        else:
            self.canvas.draw()

        # ============================================================
        # NEW TWO-PANEL RIC ERROR FIGURE (EKF & COWELL)
        # ============================================================

        # Destroy old children (old single graph)
        for widget in self.bottom_frame.winfo_children():
           widget.destroy()

        # Create new figure with 2 subplots
        self.fig2, (self.ax2_ekf, self.ax2_cowell) = plt.subplots(
           2, 1, figsize=(8, 6)
        )

        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.bottom_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.bottom_frame)
        self.toolbar2.update()
        self.toolbar2.pack(fill=tk.X)

        # ------------------------------------------------------------
        #                 EKF RIC ERRORS
        # ------------------------------------------------------------
        self.ax2_ekf.set_title("EKF RIC Errors (ΔR, ΔI, ΔC)")
        self.ax2_ekf.set_ylabel("Error (km)")
        self.ax2_ekf.set_xlabel("Time (seconds)")
        ekf_lines = []

        if self.clean_data is not None and self.x_ekf is not None:
            try:
                X = self.x_ekf
                if X.shape[0] == 6:
                    X = X.T   # (N,6)

                clean_start = self.j
                r_clean = self.clean_data[['px_km','py_km','pz_km']].values.astype(float)
                v_clean = self.clean_data[['vx_km_s','vy_km_s','vz_km_s']].values.astype(float)

                n_ekf = min(len(X), len(r_clean) - clean_start)

                # Slice clean data correctly
                r_ref = r_clean[clean_start : clean_start + n_ekf]
                v_ref = v_clean[clean_start : clean_start + n_ekf]

                r_obj = X[:n_ekf, 0:3]
                v_obj = X[:n_ekf, 3:6]

                print("[Debug] size of ekf to be plotted in RIC", len(r_obj))

                ric_errors = eci_to_ric(r_ref, v_ref, r_obj, v_obj)
                self.ric_ekf=ric_errors
                print("EKF RIC error shape", ric_errors.shape)

                delR_ekf, delI_ekf, delC_ekf = ric_errors[:, 0], ric_errors[:, 1], ric_errors[:, 2]
                self.output_box.insert(tk.END,
                    f"[EKF] delR mean: {np.mean(delR_ekf)}, max: {np.max(delR_ekf)}, min: {np.min(delR_ekf)}\n")
                self.output_box.insert(tk.END,
                    f"[EKF] delI mean: {np.mean(delI_ekf)}, max: {np.max(delI_ekf)}, min: {np.min(delI_ekf)}\n")
                self.output_box.insert(tk.END,
                    f"[EKF] delC mean: {np.mean(delC_ekf)}, max: {np.max(delC_ekf)}, min: {np.min(delC_ekf)}\n")
                
                # Create time axis in seconds
                t_sec = np.arange(n_ekf)   # assuming 1 second timestep

                

                l1, = self.ax2_ekf.plot(t_sec,ric_errors[:,0], label="ΔR (EKF - Clean 5min)")
                l2, = self.ax2_ekf.plot(t_sec,ric_errors[:,1], label="ΔI (EKF - Clean 5min)")
                l3, = self.ax2_ekf.plot(t_sec,ric_errors[:,2], label="ΔC (EKF - Clean 5min)")
                ekf_lines.extend([l1, l2, l3])

            except Exception as e:
                print("EKF RIC plot failed:", e)

        # Legend for EKF subplot - ADD PICKER SUPPORT
        if ekf_lines:
            self._legend2_ekf = self.ax2_ekf.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            
            # Create mapping for EKF legend
            self._legend2_ekf_map = {}
            self._legend2_ekf_order = ekf_lines[:]
            
            # Get handles and set pickers
            try:
                ekf_legend_handles = self._legend2_ekf.legendHandles
            except AttributeError:
                try:
                    ekf_legend_handles = self._legend2_ekf.get_lines()
                except Exception:
                    ekf_legend_handles = []
                    
            # Create mapping between legend handles and original artists
            for i, (orig, leg_handle) in enumerate(zip(ekf_lines, ekf_legend_handles)):
                if hasattr(leg_handle, 'set_picker'):
                    leg_handle.set_picker(5)
                self._legend2_ekf_map[leg_handle] = orig

        # ------------------------------------------------------------
        #                 COWELL RIC ERRORS
        # ------------------------------------------------------------
        self.ax2_cowell.set_title("Cowell RIC Errors (ΔR, ΔI, ΔC)")
        self.ax2_cowell.set_xlabel("Time in hours")
        self.ax2_cowell.set_ylabel("Error (km)")

        cowell_lines = []

        if self.clean_24hr_data is not None and hasattr(self, 'cowell_states'):
            try:
                cow = self.cowell_states
                r_cow = cow[:,0:3]
                v_cow = cow[:,3:6]

                # find clean24 start index
                idx_clean24_start = int(np.argmin(np.abs(self.clean24_abs - self.ekf_end_abs_time)))

                r_clean24 = self.clean_24hr_data[['px_km','py_km','pz_km']].values.astype(float)
                if all(col in self.clean_24hr_data.columns for col in ['vx_km_s','vy_km_s','vz_km_s']):
                    v_clean24 = self.clean_24hr_data[['vx_km_s','vy_km_s','vz_km_s']].values.astype(float)
                else:
                    v_clean24 = np.zeros_like(r_clean24)

                # slice segment
                r_ref = r_clean24[idx_clean24_start : idx_clean24_start + len(r_cow)]
                v_ref = v_clean24[idx_clean24_start : idx_clean24_start + len(r_cow)]

                n = min(len(r_ref), len(r_cow))
                ric_errors_c = eci_to_ric(r_ref[:n], v_ref[:n], r_cow[:n], v_cow[:n])
                self.ric_cowell=ric_errors_c
                print("Cowell RIC error shape", ric_errors_c.shape)

                delR_cowell, delI_cowell, delC_cowell = ric_errors_c[:,0], ric_errors_c[:,1], ric_errors_c[:,2]
                self.output_box.insert(tk.END,
                    f"[Cowell] delR mean: {np.mean(delR_cowell)}, max: {np.max(delR_cowell)}, min: {np.min(delR_cowell)}\n")
                self.output_box.insert(tk.END,
                    f"[Cowell] delI mean: {np.mean(delI_cowell)}, max: {np.max(delI_cowell)}, min: {np.min(delI_cowell)}\n")
                self.output_box.insert(tk.END,
                    f"[Cowell] delC mean: {np.mean(delC_cowell)}, max: {np.max(delC_cowell)}, min: {np.min(delC_cowell)}\n")
                
                t_hours = np.arange(n) / 3600.0       # convert seconds → hours
                c1, = self.ax2_cowell.plot(t_hours,ric_errors_c[:,0], '--', label="ΔR (Cowell - Clean24)")
                c2, = self.ax2_cowell.plot(t_hours,ric_errors_c[:,1], '--', label="ΔI (Cowell - Clean24)")
                c3, = self.ax2_cowell.plot(t_hours,ric_errors_c[:,2], '--', label="ΔC (Cowell - Clean24)")
                cowell_lines.extend([c1, c2, c3])

            except Exception as e:
                print("Cowell RIC plot failed:", e)

        # Legend for Cowell subplot - ADD PICKER SUPPORT
        if cowell_lines:
            self._legend2_cowell = self.ax2_cowell.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            
            # Create mapping for Cowell legend
            self._legend2_cowell_map = {}
            self._legend2_cowell_order = cowell_lines[:]
            
            # Get handles and set pickers
            try:
                cowell_legend_handles = self._legend2_cowell.legendHandles
            except AttributeError:
                try:
                    cowell_legend_handles = self._legend2_cowell.get_lines()
                except Exception:
                    cowell_legend_handles = []
                    
            # Create mapping between legend handles and original artists
            for i, (orig, leg_handle) in enumerate(zip(cowell_lines, cowell_legend_handles)):
                if hasattr(leg_handle, 'set_picker'):
                    leg_handle.set_picker(5)
                self._legend2_cowell_map[leg_handle] = orig

        self.canvas2.draw()

        # ------------------------------------------------------------
        # Setup pick event handlers for 2D legends
        # ------------------------------------------------------------
        def _on_legend2_pick(event):
            """Handle pick events for both EKF and Cowell 2D legends"""
            legline = event.artist
            
            # Check EKF legend first
            if hasattr(self, '_legend2_ekf_map') and legline in self._legend2_ekf_map:
                orig = self._legend2_ekf_map[legline]
                if orig is None:
                    return
                vis = not orig.get_visible()
                orig.set_visible(vis)
                legline.set_alpha(1.0 if vis else 0.2)
                self.canvas2.draw()
                return
                
            # Check Cowell legend
            if hasattr(self, '_legend2_cowell_map') and legline in self._legend2_cowell_map:
                orig = self._legend2_cowell_map[legline]
                if orig is None:
                    return
                vis = not orig.get_visible()
                orig.set_visible(vis)
                legline.set_alpha(1.0 if vis else 0.2)
                self.canvas2.draw()
                return

        # Connect the pick event handler
        try:
            if hasattr(self, '_legend2_pick_cid') and self._legend2_pick_cid is not None:
                self.canvas2.mpl_disconnect(self._legend2_pick_cid)
        except Exception:
            pass
        
        try:
            self._legend2_pick_cid = self.canvas2.mpl_connect('pick_event', _on_legend2_pick)
        except Exception:
            self._legend2_pick_cid = None

        # ------------------------------------------------------------
        # Setup click event handler for 2D plot area (for text clicking)
        # ------------------------------------------------------------
        def _on_2d_plot_click(event):
            """Handle clicks on the 2D plot area (for clicking legend text)"""
            if event.inaxes not in [self.ax2_ekf, self.ax2_cowell]:
                return
                
            try:
                renderer = self.fig2.canvas.get_renderer()
            except Exception:
                return
                
            # Check EKF legend
            if hasattr(self, '_legend2_ekf') and self._legend2_ekf:
                try:
                    texts = self._legend2_ekf.get_texts()
                    bboxes = [t.get_window_extent(renderer) for t in texts]
                    
                    for idx, bbox in enumerate(bboxes):
                        if bbox.contains(event.x, event.y):
                            if idx < len(self._legend2_ekf_order):
                                orig = self._legend2_ekf_order[idx]
                                vis = not orig.get_visible()
                                orig.set_visible(vis)
                                
                                # Update corresponding legend handle
                                for leg_handle, leg_orig in self._legend2_ekf_map.items():
                                    if leg_orig is orig:
                                        try:
                                            leg_handle.set_alpha(1.0 if vis else 0.2)
                                        except Exception:
                                            pass
                                        break
                                
                                self.canvas2.draw()
                                return
                except Exception:
                    pass
                    
            # Check Cowell legend
            if hasattr(self, '_legend2_cowell') and self._legend2_cowell:
                try:
                    texts = self._legend2_cowell.get_texts()
                    bboxes = [t.get_window_extent(renderer) for t in texts]
                    
                    for idx, bbox in enumerate(bboxes):
                        if bbox.contains(event.x, event.y):
                            if idx < len(self._legend2_cowell_order):
                                orig = self._legend2_cowell_order[idx]
                                vis = not orig.get_visible()
                                orig.set_visible(vis)
                                
                                # Update corresponding legend handle
                                for leg_handle, leg_orig in self._legend2_cowell_map.items():
                                    if leg_orig is orig:
                                        try:
                                            leg_handle.set_alpha(1.0 if vis else 0.2)
                                        except Exception:
                                            pass
                                        break
                                
                                self.canvas2.draw()
                                return
                except Exception:
                    pass

        # Connect the click event handler for the 2D plot
        try:
            if hasattr(self, '_2d_click_cid') and self._2d_click_cid is not None:
                self.canvas2.mpl_disconnect(self._2d_click_cid)
        except Exception:
            pass
        
        try:
            self._2d_click_cid = self.canvas2.mpl_connect('button_press_event', _on_2d_plot_click)
        except Exception:
            self._2d_click_cid = None

        # ------------------------------------------------------------
        #                 Checkbox controls for 2D plots
        # ------------------------------------------------------------
        self._checkbox_map = {}
        self._checkbox_order = []

        ric_plot_lines = ekf_lines + cowell_lines
        if len(ric_plot_lines) > 0:
            ttk.Label(self.legend_ctrl_frame, text="2D Series:").pack(anchor=tk.W, pady=(6,0))
            f2 = ttk.Frame(self.legend_ctrl_frame)
            f2.pack(fill=tk.X, padx=2)

            for orig in ric_plot_lines:
                try:
                    lab = orig.get_label()
                except Exception:
                    lab = str(orig)

                var = tk.BooleanVar(value=orig.get_visible())
                leg_handle = None

                # Find corresponding legend handle
                for map_dict in [getattr(self, '_legend2_ekf_map', {}), getattr(self, '_legend2_cowell_map', {})]:
                    for lh, oa in map_dict.items():
                        if oa is orig:
                            leg_handle = lh
                            break
                    if leg_handle:
                        break

                cb = tk.Checkbutton(f2, text=lab, variable=var,
                                    command=lambda a=orig, v=var, lh=leg_handle: self._on_checkbox_toggle(a, v, lh))
                cb.pack(anchor=tk.W)
                self._checkbox_map[orig] = (var, leg_handle)
                self._checkbox_order.append(var)

        self.canvas2.mpl_connect('button_press_event', self._on_tk_click_2)


    # ----------------- TK CLICK HANDLERS -----------------
    def _on_tk_click_3(self, event):
        """Handle raw Tk clicks on the 3D canvas and toggle legend items."""
        try:
            if not hasattr(self, '_legend3'):
                return
            try:
                renderer3 = self.fig.canvas.get_renderer()
                texts = self._legend3.get_texts()
                bboxes = [t.get_window_extent(renderer3) for t in texts]
            except Exception:
                return
            try:
                canvas_widget = self.canvas.get_tk_widget()
                widget_w = canvas_widget.winfo_width()
                widget_h = canvas_widget.winfo_height()
            except Exception:
                widget_w = widget_h = None
            try:
                pix_w, pix_h = self.fig.canvas.get_width_height()
            except Exception:
                pix_w = pix_h = None

            if widget_w and pix_w and widget_w > 0:
                sx = pix_w / float(widget_w)
            else:
                sx = 1.0
            if widget_h and pix_h and widget_h > 0:
                sy = pix_h / float(widget_h)
            else:
                sy = 1.0

            disp_x = event.x * sx
            disp_y = pix_h - (event.y * sy) if pix_h is not None else event.y

            for idx, bbox in enumerate(bboxes):
                if bbox.contains(disp_x, disp_y):
                    try:
                        orig = self._legend3_order[idx]
                    except Exception:
                        return
                    try:
                        vis = not orig.get_visible()
                        orig.set_visible(vis)
                    except Exception:
                        pass
                    try:
                        for leg_handle, leg_orig in self._legend3_map.items():
                            if leg_orig is orig:
                                try:
                                    leg_handle.set_alpha(1.0 if vis else 0.2)
                                except Exception:
                                    pass
                                break
                    except Exception:
                        pass
                    self.canvas.draw()
                    return

            try:
                frame = self._legend3.get_frame()
                frame_bbox = frame.get_window_extent(renderer3)
                if frame_bbox.contains(disp_x, disp_y):
                    centers = [0.5 * (bb.y0 + bb.y1) for bb in bboxes]
                    dists = [abs(c - disp_y) for c in centers]
                    if len(dists) > 0:
                        nn = int(np.argmin(dists))
                        try:
                            orig = self._legend3_order[nn]
                            vis = not orig.get_visible()
                            orig.set_visible(vis)
                            for leg_handle, leg_orig in self._legend3_map.items():
                                if leg_orig is orig:
                                    try:
                                        leg_handle.set_alpha(1.0 if vis else 0.2)
                                    except Exception:
                                        pass
                                    break
                            self.canvas.draw()
                            return
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            return

    def _on_tk_click_2(self, event):
        """Handle raw Tk clicks on the 2D canvas and toggle legend items for EKF & Cowell RIC plots."""
        try:
            # List of legends to handle
            legends = [
                (getattr(self, '_legend2_ekf', None), getattr(self, '_legend_map_ekf', {}), getattr(self, '_legend_order_ekf', [])),
                (getattr(self, '_legend2_cowell', None), getattr(self, '_legend_map_cowell', {}), getattr(self, '_legend_order_cowell', []))
            ]

            try:
                renderer = self.fig2.canvas.get_renderer()
            except Exception:
                return

            try:
                canvas_widget = self.canvas2.get_tk_widget()
                widget_w = canvas_widget.winfo_width()
                widget_h = canvas_widget.winfo_height()
            except Exception:
                widget_w = widget_h = None

            try:
                pix_w, pix_h = self.fig2.canvas.get_width_height()
            except Exception:
                pix_w = pix_h = None

            sx = pix_w / float(widget_w) if widget_w and pix_w and widget_w > 0 else 1.0
            sy = pix_h / float(widget_h) if widget_h and pix_h and widget_h > 0 else 1.0

            disp_x = event.x * sx
            disp_y = pix_h - (event.y * sy) if pix_h is not None else event.y

            for legend_obj, map_dict, order_list in legends:
                if legend_obj is None:
                    continue  # skip missing legend

                texts = legend_obj.get_texts()
                if not texts:
                    continue

                bboxes = [t.get_window_extent(renderer) for t in texts]

                # Check if click is inside any legend text
                for idx, bbox in enumerate(bboxes):
                    if bbox.contains(disp_x, disp_y):
                        try:
                            orig = order_list[idx]
                            vis = not orig.get_visible()
                            orig.set_visible(vis)

                            # Update corresponding legend handle alpha
                            for leg_handle, leg_orig in map_dict.items():
                                if leg_orig is orig:
                                    try:
                                        leg_handle.set_alpha(1.0 if vis else 0.2)
                                    except Exception:
                                        pass
                                    break

                            self.canvas2.draw()
                            return
                        except Exception:
                            continue

                # Check if click is inside the legend frame
                try:
                    frame = legend_obj.get_frame()
                    frame_bbox = frame.get_window_extent(renderer)
                    if frame_bbox.contains(disp_x, disp_y):
                        centers = [0.5 * (bb.y0 + bb.y1) for bb in bboxes]
                        dists = [abs(c - disp_y) for c in centers]
                        if len(dists) > 0:
                            nn = int(np.argmin(dists))
                            try:
                                orig = order_list[nn]
                                vis = not orig.get_visible()
                                orig.set_visible(vis)

                                for leg_handle, leg_orig in map_dict.items():
                                    if leg_orig is orig:
                                        try:
                                            leg_handle.set_alpha(1.0 if vis else 0.2)
                                        except Exception:
                                            pass
                                        break

                                self.canvas2.draw()
                                return
                            except Exception:
                                continue
                except Exception:
                    continue

        except Exception:
            return

    def _on_checkbox_toggle(self, artist, var, legend_handle=None):
        vis = var.get()
        try:
            artist.set_visible(vis)
        except Exception:
            pass

        if legend_handle is not None:
            try:
                legend_handle.set_alpha(1.0 if vis else 0.2)
            except Exception:
                pass
        else:
            for map_attr in ['_legend3_map', '_legend_map_ekf', '_legend_map_cowell']:
                try:
                    for lh, oa in getattr(self, map_attr, {}).items():
                        if oa is artist:
                            try:
                                lh.set_alpha(1.0 if vis else 0.2)
                            except Exception:
                                pass
                            break
                except Exception:
                    pass

        # redraw canvas
        try:
            if hasattr(self, '_legend3_order') and artist in getattr(self, '_legend3_order', []):
                self.canvas.draw()
            else:
                self.canvas2.draw()
        except Exception:
            try:
                self.canvas.draw()
            except Exception:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = IODApp(root)
    root.mainloop()