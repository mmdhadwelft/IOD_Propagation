import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os

# Constants
# Constants (WGS72 to match SGP4)
MU = 398600.8       # km^3/s^2
RE = 6378.135       # km
J2 = 1.082616e-3
J3 = -2.53881e-6
J4 = -1.65597e-6

# TLE for Cartosat-3 (NORAD ID 44804)
# Epoch: 2025-11-23 (Day 327)
TLE_LINE1 = "1 44804U 19081A   25327.85928487  .00004606  00000-0  22108-3 0  9991"
TLE_LINE2 = "2 44804  97.4606  27.6788 0013715 149.1101 211.0943 15.19283085332297"

def herrick_gibbs(r1, r2, r3, t1, t2, t3):
    dt21 = t2 - t1
    dt32 = t3 - t2
    dt31 = t3 - t1
    if abs(dt21) < 1e-9 or abs(dt32) < 1e-9 or abs(dt31) < 1e-9:
        return np.zeros(3)
    term1 = -dt32 / (dt21 * dt31) * np.array(r1, dtype=float)
    term2 = (dt32 - dt21) / (dt21 * dt32) * np.array(r2, dtype=float)
    term3 = dt21 / (dt32 * dt31) * np.array(r3, dtype=float)
    v2 = term1 + term2 + term3
    return v2

def get_sgp4_states(line1, line2, duration_hours=24, step_seconds=60):
    satellite = Satrec.twoline2rv(line1, line2, WGS72)
    
    # Get epoch from satellite object
    epoch_dt = datetime(satellite.epochyr + 2000 if satellite.epochyr < 57 else satellite.epochyr + 1900, 1, 1) + timedelta(days=satellite.epochdays - 1)
    #print(len(epoch_dt))
    times = []
    positions = []
    velocities = []
    
    steps = int(duration_hours * 3600 / step_seconds)
    
    for i in range(steps + 1):
        dt = timedelta(seconds=i*step_seconds)
        current_time = epoch_dt + dt
        
        jd, fr = jday(current_time.year, current_time.month, current_time.day,
                      current_time.hour, current_time.minute, current_time.second)
        
        e, r, v = satellite.sgp4(jd, fr)
        
        if e == 0:
            times.append(i * step_seconds)
            positions.append(r)
            velocities.append(v)
        else:
            print(f"SGP4 Error at step {i}: {e}")
            
    return np.array(times), np.array(positions), np.array(velocities), epoch_dt

def acceleration(t, state):
    x, y, z, vx, vy, vz = state
    r_vec = np.array([x, y, z])
    r_sq = np.dot(r_vec, r_vec)
    r_norm = np.sqrt(r_sq)
    
    # Precompute common terms
    r2 = r_sq
    r3 = r2 * r_norm
    r4 = r2 * r2
    r5 = r4 * r_norm
    r7 = r5 * r2
    
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    
    ratio_sq = (z / r_norm)**2
    
    # Two-body acceleration
    a_2body = -MU * r_vec / r3
    
    # J2 Perturbation
    # ax = -3/2 * J2 * (MU*RE^2/r^5) * x * (1 - 5*z^2/r^2)
    c2 = 1.5 * J2 * MU * (RE**2) / r5
    ax_j2 = -c2 * x * (1 - 5 * ratio_sq)
    ay_j2 = -c2 * y * (1 - 5 * ratio_sq)
    az_j2 = -c2 * z * (3 - 5 * ratio_sq)
    
    # J3 Perturbation
    # ax = -5/2 * J3 * (MU*RE^3/r^7) * x * z * (3 - 7*z^2/r^2)
    # az = -1/2 * J3 * (MU*RE^3/r^7) * (30*z^2 - 35*z^4/r^2 - 3*r^2)
    c3 = 0.5 * J3 * MU * (RE**3) / r7
    ax_j3 = -5 * c3 * x * z * (3 - 7 * ratio_sq)
    ay_j3 = -5 * c3 * y * z * (3 - 7 * ratio_sq)
    az_j3 = -c3 * (30 * z2 - 35 * z4 / r2 - 3 * r2)
    
    # J4 Perturbation
    # ax = 15/8 * J4 * (MU*RE^4/r^7) * x * (1 - 14*z^2/r^2 + 21*z^4/r^4)
    # az = 5/8 * J4 * (MU*RE^4/r^7) * z * (15 - 70*z^2/r^2 + 63*z^4/r^4)
    c4 = 0.125 * J4 * MU * (RE**4) / r7
    term_x = 1 - 14 * ratio_sq + 21 * ratio_sq**2
    term_z = 15 - 70 * ratio_sq + 63 * ratio_sq**2
    
    ax_j4 = 15 * c4 * x * term_x
    ay_j4 = 15 * c4 * y * term_x
    az_j4 = 5 * c4 * z * term_z
    
    # Atmospheric Drag
    # Simple exponential atmosphere model
    # Constants for 500km altitude (approximate for Cartosat-3)
    H_SCALE = 88.0  # km (Scale height)
    RHO_REF = 1e-13 # kg/m^3 (approx density at 500km)
    H_REF = 500.0   # km
    
    # Calculate altitude
    h = r_norm - RE
    
    # Density (kg/km^3 for consistency with km units)
    # 1 kg/m^3 = 1e9 kg/km^3
    rho_kg_m3 = RHO_REF * np.exp(-(h - H_REF) / H_SCALE)
    rho = rho_kg_m3 * 1e9 # Convert to kg/km^3
    
    # Velocity relative to rotating atmosphere
    # omega_earth = 7.2921159e-5 rad/s
    OMEGA_E = 7.2921159e-5
    v_rel_x = vx + OMEGA_E * y
    v_rel_y = vy - OMEGA_E * x
    v_rel_z = vz
    
    v_rel_sq = v_rel_x**2 + v_rel_y**2 + v_rel_z**2
    v_rel_norm = np.sqrt(v_rel_sq)
    
    # Ballistic Coefficient (Cd * A / m)
    # Estimate from B* (B_star)
    # B* = 0.5 * (Cd * A / m) * rho_0 * RE
    # We use the B* from TLE directly as a scaling factor if we assume a standard model,
    # but here we try to derive CdA/m.
    # B* from TLE: 0.00022108 (from string "22108-3")
    B_STAR = 0.00022108
    # This B* is in units of 1/EarthRadii.
    # Let's use a simplified approach: a_drag = -2 * B_STAR * n^2 * ... (SGP4 formula is complex)
    # Instead, let's use the standard drag equation with a tuned CdA/m.
    # A common approximation: B_term = 2 * B_STAR / RHO_0_SGP4 ?
    # Let's try a heuristic: B* is related to drag.
    # Let's use a fixed CdA/m that is typical for this class of satellite if B* conversion is ambiguous,
    # OR better: use B* to scale the density.
    # SGP4 density is "rho".
    # a_drag = - B_STAR * (rho / rho0) * v^2 ?
    
    # Let's use the standard drag equation: a = -0.5 * rho * (CdA/m) * v^2
    # And estimate CdA/m.
    # For Cartosat-3 (mass ~1600kg, Area ~10m2?), Cd~2.2 -> CdA/m ~ 0.013 m^2/kg
    # Let's check what B* implies.
    # B* = 0.5 * B * rho0 * RE
    # B = 2 * B* / (rho0 * RE)
    # If we use rho0 at perigee (~500km) ~ 1e-13 kg/m^3?
    # RE = 6378 km = 6.378e6 m
    # B = 2 * 2.2e-4 / (1e-13 * 6.378e6) = 4.4e-4 / 6.3e-7 = 690 m^2/kg ?? Too high.
    # This suggests rho0 in SGP4 definition is much higher (at 120km?).
    # SGP4 rho0 is at 120km? No.
    # Let's stick to a reasonable CdA/m for a LEO satellite.
    # Cartosat-3 is ~1625 kg. Dimensions are roughly 2.5m x 2.5m? Solar panels?
    # Let's assume Area ~ 8 m^2. Cd ~ 2.2.
    # CdA/m = 2.2 * 8 / 1625 = 0.0108 m^2/kg.
    # Let's use this value.
    CD_A_M = 0.012 # m^2/kg (Approximate)
    
    # Drag acceleration vector
    # a = -0.5 * rho * (CdA/m) * v_rel * vec_v_rel
    # Units: rho [kg/km^3], CdA/m [m^2/kg] -> need [km^2/kg]?
    # CdA/m in m^2/kg = (1e-6 km^2) / kg
    CD_A_M_km = CD_A_M * 1e-6
    
    drag_factor = 0.5 * rho * CD_A_M_km * v_rel_norm
    
    ax_drag = -drag_factor * v_rel_x
    ay_drag = -drag_factor * v_rel_y
    az_drag = -drag_factor * v_rel_z
    
    # Total acceleration
    ax_total = a_2body[0] + ax_j2 + ax_j3 + ax_j4 + ax_drag
    ay_total = a_2body[1] + ay_j2 + ay_j3 + ay_j4 + ay_drag
    az_total = a_2body[2] + az_j2 + az_j3 + az_j4 + az_drag
    
    return [vx, vy, vz, ax_total, ay_total, az_total]

# Solar and Lunar Ephemerides (Simplified)
def get_sun_position(jd):
    # Low precision analytical model
    n = jd - 2451545.0
    L = 280.460 + 0.9856474 * n
    g = 357.528 + 0.9856003 * n
    L = np.radians(L % 360)
    g = np.radians(g % 360)
    
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2*g)
    epsilon = np.radians(23.439 - 0.0000004 * n)
    
    r = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2*g)
    r_km = r * 149597870.7 # AU to km
    
    x = r_km * np.cos(lam)
    y = r_km * np.cos(epsilon) * np.sin(lam)
    z = r_km * np.sin(epsilon) * np.sin(lam)
    
    return np.array([x, y, z])

def get_moon_position(jd):
    # Low precision analytical model
    T = (jd - 2451545.0) / 36525.0
    
    L_prime = np.radians(218.3164477 + 481267.88123421 * T)
    D = np.radians(297.8501921 + 445267.1114034 * T)
    M = np.radians(357.5291092 + 35999.0502909 * T)
    M_prime = np.radians(134.9633964 + 477198.8675055 * T)
    F = np.radians(93.2720950 + 483202.0175233 * T)
    
    lon = L_prime + np.radians(6.289 * np.sin(M_prime))
    lat = np.radians(5.128 * np.sin(F))
    r = 385000.56 - 20905.0 * np.cos(M_prime) # km
    
    # Ecliptic to Equatorial (approximate epsilon)
    epsilon = np.radians(23.439)
    
    x_ecl = r * np.cos(lat) * np.cos(lon)
    y_ecl = r * np.cos(lat) * np.sin(lon)
    z_ecl = r * np.sin(lat)
    
    x = x_ecl
    y = y_ecl * np.cos(epsilon) - z_ecl * np.sin(epsilon)
    z = y_ecl * np.sin(epsilon) + z_ecl * np.cos(epsilon)
    
    return np.array([x, y, z])

def acceleration_extended(t, state, epoch_jd, use_j2=True, use_j3=True, use_j4=True, use_drag=True, use_sun=True, use_moon=True,H_SCALE=60.0,RHO_REF=1e-13,H_REF=500,OMEGA_E=7.2921159e-5,CD_A_M=0.012):
    # Unpack state
    x, y, z, vx, vy, vz = state
    
    r_vec = np.array([x, y, z])
    r = np.linalg.norm(r_vec)
    r_sq = np.dot(r_vec, r_vec)
    r_norm = np.sqrt(r_sq)
    
    # --- 1. Two-Body ---
    a_2body = -MU * r_vec / (r_sq * r_norm)
    
    # Initialize perturbation accelerations
    ax_j2 = ay_j2 = az_j2 = 0.0
    ax_j3 = ay_j3 = az_j3 = 0.0
    ax_j4 = ay_j4 = az_j4 = 0.0
    ax_drag = ay_drag = az_drag = 0.0
    a_sun = np.array([0.0, 0.0, 0.0])
    a_moon = np.array([0.0, 0.0, 0.0])
    
    # --- 2. J2, J3, J4 ---
    if use_j2 or use_j3 or use_j4:
        ratio_sq = (z / r_norm)**2
        r2 = r_sq
        r3 = r2 * r_norm
        r4 = r2 * r2
        r5 = r4 * r_norm
        r7 = r5 * r2
    
    # J2
    if use_j2:
        c2 = 1.5 * J2 * MU * (RE**2) / r5
        ax_j2 = -c2 * x * (1 - 5 * ratio_sq)
        ay_j2 = -c2 * y * (1 - 5 * ratio_sq)
        az_j2 = -c2 * z * (3 - 5 * ratio_sq)
    
    #J3
    if use_j3:
        c3 = 0.5 * J3 * MU * (RE**3) / r7
        ax_j3 = -5 * c3 * x * z * (3 - 7 * ratio_sq)
        ay_j3 = -5 * c3 * y * z * (3 - 7 * ratio_sq)
        az_j3 = -c3 * (30 * z**2 - 35 * z**4 / r2 - 3 * r2)

    # J4
    if use_j4:
        c4 = 0.125 * J4 * MU * (RE**4) / r7
        term_x = 1 - 14 * ratio_sq + 21 * ratio_sq**2
        term_z = 15 - 70 * ratio_sq + 63 * ratio_sq**2
        ax_j4 = 15 * c4 * x * term_x
        ay_j4 = 15 * c4 * y * term_x
        az_j4 = 5 * c4 * z * term_z
    
    # --- 3. Atmospheric Drag ---
    if use_drag:
        # Convert everything to SI units
        RE_m = RE * 1000.0  # km to m
        H_SCALE_m = H_SCALE * 1000.0  # km to m
        H_REF_m = H_REF * 1000.0  # km to m
        
        # Altitude in meters
        h_m = r * 1000.0 - RE_m
        
        # Atmospheric density in kg/m³
        rho = RHO_REF * np.exp(-(h_m - H_REF_m) / H_SCALE_m)
        
        # Earth rotation vector (rad/s)
        omega_vec = np.array([0.0, 0.0, OMEGA_E])
        
        # Position in ECI (meters)
        r_eci_m = r_vec * 1000.0
        
        # Velocity in ECI (m/s)
        v_eci_m = np.array([vx, vy, vz]) * 1000.0
        
        # Atmosphere velocity at this point (m/s) in ECI frame
        v_atm_eci_m = np.cross(omega_vec, r_eci_m)
        
        # Relative velocity (satellite minus rotating atmosphere)
        v_rel_m = v_eci_m - v_atm_eci_m
        v_rel_norm_m = np.linalg.norm(v_rel_m)
        
        # Drag acceleration in m/s²
        if v_rel_norm_m > 0:
            #a_drag_m = -0.5 * rho * CD_A_M * v_rel_norm_m * v_rel_mf
            a_drag_m = -0.5  *7.2625e-14* CD_A_M * v_rel_norm_m * v_rel_m
        else:
            a_drag_m = np.zeros(3)
        
        # Convert drag acceleration to km/s² for integration
        a_drag = a_drag_m / 1000.0
        ax_drag, ay_drag, az_drag = a_drag
    
  
    # --- 4. Third Body (Sun & Moon) ---
    current_jd = epoch_jd + t / 86400.0
    
    # Sun
    if use_sun:
        r_sun = get_sun_position(current_jd)
        r_sc_sun = r_sun - r_vec
        mu_sun = 1.32712440018e11   # m^3/s^2
        a_sun = mu_sun * (r_sc_sun / np.linalg.norm(r_sc_sun)**3 - r_sun / np.linalg.norm(r_sun)**3)
    
    # Moon
    if use_moon:
        r_moon = get_moon_position(current_jd)
        r_sc_moon = r_moon - r_vec
        mu_moon = 4902.800066
        a_moon = mu_moon * (r_sc_moon / np.linalg.norm(r_sc_moon)**3 - r_moon / np.linalg.norm(r_moon)**3)
    
    # Sum
    ax_total = a_2body[0] + ax_j2 + ax_j3 + ax_j4 + ax_drag + a_sun[0] + a_moon[0]
    ay_total = a_2body[1] + ay_j2 + ay_j3 + ay_j4 + ay_drag + a_sun[1] + a_moon[1]
    az_total = a_2body[2] + az_j2 + az_j3 + az_j4 + az_drag + a_sun[2] + a_moon[2]
    
    return [vx, vy, vz, ax_total, ay_total, az_total]

def propagate_cowell(initial_r, initial_v, t_eval, epoch_jd, use_j2=True, use_j3=True, use_j4=True, use_drag=True, use_sun=True, use_moon=True,h_scale=60,rho_ref=1e-13,h_ref=500,omega_e=7.2921159e-5,cd_a_m=0.012):
    y0 = np.concatenate([initial_r, initial_v])
    t_span = (t_eval[0], t_eval[-1])
    
    # Pass epoch_jd and perturbation flags to acceleration function
    sol = solve_ivp(
        lambda t, y: acceleration_extended(t, y, epoch_jd, use_j2, use_j3, use_j4, use_drag, use_sun, use_moon,h_scale,rho_ref,h_ref,omega_e,cd_a_m),
        t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9
    )
    
    return sol.y.T

def eci_to_ric(r_ref, v_ref, r_obj, v_obj):
    """
    Calculate RIC (Radial, In-Track, Cross-Track) frame difference.
    ref: Reference orbit (SGP4)
    obj: Object orbit (Cowell)
    """
    diff_eci = r_obj - r_ref
    
    ric_diffs = []
    
    for i in range(len(r_ref)):
        r = r_ref[i]
        v = v_ref[i]
        
        # Radial unit vector
        u_r = r / np.linalg.norm(r)
        
        # Cross-track unit vector (angular momentum direction)
        h = np.cross(r, v)
        u_c = h / np.linalg.norm(h)
        
        # In-track unit vector
        u_i = np.cross(u_c, u_r)
        
        # Transformation matrix from ECI to RIC
        # Rows are the unit vectors
        M = np.array([u_r, u_i, u_c])
        
        # Rotate difference vector
        d_ric = M @ diff_eci[i]
        ric_diffs.append(d_ric)
        
    return np.array(ric_diffs)

def make_legend_interactive(ax, lines):
    leg = ax.legend()
    lined = dict()
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)
        legline.set_pickradius(5)
        lined[legline] = origline

    def onpick(event):
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        ax.figure.canvas.draw()

    ax.figure.canvas.mpl_connect('pick_event', onpick)
    return leg

class OrbitPropagationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Orbit Propagation - SGP4 vs Cowell")
        self.root.geometry("1400x900")
        
        self.fig = None
        self.canvas = None
        self.toolbar = None
        self.ric=None
        
        # Predefined TLE options
        self.tle_options = {
            "Cartosat-3 (NORAD 44804)": {
                "line1": "1 44804U 19081A   25327.85928487  .00004606  00000-0  22108-3 0  9991",
                "line2": "2 44804  97.4606  27.6788 0013715 149.1101 211.0943 15.19283085332297"
            },
            "ISS (ZARYA)": {
                "line1": "1 25544U 98067A   24001.51007060  .00020194  00000-0  36276-3 0  9996",
                "line2": "2 25544  51.6403  59.2826 0008551  77.7409  35.2029 15.49859211428906"
            },
            "NOAA 18": {
                "line1": "1 28654U 05018A   24001.73993056  .00000072  00000-0  00000-0 0  9997",
                "line2": "2 28654  99.0605 158.6265 0013368  58.3105 301.9365 14.12568918311559"
            },
            "Custom TLE": {
                "line1": "",
                "line2": ""
            }
        }
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        # control_frame = ttk.LabelFrame(main_frame, text="Propagation Controls", padding=10)
        # control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # ==== Scrollable Left Panel ====
        left_canvas = tk.Canvas(main_frame)
        left_canvas.pack(side=tk.LEFT, fill=tk.Y)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=left_canvas.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas.configure(yscrollcommand=scrollbar.set)
        left_canvas.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))

        scroll_frame = ttk.Frame(left_canvas)
        left_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        # All controls go into scroll_frame instead of main_frame directly
        control_frame = ttk.LabelFrame(scroll_frame, text="Propagation Controls", padding=10)
        control_frame.pack(fill=tk.Y, padx=10, pady=10)
        
        # TLE Selection
        tle_frame = ttk.LabelFrame(control_frame, text="TLE Selection", padding=5)
        tle_frame.pack(fill=tk.X, pady=5)
        
        self.tle_var = tk.StringVar(value="Cartosat-3 (NORAD 44804)")
        tle_combo = ttk.Combobox(tle_frame, textvariable=self.tle_var,
                                values=list(self.tle_options.keys()), state="readonly", width=35)
        tle_combo.pack(fill=tk.X, pady=2)
        tle_combo.bind('<<ComboboxSelected>>', self.on_tle_selection)
        
        # TLE Input Fields
        ttk.Label(tle_frame, text="TLE Line 1:").pack(anchor=tk.W, pady=(10, 0))
        self.tle_line1_var = tk.StringVar()
        self.tle_line1_entry = scrolledtext.ScrolledText(tle_frame, height=2, width=40, wrap=tk.WORD)
        self.tle_line1_entry.pack(fill=tk.X, pady=2)
        
        ttk.Label(tle_frame, text="TLE Line 2:").pack(anchor=tk.W, pady=(5, 0))
        self.tle_line2_var = tk.StringVar()
        self.tle_line2_entry = scrolledtext.ScrolledText(tle_frame, height=2, width=40, wrap=tk.WORD)
        self.tle_line2_entry.pack(fill=tk.X, pady=2)
        
        # Propagation Parameters
        param_frame = ttk.LabelFrame(control_frame, text="Propagation Parameters", padding=5)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Duration (hours):").pack(anchor=tk.W)
        self.duration_var = tk.StringVar(value="24")
        ttk.Entry(param_frame, textvariable=self.duration_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(param_frame, text="Step Size (seconds):").pack(anchor=tk.W)
        self.step_size_var = tk.StringVar(value="60")
        ttk.Entry(param_frame, textvariable=self.step_size_var).pack(fill=tk.X, pady=2)

        #------------------------------------------------------------------------------------
        # Source Selection (TLE or CSV)
        source_frame = ttk.LabelFrame(control_frame, text="Input Source", padding=5)
        source_frame.pack(fill=tk.X, pady=5)

        self.source_var = tk.StringVar(value="TLE")

        ttk.Radiobutton(source_frame, text="Use TLE", variable=self.source_var,
                        value="TLE").pack(anchor=tk.W)

        ttk.Radiobutton(source_frame, text="Use CSV File", variable=self.source_var,
                        value="CSV").pack(anchor=tk.W)

        # CSV selector button
        self.csv_path_var = tk.StringVar(value="")
        ttk.Button(source_frame, text="Select CSV File",
                command=self.select_csv_file).pack(fill=tk.X, pady=5)
        
        # Label to show selected CSV file name
        self.csv_label = ttk.Label(source_frame, text="No file selected", foreground="blue")
        self.csv_label.pack(anchor=tk.W, pady=(0, 5))

        #---------------------------------------------------------------------------------------
        button_frame1 = ttk.Frame(control_frame)
        button_frame1.pack(fill=tk.X, pady=6)
        ttk.Button(button_frame1, text="Start Propagation", 
                  command=self.start_propagation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame1, text="Save plots", 
                   command=self.save_states_to_csv).pack(fill=tk.X, pady=2)
        #----------------------------------------------------------------------------------------
        self.H_SCALE_var = tk.DoubleVar(value=60.0)  # H_SCALE (in km)
        self.RHO_REF_var = tk.DoubleVar(value=1e-13)  # RHO_REF (kg/m^3)
        self.H_REF_var = tk.DoubleVar(value=500)  # H_REF (in km)
        self.OMEGA_E_var = tk.DoubleVar(value=7.2921159e-5)  # OMEGA_E
        self.B_STAR_var = tk.DoubleVar(value=0.00022108)  # B_STAR
        self.CD_A_M_var = tk.DoubleVar(value=0.012)  # CD_A_M
        
        # Perturbation Options
        pert_frame = ttk.LabelFrame(control_frame, text="Perturbation Options", padding=5)
        pert_frame.pack(fill=tk.X, pady=5)
        
        self.j2_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="J2 Perturbation", variable=self.j2_var).pack(anchor=tk.W)
        
        self.j3_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="J3 Perturbation", variable=self.j3_var).pack(anchor=tk.W)
        
        self.j4_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="J4 Perturbation", variable=self.j4_var).pack(anchor=tk.W)
        
        self.drag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="Atmospheric Drag", variable=self.drag_var).pack(anchor=tk.W)
        
        self.sun_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="Sun Gravity", variable=self.sun_var).pack(anchor=tk.W)
        
        self.moon_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pert_frame, text="Moon Gravity", variable=self.moon_var).pack(anchor=tk.W)
        
        # New input fields for the additional parameters
        ttk.Label(pert_frame, text="H_SCALE (km):").pack(anchor=tk.W, pady=(10, 0))
        ttk.Entry(pert_frame, textvariable=self.H_SCALE_var).pack(fill=tk.X, pady=2)

        ttk.Label(pert_frame, text="RHO_REF (kg/m^3):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Entry(pert_frame, textvariable=self.RHO_REF_var).pack(fill=tk.X, pady=2)

        ttk.Label(pert_frame, text="H_REF (km):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Entry(pert_frame, textvariable=self.H_REF_var).pack(fill=tk.X, pady=2)

        ttk.Label(pert_frame, text="OMEGA_E:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Entry(pert_frame, textvariable=self.OMEGA_E_var).pack(fill=tk.X, pady=2)

        ttk.Label(pert_frame, text="B_STAR:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Entry(pert_frame, textvariable=self.B_STAR_var).pack(fill=tk.X, pady=2)

        ttk.Label(pert_frame, text="CD_A_M:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Entry(pert_frame, textvariable=self.CD_A_M_var).pack(fill=tk.X, pady=2)

        
        
        # Control Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).pack(fill=tk.X, pady=2)
        
        # Status Display
        status_frame = ttk.LabelFrame(control_frame, text="Status & Results", padding=5)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=15, width=40)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for plots
        self.plot_frame = ttk.LabelFrame(main_frame, text="Propagation Results", padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Initialize with default TLE
        self.on_tle_selection()
    
    def select_csv_file(self):
        path = filedialog.askopenfilename(
            title="Select Clean 24hr CSV",
            filetypes=[("CSV Files", "*.csv")]
        )
        if path:
            self.csv_path_var.set(path)
            self.log_status(f"Selected CSV file: {path}")

            # Show file name next to button
            file_name = os.path.basename(path)
            self.csv_label.config(text=f"Selected: {file_name}")
        else:
            self.csv_label.config(text="No file selected")

    def load_csv_states(self, csv_path):
        df = pd.read_csv(csv_path)

        # ------------------------------------------------------------
        # 1) Convert timestamp to seconds from epoch t0
        # ------------------------------------------------------------
        t_abs = pd.to_datetime(df["timestamp"]).astype("int64").values / 1e9
        t_eval = t_abs - t_abs[0]
        print(t_eval[:10])
        print(len(t_eval))

        # ------------------------------------------------------------
        # 2) Extract position (must exist)
        # ------------------------------------------------------------
        r = df[["px_km", "py_km", "pz_km"]].values.astype(float)

        # ------------------------------------------------------------
        # 3) Velocity columns check
        # ------------------------------------------------------------
        vel_cols = ["vx_km_s", "vy_km_s", "vz_km_s"]
        has_vel = all(col in df.columns for col in vel_cols)

        if has_vel:
            # ✔ Use existing velocities directly
            v = df[vel_cols].values.astype(float)

        else:
            # ------------------------------------------------------------
            # ❌ No velocity columns → compute them from positions & time
            # ------------------------------------------------------------
            n = len(df)
            v = np.zeros((n, 3), dtype=float)   # initialize v1 = 0,0,0 automatically

            for i in range(1, n):
                dt = t_abs[i] - t_abs[i - 1] 
                if i<=10:
                   print(dt)    # seconds
                if dt == 0:
                    # avoid divide-by-zero
                    v[i] = v[i - 1]
                else:
                    dr = r[i] - r[i - 1]         # km difference
                    v[i] = dr / dt               # km/s

            # v[0] stays [0,0,0] as required

        # ------------------------------------------------------------
        # 4) Epoch is the first timestamp (datetime object)
        # ------------------------------------------------------------
        epoch = pd.to_datetime(df["timestamp"].iloc[0]).to_pydatetime()

        return t_eval, r, v, epoch
    
    def save_states_to_csv(self):
        
        if self.ric is None:
            messagebox.showinfo("Error", "RIC values are not available!")
            return
        folder = filedialog.askdirectory(title="Select folder to save CSV files")
        if not folder:
            return

        try:
            df_ric_cowell= pd.DataFrame(self.ric, columns=["dR","dI","dC"])
            df_ric_cowell.to_csv(f"{folder}/RIC.csv", index=False)
            messagebox.showinfo("Success", "Files saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_tle_selection(self, event=None):
        selected = self.tle_var.get()
        if selected in self.tle_options:
            tle_data = self.tle_options[selected]
            self.tle_line1_entry.delete(1.0, tk.END)
            self.tle_line1_entry.insert(1.0, tle_data["line1"])
            self.tle_line2_entry.delete(1.0, tk.END)
            self.tle_line2_entry.insert(1.0, tle_data["line2"])
    
    def log_status(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def clear_results(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()
        self.status_text.delete(1.0, tk.END)
        self.log_status("Ready for new propagation...")
    
    def start_propagation(self):
        try:
            self.clear_results()
            self.log_status("Starting propagation...")
            
            # Get TLE data
            tle_line1 = self.tle_line1_entry.get(1.0, tk.END).strip()
            tle_line2 = self.tle_line2_entry.get(1.0, tk.END).strip()
            
            if not tle_line1 or not tle_line2:
                messagebox.showerror("Error", "Please provide valid TLE data")
                return
            
            self.log_status(f"TLE Line 1: {tle_line1}")
            self.log_status(f"TLE Line 2: {tle_line2}")
            
            # Get parameters
            try:
                duration_hours = float(self.duration_var.get())
                step_seconds = float(self.step_size_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid duration or step size")
                return
            
            # Run propagation
            source = self.source_var.get()
            self.log_status("\n1. Getting SGP4 states...")
            # t_eval, r_sgp4, v_sgp4, epoch = get_sgp4_states(tle_line1, tle_line2, 
            #                                                 duration_hours, step_seconds)
            r_inital=None
            v_initial=None
            r_prop=None
            v_prop=None
            if source == "TLE":
                self.log_status("Using TLE for propagation...")

                t_eval, r_sgp4, v_sgp4, epoch = get_sgp4_states(
                    tle_line1, tle_line2,
                    duration_hours, step_seconds
                )
                r_inital=r_sgp4[0]
                v_initial=v_sgp4[0]
                t_eval=t_eval
                r_sgp4=r_sgp4
                v_sgp4=v_sgp4

            else:  # CSV
                if not self.csv_path_var.get():
                    messagebox.showerror("Error", "Please select a CSV file")
                    return

                self.log_status("Using CSV file as input states...")
                csv_path = self.csv_path_var.get()

                t_eval, r_sgp4, v_sgp4, epoch = self.load_csv_states(csv_path)
                r1,r2,r3,t1,t2,t3=r_sgp4[0],r_sgp4[1],r_sgp4[2],t_eval[0],t_eval[1],t_eval[2]
                v_hg = herrick_gibbs(r1, r2, r3, t1, t2, t3)
                r_inital=r2
                v_initial=v_hg
                t_eval=t_eval[1:]
                r_sgp4=r_sgp4[1:]
                v_sgp4=v_sgp4[1:]

            # r1,r2,r3,t1,t2,t3=r_sgp4[0],r_sgp4[1],r_sgp4[2],t_eval[0],t_eval[1],t_eval[2]
            # v_hg = herrick_gibbs(r1, r2, r3, t1, t2, t3)
            
            # Calculate Epoch JD
            jd_epoch, fr_epoch = jday(epoch.year, epoch.month, epoch.day, 
                                     epoch.hour, epoch.minute, epoch.second)
            full_epoch_jd = jd_epoch + fr_epoch
            
            self.log_status(f"Epoch: {epoch}")
            # self.log_status(f"Initial Position: {r_sgp4[0]}")
            # self.log_status(f"Initial Velocity: {v_sgp4[0]}")
            self.log_status(f"Initial Position: {r_inital}")
            self.log_status(f"Initial Velocity: {v_initial}")
            
            # Get perturbation options
            use_j2 = self.j2_var.get()
            use_j3 = self.j3_var.get()
            use_j4 = self.j4_var.get()
            use_drag = self.drag_var.get()
            use_sun = self.sun_var.get()
            use_moon = self.moon_var.get()
            h_scale=self.H_SCALE_var.get()
            rho_ref=self.RHO_REF_var.get()
            h_ref=self.H_REF_var.get()
            omega_e=self.OMEGA_E_var.get()
            b_star=self.B_STAR_var.get()
            cd_a_m=self.CD_A_M_var.get()
            
            # Log selected perturbations
            self.log_status("\nSelected Perturbations:")
            self.log_status(f"  J2: {use_j2}")
            self.log_status(f"  J3: {use_j3}")
            self.log_status(f"  J4: {use_j4}")
            self.log_status(f"  Atmospheric Drag: {use_drag}")
            self.log_status(f"  Sun Gravity: {use_sun}")
            self.log_status(f"  Moon Gravity: {use_moon}")
            self.log_status(f"  H_SCALE: {h_scale}")
            self.log_status(f"  RHO_REF: {rho_ref}")
            self.log_status(f"  H_REF: {h_ref}")
            self.log_status(f"  OMEGA_E: {omega_e}")
            self.log_status(f"  B_STAR: {b_star}")
            self.log_status(f"  CD_A_M: {cd_a_m}")
            
            # Propagate using Cowell's Method
            self.log_status("\n2. Running Cowell propagation...")
            # cowell_states = propagate_cowell(r_sgp4[0], v_sgp4[0], t_eval, full_epoch_jd,
            #                                 use_j2, use_j3, use_j4, use_drag, use_sun, use_moon)
            # t_eval=t_eval[1:]
            # r_sgp4=r_sgp4[1:]
            # v_sgp4=v_sgp4[1:]
            cowell_states = propagate_cowell(r_inital, v_initial, t_eval, full_epoch_jd,
                                            use_j2, use_j3, use_j4, use_drag, use_sun, use_moon,h_scale,rho_ref,h_ref,omega_e,cd_a_m)
            print("shape of cowell:",np.shape(cowell_states))
            r_cowell = cowell_states[:, :3]
            v_cowell = cowell_states[:, 3:6]
            print(cowell_states[0])
            
            # Calculate RIC Differences
            self.log_status("\n3. Calculating RIC differences...")
            ric_diff = eci_to_ric(r_sgp4, v_sgp4, r_cowell, np.zeros_like(r_cowell))
            self.ric=ric_diff
            
            # Create plots
            self.log_status("\n4. Creating plots...")
            self.create_plots(r_sgp4, r_cowell, ric_diff, t_eval)
            
            # Statistics
            pos_diff = np.linalg.norm(r_sgp4 - r_cowell, axis=1)
            self.log_status("\n=== RESULTS ===")
            self.log_status(f"Duration: {duration_hours} hours")
            self.log_status(f"Final position difference: {pos_diff[-1]:.6f} km")
            self.log_status(f"Max position difference: {np.max(pos_diff):.6f} km")
            self.log_status(f"Avg position difference: {np.mean(pos_diff):.6f} km")
            self.log_status(f"Final in-track error: {ric_diff[-1, 1]:.6f} km")
            self.log_status("\nPropagation complete!")
            
        except Exception as e:
            self.log_status(f"\nERROR: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Propagation failed: {str(e)}")
    
    def create_plots(self, r_sgp4, r_cowell, ric_diff, t_eval):
        # Create figure
        self.fig = Figure(figsize=(12, 6))
        
        # 3D Orbit Plot
        ax1 = self.fig.add_subplot(121, projection='3d')
        
        # Plot Earth
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = RE * np.cos(u) * np.sin(v)
        y = RE * np.sin(u) * np.sin(v)
        z = RE * np.cos(v)
        ax1.plot_wireframe(x, y, z, color='gray', alpha=0.3)
        
        # Plot orbits
        ax1.plot(r_sgp4[:, 0], r_sgp4[:, 1], r_sgp4[:, 2], 
                label='SGP4', color='blue', linewidth=1)
        ax1.plot(r_cowell[:, 0], r_cowell[:, 1], r_cowell[:, 2], 
                label='Cowell', color='red', linestyle='--', linewidth=1)
        
        ax1.set_title('3D Orbit Propagation (24h)')
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Z (km)')
        ax1.legend()
        
        # RIC Differences Plot
        ax2 = self.fig.add_subplot(122)
        hours = t_eval / 3600
        
        ax2.plot(hours, ric_diff[:, 0], label='Radial', linewidth=2)
        ax2.plot(hours, ric_diff[:, 1], label='In-Track', linewidth=2)
        ax2.plot(hours, ric_diff[:, 2], label='Cross-Track', linewidth=2)
        
        ax2.set_title('Difference (Cowell - SGP4) in RIC Frame')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Difference (km)')
        ax2.grid(True)
        ax2.legend()
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

def main():
    root = tk.Tk()
    app = OrbitPropagationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

