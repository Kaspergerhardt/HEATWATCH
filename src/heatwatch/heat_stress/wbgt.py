"""Outdoor Wet Bulb Globe Temperature (WBGT) with the Liljegren method, as used by the KNMI.

    WBGT = 0.7 * T_natural_wet_bulb + 0.2 * T_globe + 0.1 * T_air

Inputs per moment: air temperature (degC), relative humidity (%), air pressure (hPa), wind speed at 10 m
(m/s), global (shortwave) radiation (W/m2), position and UTC time. T_globe and T_natural_wet_bulb are
solved iteratively from the heat balance of a black globe and a wet wick (Liljegren et al. 2008,
J Occup Environ Hyg 5:645-655; constants and formulas from the original Argonne C code). The fraction of
direct sunlight and the 2-m wind speed are estimated as in that code (no extra data needed).
"""
import math

# physical constants (Liljegren)
STEFANB, CP, M_AIR, M_H2O, R_GAS = 5.6696e-8, 1003.5, 28.97, 18.015, 8314.34
R_AIR = R_GAS / M_AIR
RATIO = CP * M_AIR / M_H2O
PR = CP / (CP + 1.25 * R_AIR)
SOLAR_CONST, CZA_MIN, NORMSOLAR_MAX, MIN_SPEED, REF_HEIGHT = 1367.0, 0.00873, 0.85, 0.13, 2.0
EMIS_WICK, ALB_WICK, D_WICK, L_WICK = 0.95, 0.4, 0.007, 0.0254
EMIS_GLOBE, ALB_GLOBE, D_GLOBE = 0.95, 0.05, 0.0508
EMIS_SFC, ALB_SFC = 0.999, 0.45
CONVERGENCE, MAX_ITER = 0.02, 50


# ------------------------------------------------------------------ sun
def sun(t_utc, lat, lon):
    """cos(solar zenith angle) and sun-earth distance (AU) for a UTC time (NOAA equations)."""
    doy = t_utc.timetuple().tm_yday
    hour = t_utc.hour + t_utc.minute / 60 + t_utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)                       # fractional year
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))   # equation of time, min
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
           + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)                     # hour angle
    la = math.radians(lat)
    cza = math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)
    alt = math.degrees(math.asin(max(min(cza, 1.0), -1.0)))                    # elevation, degrees
    if alt >= -1.0:                                                              # refraction (as Liljegren)
        if alt < 19.225:
            alt += (0.1594 + alt * (0.0196 + 0.00002 * alt)) * 1013.25 / ((1 + alt * (0.505 + 0.0845 * alt)) * 288.0)
        else:
            alt += 0.00452 * (1013.25 / 288.0) / math.tan(math.radians(alt))
    cza = math.sin(math.radians(alt))
    dist = 1.00014 - 0.01671 * math.cos(2 * math.pi * (doy - 4) / 365.25)
    return cza, dist


def solar_parameters(solar, cza, dist):
    """Limit radiation to a physical maximum and estimate the direct-beam fraction (Liljegren)."""
    toa = SOLAR_CONST * max(0.0, cza) / dist ** 2 if cza >= CZA_MIN else 0.0
    if toa <= 0:
        return 0.0, 0.0
    norm = min(solar / toa, NORMSOLAR_MAX)
    fdir = max(min(math.exp(3 - 1.34 * norm - 1.65 / norm), 0.9), 0.0) if norm > 0 else 0.0
    return norm * toa, fdir


# ------------------------------------------------------------------ properties of air / water
def esat(tk):                        # saturation vapour pressure over water, hPa (Buck 1981)
    return 1.004 * 6.1121 * math.exp(17.502 * (tk - 273.15) / (tk - 32.18))


def dew_point(e):                    # K
    z = math.log(e / (6.1121 * 1.004))
    return 273.15 + 240.97 * z / (17.502 - z)


def viscosity(tk):
    """Viscosity of air (kg/m/s)."""
    omega = (tk / 97.0 - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * math.sqrt(M_AIR * tk) / (3.617 ** 2 * omega)


def thermal_cond(tk):
    """Thermal conductivity of air (W/m/K)."""
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, p):
    """Diffusivity of water vapour in air (m2/s)."""
    pcrit13 = (36.4 * 218.0) ** (1 / 3)
    tcrit512 = (132.0 * 647.3) ** (5 / 12)
    tcrit12 = math.sqrt(132.0 * 647.3)
    mmix = math.sqrt(1 / M_AIR + 1 / M_H2O)
    return 3.640e-4 * (tk / tcrit12) ** 2.334 * pcrit13 * tcrit512 * mmix / (p / 1013.25) * 1e-4


def evap(tk):                        # heat of evaporation, J/kg
    return (313.15 - tk) / 30 * (-71100.0) + 2.4073e6


def emis_atm(tk, rh):
    """Emissivity of the atmosphere."""
    return 0.575 * (rh * esat(tk)) ** 0.143


def h_sphere(tk, p, speed):
    """Heat transfer coefficient of a sphere (the globe), W/m2/K."""
    re = max(speed, MIN_SPEED) * (p * 100 / (R_AIR * tk)) * D_GLOBE / viscosity(tk)
    return (2.0 + 0.6 * math.sqrt(re) * PR ** 0.3333) * thermal_cond(tk) / D_GLOBE


def h_cylinder(tk, p, speed):
    """Heat transfer coefficient of a cylinder (the wet wick), W/m2/K."""
    re = max(speed, MIN_SPEED) * (p * 100 / (R_AIR * tk)) * D_WICK / viscosity(tk)
    return 0.281 * re ** 0.6 * PR ** 0.44 * thermal_cond(tk) / D_WICK


# ------------------------------------------------------------------ wind at 2 m
def wind_2m(speed10, solar, daytime):
    """10-m wind -> 2-m wind with a power law; exponent from the stability class (EPA-454/5-99-005, rural)."""
    if daytime:
        j = 0 if solar >= 925 else 1 if solar >= 675 else 2 if solar >= 175 else 3
        i = 4 if speed10 >= 6 else 3 if speed10 >= 5 else 2 if speed10 >= 3 else 1 if speed10 >= 2 else 0
    else:                                          # night: no vertical temperature gradient known -> stable
        j = 6
        i = 2 if speed10 >= 2.5 else 1 if speed10 >= 2 else 0
    table = [[1, 1, 2, 4, 0, 5, 6], [1, 2, 3, 4, 0, 5, 6], [2, 2, 3, 4, 0, 4, 4], [3, 3, 4, 4, 0, 0, 0],
             [3, 4, 4, 4, 0, 0, 0]]
    cls = table[i][j]
    exponent = [0.07, 0.07, 0.10, 0.15, 0.35, 0.55][cls - 1]
    return max(speed10 * (REF_HEIGHT / 10.0) ** exponent, MIN_SPEED)


# ------------------------------------------------------------------ globe and natural wet bulb
def t_globe(tk, rh, p, speed, solar, fdir, cza):
    """Globe temperature (degC), solved iteratively from its heat balance."""
    tg = tk
    for _ in range(MAX_ITER):
        tref = 0.5 * (tg + tk)
        h = h_sphere(tref, p, speed)
        direct = fdir * (1 / (2 * cza) - 1) if cza > 0 else 0.0
        new = (0.5 * (emis_atm(tk, rh) * tk ** 4 + EMIS_SFC * tk ** 4) - h / (STEFANB * EMIS_GLOBE) * (tg - tk)
               + solar / (2 * STEFANB * EMIS_GLOBE) * (1 - ALB_GLOBE) * (direct + 1 + ALB_SFC)) ** 0.25
        if abs(new - tg) < CONVERGENCE:
            return new - 273.15
        tg = 0.9 * tg + 0.1 * new
    return float("nan")


def t_natural_wet_bulb(tk, rh, p, speed, solar, fdir, cza):
    """Natural wet bulb temperature (degC), solved iteratively from its heat balance."""
    eair = rh * esat(tk)
    tw = dew_point(eair)
    sza = math.acos(max(min(cza, 1.0), -1.0))
    for _ in range(MAX_ITER):
        tref = 0.5 * (tw + tk)
        h = h_cylinder(tref, p, speed)
        fatm = (STEFANB * EMIS_WICK * (0.5 * (emis_atm(tk, rh) * tk ** 4 + EMIS_SFC * tk ** 4) - tw ** 4)
                + (1 - ALB_WICK) * solar * ((1 - fdir) * (1 + 0.25 * D_WICK / L_WICK)
                                            + fdir * (math.tan(sza) / math.pi + 0.25 * D_WICK / L_WICK) + ALB_SFC))
        ewick = esat(tw)
        density = p * 100 / (R_AIR * tref)
        sc = viscosity(tref) / (density * diffusivity(tref, p))
        new = tk - evap(tref) / RATIO * (ewick - eair) / (p - ewick) * (PR / sc) ** 0.56 + fatm / h
        if abs(new - tw) < CONVERGENCE:
            return new - 273.15
        tw = 0.9 * tw + 0.1 * new
    return float("nan")


# ------------------------------------------------------------------ public
def wbgt(t_air, rh_pct, pressure, wind10, radiation, lat, lon, t_utc):
    """Outdoor WBGT in degC (NaN if an input is missing)."""
    vals = (t_air, rh_pct, pressure, wind10, radiation, lat, lon)
    if any(v is None or v != v for v in vals):
        return float("nan")
    cza, dist = sun(t_utc, lat, lon)
    solar, fdir = solar_parameters(max(radiation, 0.0), cza, dist)
    speed = wind_2m(wind10, solar, cza > 0)
    tk, rh = t_air + 273.15, max(min(rh_pct, 100.0), 1.0) / 100
    tg = t_globe(tk, rh, pressure, speed, solar, fdir, cza)
    tnwb = t_natural_wet_bulb(tk, rh, pressure, speed, solar, fdir, cza)
    return 0.7 * tnwb + 0.2 * tg + 0.1 * t_air


def indoor_wbgt(t_air, rh_pct, speed=0.5, pressure=1013.25):
    """Indoor WBGT without sun (ISO 7243): 0.7 * natural wet bulb + 0.3 * globe. Indoors the globe takes
    the temperature of air and walls (globe = air temperature); the wet wick exchanges radiation with
    walls at air temperature instead of with the sky."""
    tk, rh = t_air + 273.15, rh_pct / 100
    eair = rh * esat(tk)
    tw = dew_point(eair)
    for _ in range(MAX_ITER):
        tref = 0.5 * (tw + tk)
        h = h_cylinder(tref, pressure, speed)
        fatm = STEFANB * EMIS_WICK * (tk ** 4 - tw ** 4)
        ewick = esat(tw)
        sc = viscosity(tref) / ((pressure * 100 / (R_AIR * tref)) * diffusivity(tref, pressure))
        new = tk - evap(tref) / RATIO * (ewick - eair) / (pressure - ewick) * (PR / sc) ** 0.56 + fatm / h
        if abs(new - tw) < CONVERGENCE:
            break
        tw = 0.9 * tw + 0.1 * new
    return 0.7 * (new - 273.15) + 0.3 * t_air
