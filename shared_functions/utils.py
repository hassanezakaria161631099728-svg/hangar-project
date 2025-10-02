import numpy as np
import pandas as pd
def comb(G,Q1,Qi,cas,a,psi0):
    gamg = 1.35
    gamq = 1.5
    # Get psi factors from Excel
    if cas == "ELU":
        S = (gamg * G + gamq * Q1 + gamq * psi0 * Qi) * a
        if Q1 < 0:
            gamg = 1
            S = -1 * (gamg * G + gamq * Q1 + gamq * psi0 * Qi) * a
    elif cas == "ELS":
        S = (G + Q1 + psi0 * Qi) * a
    else:
        raise ValueError(f"Unknown case: {cas}")
    return S

def snow(geo, ba, b, Lx, Ly):
    alpha = ba["slope_angle_deg"].iloc[0]   # ba{1,5} in MATLAB
    zone = geo["snow_zone"].iloc[0]   # geo{1,3}
    H = geo["altitude"].iloc[0]      # geo{1,4}
    # --- snow characteristic load (kN/m^2) ---
    if zone == 'A':
        sk = (0.07 * H + 15) / 100.0
    elif zone == 'B':
        sk = (0.04 * H + 10) / 100.0
    elif zone == 'C':
        sk = 0.0325 * H / 100.0
    elif zone == 'D':
        sk = 0.0
        print("no snow load")
    else:
        raise ValueError(f"Unknown zone type: {zone}")
    # convert to daN/m^2
    sk1 = sk * 100.0
    # --- roof slope form factor ---
    if 0 <= alpha <= 30:
        u = 0.8
    else:
        u = 1.0  # MATLAB code only defined case for [0,30], you may refine this
    s = u * sk1  # daN/m^2
    # --- select direction ---
    if b == Ly:
        res = s
    elif b == Lx:
        res = 0.5 * s
    else:
        res = 0.0  # fallback if b is neither Lx nor Ly
    return res

def beam3(L,q,T):
    lb = L / 250 * 100  # cm
    Ea = 2.1e5  # MPa
    Iymin = (5 * 250 / 384) * (q / 100) * (L * 100) ** 3 / (Ea * 10)  # cm^4
    sb = beam4(Iymin, 10, T)  # Call beam4
    return sb, Iymin, lb

def beam4(minval,column,T):
    # Adjust MATLAB 1-based column index to Python 0-based
    col_idx = column - 1
    vals = pd.to_numeric(T.iloc[:, col_idx], errors="coerce")
    valid_rows = vals[vals >= minval]
    if valid_rows.empty:
        print("No rows satisfy the condition.")
        return None
    else:
        t_val = valid_rows.min()
        t_idx = vals[vals == t_val].index[0]
    return T.loc[[t_idx]]

def fvplrd(cas, sb, b, Vsd):
    # Extract properties (MATLAB sb{1,n} corresponds to sb.iloc[n-1])
    bf = sb["b"].iloc[0]   # flange width (cm)
    tw = sb["tw"].iloc[0]   # web thickness (cm)
    tf = sb["tf"].iloc[0]   # flange thickness (cm)
    r  = sb["r"].iloc[0]   # radius (cm)
    A  = sb["A"].iloc[0]   # cross-sectional area (cm^2)
    if cas == "y":
        if b == "UPE":
            p = 1
        else:
            p = 2
        Av = A - 2 * bf * tf + (tw + p * r) * tf
    elif cas == "z":
        Av = 2 * bf * tf
    else:
        raise ValueError("cas must be 'y' or 'z'")
    gamma = 1.1
    fy = 2350  # daN/m
    Vplrd = Av * fy / np.sqrt(3) / gamma  # kg
    # Check shear resistance
    if Vsd < 0.5 * Vplrd:
        if cas == "y":
            print("shear resistance on y axis has been aquired")
        elif cas == "z":
            print("shear resistance on z axis has been aquired")
    return Av, Vplrd

def bending(del2, delmax, L):
    """
    Check bending resistance.
    Parameters
    ----------
    del2 : float
        Deformation case 2
    delmax : float
        Maximum deformation
    L : float
        Span length (m)
    Returns
    -------
    lb1 : float
        Limit deformation L/2
    lb2 : float
        Limit deformation L/2.5
    """
    lb1 = L / 2
    lb2 = L / 2.5
    if del2 < lb2 and delmax < lb1:
        print("Bending resistance has been acquired")
    return lb1, lb2

def resistance(T, Mysd, Mzsd):
    """
    Check resistance based on plastic section modulus.
    Parameters
    ----------
    T : pd.Series or list
        Section properties (row from profile table).
        wply expected at index 17, wplz at index 18 (MATLAB 18,19).
    Mysd : float
        Design bending moment about y-axis
    Mzsd : float
        Design bending moment about z-axis
    Returns
    -------
    r : float
        Resistance ratio (must be <= 1 to be acceptable)
    """
    fy = 2350  # daN/m
    row = T.squeeze()
    wply = row["wply"]  # plastic section modulus about y
    wplz = row["wplz"]  # plastic section modulus about z
    r = (Mysd / (wply * fy / 1.1) + Mzsd / (wplz * fy / 1.1)) * 100
    if r <= 1:
        print("Resistance has been acquired")
    return r

def lateral_torsional_buckling(L, lazb, lt, Nsd, xiz, fy, T, Mzscorr, kz, Mysdn):
    """
    Vérification de la stabilité au déversement (lateral-torsional buckling).
    Parameters
    ----------
    L : float
        Beam span (m)
    lazb : float
        Slenderness parameter
    lt : str
        Boundary condition ('double fixed' or 'simplement appuye')
    Nsd : float
        Axial force (daN)
    xiz : float
        Reduction factor
    fy : float
        Yield strength (daN/cm²?)
    T : pd.Series or list
        Section properties (row from profile table).
        Expected: A at index 8, h at 1, Iz at 13, It at 16, Wpl,y at 17, Wpl,z at 18, Iw at 19.
    Mzscorr : float
        Corrected bending moment about z-axis
    kz : float
        Buckling factor for z-axis
    Mysdn : float
        Design bending moment about y-axis
    Returns
    -------
    tuple
        (ult, klt, L1, Lfz, zg, k, c1, c2, lalt, laltb, alphalt, philt, xlt, r)
    """
    # Extract beam properties (MATLAB is 1-based, Python is 0-based)
    row = T.squeeze()  # converts 1-row DataFrame → Series
    A = row["A"]   # cm^2
    h = row["h"]   # cm
    Iz = row["Iz"] # cm^4
    It = row["It"] # cm^4
    wply = row["wply"] # cm^3
    wplz = row["wplz"] # cm^3
    Iw = row["Iw"]  # cm^6 ?
    betam = 1.3
    ult = 0.15 * (lazb * betam - 1)
    klt = 1 - (ult * Nsd) / (xiz * A * fy)
    if klt < 1.5 and ult < 0.9:
        print("On peut vérifier la stabilité au déversement")
    # Characteristic lengths
    L1 = L * 100  # cm
    zg = -h / 2   # cm
    if lt == "double fixed":
        Lfz = 0.5 * L1
    elif lt == "simplement appuye":
        Lfz = L1
    else:
        raise ValueError("lt must be 'double fixed' or 'simplement appuye'")
    k = Lfz / L1
    if np.isclose(k, 1):
        c1, c2 = 1.132, 0.459
    elif np.isclose(k, 0.5):
        c1, c2 = 0.972, 0.304
    else:
        c1, c2 = np.nan, np.nan  # undefined case
    # Stability calculations
    lt1 = wply**2 / (Iz * Iw)
    lt2 = lt1**0.25
    lt3 = k * L1 * lt2
    lt4 = ((k * L1) ** 2 * It) / (np.pi**2 * Iw * 2.6)
    lt5 = c2 * zg
    lt6 = Iz / Iw
    lt7 = lt5**2 * lt6
    lt8 = lt5 * np.sqrt(lt6)
    lt9 = np.sqrt(k**2 + lt4 + lt7)
    lt10 = np.sqrt(lt9 - lt8)
    lt11 = np.sqrt(c1) * lt10
    up = lt3
    down = lt11
    lalt = up / down
    epsilon = 1
    alphalt = 0.21
    laltb = lalt / (93.9 * epsilon)
    philt = 0.5 * (1 + alphalt * (laltb - 0.2) + laltb**2)
    xlt = 1 / (philt + np.sqrt(philt**2 - laltb**2))
    # Resistance check (Mysdn and Mzscorr expected in daN·cm)
    r = (
        Nsd / (xiz * A * fy / 1.1)
        + (klt * Mysdn * 100) / (xlt * wply * fy / 1.1)
        + (kz * Mzscorr * 100) / (wplz * fy / 1.1)
        )
    if r < 1:
        print("Résistance au déversement est valide")
    return ult, klt, L1, Lfz, zg, k, c1, c2, lalt, laltb, alphalt, philt, xlt, r

def find_epf(epf1, epf10, s):
    """
    Interpolates epf value depending on s.    
    Parameters
    ----------
    epf1 : float
        Value of epf at s = 1
    epf10 : float
        Value of epf at s = 10
    s : float
        Input parameter
        Returns
    -------
    float
        Interpolated epf value
    """
    if s == 0:
        return 0
    elif s <= 1:
        return epf1
    elif 1 < s < 10:
        return epf1 + (epf10 - epf1) * np.log10(s)
    else:  # s >= 10
        return epf10

def moment_shear_defection(lt, cas, q, L, I):
    """Maximum moment, shear force, and deflection of a beam.
    Parameters:
        lt (str): load type ('uniform' or 'double ponctuel')
        cas (str): case ('y' or 'z')
        q (float): load [MN/m or force unit consistent with formulas]
        L (float): beam length [m]
        I (float): moment of inertia [m^4]
    Returns:
        Mmax (float): maximum moment
        Vmax (float): maximum shear
        deltamax (float): maximum deflection [m]
    """
    # Steel Young's modulus
    E = 2 * 10**5  # MPa = N/mm² (be careful with unit consistency!)
    if lt == "uniform":
        if cas == "y":
            Mmax = q * L**2 / 8
            Vmax = q * L / 2
            deltamax = (5/384) * (q * L**4) / (E * I)
        elif cas == "z":
            Mmax = q * L**2 / 32
            Vmax = 0.625 * q * L / 2
            deltamax = 0.0
    elif lt == "double ponctuel":
        a = L / 3
        if cas == "y":
            Mmax = q * a
            Vmax = q
            deltamax = q * a * (3*L**2 - 4*a**2) / (24 * E * I)
        elif cas == "z":
            Mmax = 0.0926 * q * L
            Vmax = 0.852 * q
            deltamax = 0.0
    else:
        raise ValueError("Unknown load type")
    return Mmax, Vmax, deltamax
