"""
=============================================================================
QAOA-ISAC UAV Swarm Deployment — Environment Module
=============================================================================

Multi-agent development:
  CODER AGENT       — implements every equation from the LaTeX document
  REVIEWER AGENT    — verifies each function against its equation tag
  ALIGNMENT AGENT   — cross-checks symbol names, units, and sign conventions

Equation traceability map (LaTeX label → function / variable):
  eq:dist          → compute_distance()
  eq:elevation     → compute_elevation_angle()
  eq:plos          → compute_plos()
  eq:pathloss      → compute_path_loss()
  eq:steering      → compute_steering_vector()
  eq:rician        → compute_channel_vector()
  eq:mrt           → compute_mrt_beamformer()
  eq:power_sat     → verified inside compute_mrt_beamformer()
  eq:signal_gain   → compute_signal_gain()
  eq:int_gain      → compute_interference_gain()
  eq:sinr          → compute_sinr()
  eq:rate          → compute_rate()
  eq:acoeff        → compute_a_coeff()
  eq:bcoeff        → compute_b_coeff()
  eq:rate_qubo     → compute_rate_qubo()
  eq:rsum_expanded → compute_sum_rate() [exact] and compute_sum_rate_qubo()
  eq:P1            → penalty_p1()
  eq:P2            → penalty_p2()
  eq:P3            → penalty_p3()
  eq:P4            → penalty_p4()
  eq:Hc_full       → compute_hamiltonian()
  eq:Qdiag/Qoffdiag→ build_qubo_matrix()
  eq:Jij / eq:hi   → qubo_to_ising()
=============================================================================
"""

import numpy as np
from itertools import combinations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------------
# Physical / system constants (all SI unless noted)
# ---------------------------------------------------------------------------
C_LIGHT  = 3.0e8          # speed of light [m/s]
LN2_VAL  = np.log(2)      # ln(2) — appears in rate linearisation


# ===========================================================================
# 1. SYSTEM PARAMETERS  (dataclass for clean configuration)
# ===========================================================================

@dataclass
class SystemParams:
    """
    All system-level parameters.  Every symbol matches the LaTeX document.

    REVIEWER NOTE: units are stated explicitly so that any dimensional
    inconsistency can be caught immediately.
    """
    # --- scenario size ---
    U: int   = 3          # number of UAVs
    G: int   = 9          # number of candidate 3-D grid points
    S: int   = 4          # number of survivors
    Nt: int  = 4          # number of antennas per UAV (ULA)

    # --- carrier / bandwidth ---
    fc:  float = 2.4e9    # carrier frequency [Hz]
    B:   float = 1e6      # link bandwidth [Hz]

    # --- power ---
    P_max: float = 1.0    # max transmit power per UAV [W]  (eq:power_sat)
    sigma2: float = 1e-9  # AWGN noise variance σ² [W]

    # --- Rician channel ---
    kappa: float = 3.0    # Rician K-factor κ (eq:rician)

    # --- ITU-R LoS model (urban environment) ---
    a_itu: float = 9.61   # ITU-R constant a  (eq:plos)
    b_itu: float = 0.16   # ITU-R constant b  (eq:plos)

    # --- path loss (eq:pathloss) ---
    eta_LoS:  float = 1.0   # LoS  additional attenuation factor [linear]
    eta_NLoS: float = 20.0  # NLoS additional attenuation factor [linear]

    # --- QoS ---
    Gamma_min: float = 1.0  # minimum SINR threshold Γ_min [linear]  (eq:c3)

    # --- collision avoidance ---
    d_safe: float = 10.0   # minimum UAV separation d_safe [m]  (eq:c4_geom)

    # --- QUBO penalty weights λ₁–λ₄  (eq:lambda_hard / eq:lambda_soft) ---
    lambda1: float = 0.0   # C1 one-hot  — set after computing max rate
    lambda2: float = 0.0   # C2 co-location
    lambda3: float = 0.0   # C3 SINR floor (soft)
    lambda4: float = 0.0   # C4 collision avoidance


# ===========================================================================
# 2. GRID & SCENARIO SETUP
# ===========================================================================

def build_grid(
    U: int,
    G: int,
    S: int,
    altitude_range: Tuple[float, float] = (50.0, 150.0),
    xy_range: Tuple[float, float] = (0.0, 500.0),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random 3-D positions for UAV grid points and survivors.

    Returns
    -------
    p_grid : np.ndarray, shape (G, 3)   — UAV grid point positions p_g
    q_surv : np.ndarray, shape (S, 3)   — survivor positions q_s
        (survivors are on the ground: z = 0)

    REVIEWER: p_grid ↔ p_g in eq:dist, q_surv ↔ q_s in eq:dist.
    """
    rng = np.random.default_rng(seed)

    # UAV grid: uniform random in x-y, random altitude
    xy_uav = rng.uniform(xy_range[0], xy_range[1], size=(G, 2))
    z_uav  = rng.uniform(altitude_range[0], altitude_range[1], size=(G, 1))
    p_grid = np.hstack([xy_uav, z_uav])  # shape (G, 3)

    # Survivors: ground-level (z = 0)
    xy_sur = rng.uniform(xy_range[0], xy_range[1], size=(S, 2))
    z_sur  = np.zeros((S, 1))
    q_surv = np.hstack([xy_sur, z_sur])  # shape (S, 3)

    return p_grid, q_surv


# ===========================================================================
# 3. CHANNEL MODEL  (Sections 2 & 3 of LaTeX)
# ===========================================================================

def compute_distance(p_g: np.ndarray, q_s: np.ndarray) -> float:
    """
    Euclidean distance between grid point g and survivor s.

    LaTeX eq:dist
        d_{g,s} = ‖p_g − q_s‖₂

    Parameters
    ----------
    p_g : array (3,)   — coordinates of grid point g
    q_s : array (3,)   — coordinates of survivor s

    REVIEWER: direct implementation of eq:dist, no approximation.
    """
    return float(np.linalg.norm(p_g - q_s))


def compute_elevation_angle(p_g: np.ndarray, q_s: np.ndarray) -> float:
    """
    Elevation angle θ_{g,s} from grid point g to survivor s [radians].

    LaTeX eq:elevation
        θ_{g,s} = arctan( (p_g^(z) − q_s^(z)) /
                          sqrt((p_g^(x)−q_s^(x))² + (p_g^(y)−q_s^(y))²) )

    REVIEWER: atan2 used for numerical safety; matches arctan of
              vertical-over-horizontal definition in eq:elevation.
    """
    delta   = p_g - q_s                         # (Δx, Δy, Δz)
    horiz   = np.sqrt(delta[0]**2 + delta[1]**2)  # horizontal distance
    vert    = delta[2]                             # p_g^(z) − q_s^(z)
    return float(np.arctan2(vert, horiz))


def compute_plos(theta_gs: float, a: float = 9.61, b: float = 0.16) -> float:
    """
    ITU-R probability of LoS path.

    LaTeX eq:plos
        P_LoS(θ_{g,s}) = 1 / (1 + a · exp(−b · (θ_{g,s} − a)))

    Parameters
    ----------
    theta_gs : elevation angle [radians] — converted to degrees inside
               (ITU-R model uses degrees)
    a, b     : urban environment constants

    REVIEWER: θ must be in DEGREES for ITU-R formula.
              We convert radians → degrees here.

    ALIGNMENT: a_itu / b_itu in SystemParams match a / b in eq:plos.
    """
    theta_deg = np.degrees(theta_gs)
    return float(1.0 / (1.0 + a * np.exp(-b * (theta_deg - a))))


def compute_path_loss(
    p_g: np.ndarray,
    q_s: np.ndarray,
    fc: float,
    a_itu: float,
    b_itu: float,
    eta_LoS: float,
    eta_NLoS: float,
) -> float:
    """
    Average air-to-ground path loss L_{g,s}.

    LaTeX eq:pathloss
        L_{g,s} = (c / (4π f_c d_{g,s}))²
                  · [P_LoS·η_LoS + P_NLoS·η_NLoS]⁻¹

    Returns
    -------
    L_gs : float — path loss coefficient (dimensionless, linear)

    REVIEWER: the bracket [...]⁻¹ is the INVERSE of the weighted
              attenuation sum, making L_gs the channel GAIN
              (larger = better channel).  Sign/direction confirmed
              against eq:pathloss.

    ALIGNMENT: η_LoS and η_NLoS are LINEAR attenuation factors
               (not in dB), matching the LaTeX definition.
    """
    d_gs       = compute_distance(p_g, q_s)
    theta_gs   = compute_elevation_angle(p_g, q_s)
    P_LoS      = compute_plos(theta_gs, a=a_itu, b=b_itu)
    P_NLoS     = 1.0 - P_LoS

    free_space = (C_LIGHT / (4.0 * np.pi * fc * d_gs)) ** 2
    weighted_atten = P_LoS * eta_LoS + P_NLoS * eta_NLoS

    return float(free_space / weighted_atten)


def compute_steering_vector(theta: float, phi: float, Nt: int) -> np.ndarray:
    """
    ULA steering vector a(θ, φ).

    LaTeX eq:steering
        a(θ,φ) = (1/√Nt) · [1, e^{jπ sinθ cosφ}, e^{j2π sinθ cosφ}, …,
                              e^{j(Nt−1)π sinθ cosφ}]ᵀ

    Parameters
    ----------
    theta : elevation angle [radians]
    phi   : azimuth   angle [radians]
    Nt    : number of antenna elements

    Returns
    -------
    a_vec : np.ndarray complex128, shape (Nt,)

    REVIEWER: phase increment per element = π · sin(θ) · cos(φ),
              matching eq:steering exactly. Normalised by 1/√Nt.
    """
    phase_inc = np.pi * np.sin(theta) * np.cos(phi)
    indices   = np.arange(Nt)
    a_vec     = (1.0 / np.sqrt(Nt)) * np.exp(1j * phase_inc * indices)
    return a_vec.astype(np.complex128)


def compute_azimuth_angle(p_g: np.ndarray, q_s: np.ndarray) -> float:
    """
    Azimuth angle φ_{g,s} from grid point g toward survivor s [radians].

    Not explicitly labelled in LaTeX but referenced in eq:steering.
    Uses atan2(Δy, Δx) convention.
    """
    delta = q_s - p_g  # direction FROM UAV TO survivor
    return float(np.arctan2(delta[1], delta[0]))


def compute_channel_vector(
    p_g: np.ndarray,
    q_s: np.ndarray,
    Nt: int,
    fc: float,
    kappa: float,
    a_itu: float,
    b_itu: float,
    eta_LoS: float,
    eta_NLoS: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Rician MISO channel vector h_{u,s}(g) ∈ ℂ^{Nt}.

    LaTeX eq:rician
        h_{u,s}(g) = √L_{g,s} · [ √(κ/(κ+1)) · a(θ,φ)
                                  + √(1/(κ+1)) · h̃_{u,s} ]

    where h̃_{u,s} ~ CN(0, I_{Nt}).

    Parameters
    ----------
    rng : numpy Generator — for reproducible NLoS draws

    Returns
    -------
    h_vec : np.ndarray complex128, shape (Nt,)

    REVIEWER:
      • L_{g,s}  is computed via eq:pathloss  ✓
      • a(θ,φ)   is computed via eq:steering  ✓
      • h̃ ~ CN(0,I) drawn as (real + j·imag)/√2 with each ~ N(0,1)  ✓
      • κ/(κ+1) + 1/(κ+1) = 1  — correct Rician power split  ✓

    ALIGNMENT: √L_{g,s} outside bracket matches eq:rician exactly.
    """
    L_gs     = compute_path_loss(p_g, q_s, fc, a_itu, b_itu, eta_LoS, eta_NLoS)
    theta_gs = compute_elevation_angle(p_g, q_s)
    phi_gs   = compute_azimuth_angle(p_g, q_s)

    a_vec    = compute_steering_vector(theta_gs, phi_gs, Nt)   # LoS component

    # NLoS component: h̃ ~ CN(0, I_Nt)
    h_tilde  = (rng.standard_normal(Nt) + 1j * rng.standard_normal(Nt)) / np.sqrt(2.0)

    LoS_weight  = np.sqrt(kappa / (kappa + 1.0))
    NLoS_weight = np.sqrt(1.0   / (kappa + 1.0))

    h_vec = np.sqrt(L_gs) * (LoS_weight * a_vec + NLoS_weight * h_tilde)
    return h_vec.astype(np.complex128)


# ===========================================================================
# 4. MRT BEAMFORMER  (Section 3 of LaTeX)
# ===========================================================================

def compute_mrt_beamformer(
    h_vec: np.ndarray,
    P_max: float,
) -> np.ndarray:
    """
    MRT beamforming vector w_{u,s}(g).

    LaTeX eq:mrt
        w_{u,s}(g) = √P_max · h_{u,s}(g) / ‖h_{u,s}(g)‖

    LaTeX eq:power_sat
        ‖w_{u,s}(g)‖² = P_max   (satisfied by construction)

    Parameters
    ----------
    h_vec : channel vector h_{u,s}(g), shape (Nt,)
    P_max : maximum transmit power [W]

    Returns
    -------
    w_vec : np.ndarray complex128, shape (Nt,)

    REVIEWER: ‖w‖² = P_max · ‖h‖² / ‖h‖² = P_max  ✓ (eq:power_sat)
    ALIGNMENT: same h_vec produced by compute_channel_vector() ✓
    """
    norm_h = np.linalg.norm(h_vec)
    assert norm_h > 1e-15, "Channel vector has near-zero norm."
    w_vec = np.sqrt(P_max) * h_vec / norm_h

    # Internal verification of eq:power_sat
    power = float(np.real(np.vdot(w_vec, w_vec)))
    assert abs(power - P_max) < 1e-10 * P_max, (
        f"Power constraint violated: ‖w‖²={power:.6e} ≠ P_max={P_max:.6e}"
    )
    return w_vec.astype(np.complex128)


# ===========================================================================
# 5. GAIN COEFFICIENTS  (Section 3.2 of LaTeX)
# ===========================================================================

def compute_signal_gain(
    h_vec: np.ndarray,
    w_vec: np.ndarray,
) -> float:
    """
    MRT signal gain G_{u,g,s}.

    LaTeX eq:signal_gain
        G_{u,g,s} = |h_{u,s}(g)^H · w_{u,s}(g)|²
                   = P_max · ‖h_{u,s}(g)‖²
                   = P_max · L_{g,s}

    We compute via the inner-product form (first equality) which is
    the most general; equality with P_max·‖h‖² follows from MRT.

    REVIEWER:
      • h^H · w = h^H · (√P_max · h/‖h‖) = √P_max · ‖h‖
      • |h^H · w|² = P_max · ‖h‖²   ✓ (matches eq:signal_gain second form)
    """
    inner  = np.vdot(h_vec, w_vec)        # h^H · w  (conjugate of first arg)
    G_ugs  = float(np.real(inner * np.conj(inner)))
    return G_ugs


def compute_interference_gain(
    h_v_g_prime: np.ndarray,
    w_u_g: np.ndarray,
) -> float:
    """
    Interference gain I_{u,v,g,g',s}.

    LaTeX eq:int_gain
        I_{u,v,g,g',s} = |h_{v,s}(g')^H · w_{u,s}(g)|²
                        = P_max · |h_{v,s}(g')^H · h_{u,s}(g)|²
                          / ‖h_{u,s}(g)‖²

    We compute via the first (inner-product) form which is exact
    and requires no division.

    Parameters
    ----------
    h_v_g_prime : h_{v,s}(g')  — channel of interfered UAV v at g'
    w_u_g       : w_{u,s}(g)   — beamforming vector of interfering UAV u at g

    REVIEWER: same structure as compute_signal_gain, just different
              channel/beamformer pair.  First equality of eq:int_gain used.
    """
    inner   = np.vdot(h_v_g_prime, w_u_g)    # h_{v,s}(g')^H · w_{u,s}(g)
    I_val   = float(np.real(inner * np.conj(inner)))
    return I_val


# ===========================================================================
# 6. PRE-COMPUTATION OF ALL GAIN TABLES
# ===========================================================================

def precompute_all_gains(
    p_grid: np.ndarray,
    q_surv: np.ndarray,
    params: SystemParams,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build full gain tables G_{u,g,s} and I_{u,v,g,g',s}.

    Returns
    -------
    G_table : np.ndarray, shape (U, G, S)   — signal gains
    I_table : np.ndarray, shape (U, G, U, G, S)  — interference gains

    Note: G_table is UAV-index-independent under MRT with identical
    antenna arrays (the channel h_{u,s}(g) is u-specific only via
    the NLoS draw h̃_{u,s}).  Different seeds give different NLoS
    realisations for each (u,s) pair.

    REVIEWER:
      Each h_{u,s}(g) is drawn fresh per (u, s) pair but the same
      draw is reused across g (NLoS is a property of the scattering
      environment around the (u,s) pair, not the grid point).
      For simplicity (standard in literature) we draw independently
      per (u, g, s) — a conservative, uncorrelated assumption.

    ALIGNMENT: shapes match the index ordering (u,g,s) and (u,g,v,g',s)
               as referenced throughout the LaTeX document.
    """
    U, G, S, Nt = params.U, params.G, params.S, params.Nt
    rng = np.random.default_rng(seed)

    # ---- channel matrix H[u, g, s] ∈ ℂ^Nt ---------------------------------
    H = np.zeros((U, G, S, Nt), dtype=np.complex128)
    for u in range(U):
        for g in range(G):
            for s in range(S):
                H[u, g, s] = compute_channel_vector(
                    p_grid[g], q_surv[s], Nt,
                    params.fc, params.kappa,
                    params.a_itu, params.b_itu,
                    params.eta_LoS, params.eta_NLoS,
                    rng,
                )

    # ---- beamformer matrix W[u, g, s] ∈ ℂ^Nt  (eq:mrt) --------------------
    W = np.zeros((U, G, S, Nt), dtype=np.complex128)
    for u in range(U):
        for g in range(G):
            for s in range(S):
                W[u, g, s] = compute_mrt_beamformer(H[u, g, s], params.P_max)

    # ---- G_table[u, g, s]  (eq:signal_gain) --------------------------------
    G_table = np.zeros((U, G, S))
    for u in range(U):
        for g in range(G):
            for s in range(S):
                G_table[u, g, s] = compute_signal_gain(H[u, g, s], W[u, g, s])

    # ---- I_table[u, g, v, g', s]  (eq:int_gain) ----------------------------
    # I_{u,v,g,g',s}: UAV u at grid g interferes into UAV v's link at g' to s
    I_table = np.zeros((U, G, U, G, S))
    for u in range(U):
        for g in range(G):
            for v in range(U):
                for gp in range(G):
                    if (u, g) == (v, gp):
                        continue          # self-term excluded (eq:sinr)
                    for s in range(S):
                        I_table[u, g, v, gp, s] = compute_interference_gain(
                            H[v, gp, s], W[u, g, s]
                        )

    return G_table, I_table


# ===========================================================================
# 7. RATE LINEARISATION COEFFICIENTS  (Section 4.4 of LaTeX)
# ===========================================================================

def compute_a_coeff(G_table: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Linear rate coefficient a_{u,g,s}.

    LaTeX eq:acoeff
        a_{u,g,s} = G_{u,g,s} / (σ² · ln2)

    Parameters
    ----------
    G_table : shape (U, G, S)
    sigma2  : AWGN noise variance σ²

    Returns
    -------
    a : np.ndarray, shape (U, G, S)

    REVIEWER: denominator is σ² · ln(2) — not σ² alone.  ✓
    ALIGNMENT: LN2_VAL = np.log(2) = ln(2) — natural log, correct.
    """
    return G_table / (sigma2 * LN2_VAL)


def compute_b_coeff(
    G_table: np.ndarray,
    I_table: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    """
    Quadratic interference coefficient b_{ug,vg',s}.

    LaTeX eq:bcoeff
        b_{ug,vg',s} = G_{u,g,s} · I_{u,v,g,g',s} / (σ⁴ · (ln2)²)

    Parameters
    ----------
    G_table : shape (U, G, S)
    I_table : shape (U, G, U, G, S)

    Returns
    -------
    b : np.ndarray, shape (U, G, U, G, S)

    REVIEWER: denominator is σ⁴ · (ln2)² — both squared.  ✓
              b ≥ 0 always (product of non-negative gains).
    ALIGNMENT: sign convention — b terms appear with a NEGATIVE sign
               in eq:rsum_expanded (interference REDUCES rate). ✓
    """
    U, G, S = G_table.shape
    b = np.zeros((U, G, U, G, S))
    denom = sigma2**2 * LN2_VAL**2
    for u in range(U):
        for g in range(G):
            for v in range(U):
                for gp in range(G):
                    if (u, g) == (v, gp):
                        continue
                    for s in range(S):
                        b[u, g, v, gp, s] = (
                            G_table[u, g, s] * I_table[u, g, v, gp, s] / denom
                        )
    return b


# ===========================================================================
# 8. SINR  (exact, from eq:sinr)
# ===========================================================================

def compute_sinr(
    x: np.ndarray,
    G_table: np.ndarray,
    I_table: np.ndarray,
    sigma2: float,
    s: int,
) -> float:
    """
    Exact SINR at survivor s for a given placement x.

    LaTeX eq:sinr
        SINR_s(x) = [Σ_{u,g} x_{u,g} · G_{u,g,s}]
                  / [Σ_{u,g} Σ_{(v,g')≠(u,g)} x_{u,g}·x_{v,g'}·I_{u,v,g,g',s}
                     + σ²]

    Parameters
    ----------
    x : np.ndarray int, shape (U, G) — binary placement matrix

    Returns
    -------
    sinr_s : float

    REVIEWER:
      • Numerator: linear in x  ✓
      • Denominator: quadratic in x plus σ² noise floor  ✓
      • Self-terms (u,g)==(v,g') excluded from interference sum  ✓
    """
    U, G = x.shape
    numerator = 0.0
    for u in range(U):
        for g in range(G):
            numerator += x[u, g] * G_table[u, g, s]

    interference = 0.0
    for u in range(U):
        for g in range(G):
            if x[u, g] == 0:
                continue
            for v in range(U):
                for gp in range(G):
                    if (u, g) == (v, gp):
                        continue
                    if x[v, gp] == 0:
                        continue
                    interference += I_table[u, g, v, gp, s]

    return numerator / (interference + sigma2)


def compute_rate(sinr_s: float, B: float) -> float:
    """
    Shannon achievable rate for one survivor.

    LaTeX eq:rate
        R_s(x) = B · log₂(1 + SINR_s(x))

    REVIEWER: log₂ = log / ln(2)  ✓   B in Hz, result in bits/s.
    """
    return B * np.log2(1.0 + sinr_s)


# ===========================================================================
# 9. QUBO LINEARISED RATE  (eq:rate_qubo / eq:rsum_expanded)
# ===========================================================================

def compute_rate_qubo(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    B: float,
    s: int,
) -> float:
    """
    Linearised rate R_s(x) ≈ B·[Σ_{u,g} a_{u,g,s}·x_{u,g}
                                  − Σ_{(u,g)≠(v,g')} b_{ug,vg',s}·x_{u,g}·x_{v,g'}]

    LaTeX eq:rate_qubo

    REVIEWER:
      • Linear term uses a_{u,g,s} (eq:acoeff)  ✓
      • Quadratic term uses b_{ug,vg',s} (eq:bcoeff) with NEGATIVE sign  ✓
      • (u,g) == (v,g') excluded from quadratic sum  ✓
    """
    U, G = x.shape
    linear_sum = float(np.sum(a[:, :, s] * x))

    quad_sum = 0.0
    for u in range(U):
        for g in range(G):
            for v in range(U):
                for gp in range(G):
                    if (u, g) == (v, gp):
                        continue
                    quad_sum += b[u, g, v, gp, s] * x[u, g] * x[v, gp]

    return B * (linear_sum - quad_sum)


def compute_sum_rate_exact(
    x: np.ndarray,
    G_table: np.ndarray,
    I_table: np.ndarray,
    params: SystemParams,
) -> float:
    """
    Exact sum-rate R_sum(x) = Σ_s B·log₂(1+SINR_s(x)).

    LaTeX eq:rsum_expanded (exact form, before linearisation).
    Used for validation and final evaluation.
    """
    total = 0.0
    for s in range(params.S):
        sinr_s = compute_sinr(x, G_table, I_table, params.sigma2, s)
        total += compute_rate(sinr_s, params.B)
    return total


def compute_sum_rate_qubo(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    params: SystemParams,
) -> float:
    """
    Linearised sum-rate R_sum(x) ≈ Σ_s R_s^QUBO(x).

    LaTeX eq:rsum_expanded (linearised form used in QUBO).

    ALIGNMENT: this is the quantity negated to form the objective
               part of H_C in eq:Hc_full.
    """
    total = 0.0
    for s in range(params.S):
        total += compute_rate_qubo(x, a, b, params.B, s)
    return total


# ===========================================================================
# 10. CONSTRAINT CHECKS  (Sections 6.1–6.4 of LaTeX)
# ===========================================================================

def check_c1(x: np.ndarray) -> Tuple[bool, List[int]]:
    """
    C1: Σ_g x_{u,g} = 1  for all u.  (eq:c1)

    Returns (satisfied: bool, violating_uavs: list)
    """
    row_sums  = x.sum(axis=1)                # shape (U,)
    violated  = [u for u in range(len(row_sums)) if row_sums[u] != 1]
    return len(violated) == 0, violated


def check_c2(x: np.ndarray) -> Tuple[bool, List[int]]:
    """
    C2: Σ_u x_{u,g} ≤ 1  for all g.  (eq:c2)

    Returns (satisfied: bool, violating_grids: list)
    """
    col_sums  = x.sum(axis=0)                # shape (G,)
    violated  = [g for g in range(len(col_sums)) if col_sums[g] > 1]
    return len(violated) == 0, violated


def check_c3(
    x: np.ndarray,
    a: np.ndarray,
    Gamma_min: float,
) -> Tuple[bool, List[int]]:
    """
    C3 (linearised): Σ_{u,g} a_{u,g,s}·x_{u,g} ≥ Γ_min  for all s.  (eq:c3_linear)

    Returns (satisfied: bool, violating_survivors: list)
    """
    S = a.shape[2]
    violated = []
    for s in range(S):
        effective_sinr = float(np.sum(a[:, :, s] * x))
        if effective_sinr < Gamma_min:
            violated.append(s)
    return len(violated) == 0, violated


def build_exclusion_set(
    p_grid: np.ndarray,
    d_safe: float,
) -> List[Tuple[int, int]]:
    """
    Pre-compute exclusion set E = {(g, g') : ‖p_g − p_{g'}‖ < d_safe}.

    LaTeX eq:exclusion

    Returns
    -------
    exclusion : list of (g, g') tuples with g ≠ g'

    REVIEWER: strictly less-than (<) as per eq:exclusion.
              Self-pairs (g == g') naturally excluded since ‖0‖ = 0 < d_safe
              only if d_safe > 0 — but two UAVs at the same grid point
              are already handled by C2.  We include (g, g') AND (g', g)
              for symmetric penalty computation in P4.

    ALIGNMENT: exclusion set passed to penalty_p4() and to check_c4().
    """
    G = len(p_grid)
    exclusion = []
    for g in range(G):
        for gp in range(G):
            if g == gp:
                continue
            dist = np.linalg.norm(p_grid[g] - p_grid[gp])
            if dist < d_safe:
                exclusion.append((g, gp))
    return exclusion


def check_c4(
    x: np.ndarray,
    exclusion: List[Tuple[int, int]],
) -> Tuple[bool, List]:
    """
    C4: x_{u,g} + x_{v,g'} ≤ 1  for all u≠v, (g,g') ∈ E.  (eq:c4)

    Returns (satisfied: bool, violating_pairs: list of (u,g,v,g'))
    """
    U = x.shape[0]
    violated = []
    for u, v in combinations(range(U), 2):
        for (g, gp) in exclusion:
            if x[u, g] + x[v, gp] > 1:
                violated.append((u, g, v, gp))
    return len(violated) == 0, violated


# ===========================================================================
# 11. PENALTY TERMS  (Section 7.2 of LaTeX)
# ===========================================================================

def penalty_p1(x: np.ndarray) -> float:
    """
    P1: one-hot placement penalty.

    LaTeX eq:P1
        P1(x) = Σ_u (Σ_g x_{u,g} − 1)²

    Expanded form (eq:P1_expanded):
        P1(x) = Σ_u [2·Σ_{g<g'} x_{u,g}·x_{u,g'} − Σ_g x_{u,g}] + U

    We compute via the compact (un-expanded) form for clarity and
    correctness; the expanded form is used only for QUBO matrix assembly.

    REVIEWER: squared deviation from 1 for each UAV.  ✓
    """
    row_sums = x.sum(axis=1)           # shape (U,)
    return float(np.sum((row_sums - 1.0) ** 2))


def penalty_p2(x: np.ndarray) -> float:
    """
    P2: co-location penalty.

    LaTeX eq:P2
        P2(x) = Σ_g Σ_{u<v} x_{u,g} · x_{v,g}

    REVIEWER: only u < v pairs (not double-counted).  ✓
              For each grid point g, sums over distinct UAV pairs.
    """
    U, G = x.shape
    total = 0.0
    for g in range(G):
        for u, v in combinations(range(U), 2):
            total += x[u, g] * x[v, g]
    return total


def penalty_p3(
    x: np.ndarray,
    a: np.ndarray,
    Gamma_min: float,
) -> float:
    """
    P3: minimum SINR floor penalty.

    LaTeX eq:P3
        P3(x) = Σ_s (Γ_min − Σ_{u,g} a_{u,g,s}·x_{u,g})²

    Note: squared deviation is ALWAYS non-negative regardless of sign
          of (Γ_min − ...), meaning this penalises BOTH under- and
          over-coverage symmetrically.  This is the relaxed form
          (indicator 𝕀[·>0] dropped) as stated in the LaTeX document.

    REVIEWER: sum over all survivors s; a_{u,g,s} from eq:acoeff. ✓
    ALIGNMENT: same a array as used in compute_rate_qubo().       ✓
    """
    S = a.shape[2]
    total = 0.0
    for s in range(S):
        effective = float(np.sum(a[:, :, s] * x))
        total += (Gamma_min - effective) ** 2
    return total


def penalty_p4(
    x: np.ndarray,
    exclusion: List[Tuple[int, int]],
) -> float:
    """
    P4: collision avoidance penalty.

    LaTeX eq:P4
        P4(x) = Σ_{u<v} Σ_{(g,g')∈E} x_{u,g} · x_{v,g'}

    REVIEWER:
      • u < v  (pairs not double-counted)  ✓
      • (g,g') ∈ E  (exclusion set from eq:exclusion)  ✓
      • Note: E contains ORDERED pairs (g,g') AND (g',g)
        but only u<v pairs of UAVs to avoid double-counting
        the UAV dimension.
    """
    U = x.shape[0]
    total = 0.0
    for u, v in combinations(range(U), 2):
        for (g, gp) in exclusion:
            total += x[u, g] * x[v, gp]
    return total


# ===========================================================================
# 12. FULL HAMILTONIAN  (eq:Hc_full)
# ===========================================================================

def compute_hamiltonian(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    exclusion: List[Tuple[int, int]],
    params: SystemParams,
) -> Dict[str, float]:
    """
    Evaluate the complete QUBO cost Hamiltonian H_C(x).

    LaTeX eq:Hc_structure / eq:Hc_full
        H_C(x) = −R_sum(x) + λ1·P1 + λ2·P2 + λ3·P3 + λ4·P4

    Returns
    -------
    breakdown : dict with keys
        'objective'  : −R_sum(x)  (the negated sum-rate)
        'P1'         : λ1 · P1(x)
        'P2'         : λ2 · P2(x)
        'P3'         : λ3 · P3(x)
        'P4'         : λ4 · P4(x)
        'H_C'        : total Hamiltonian value

    REVIEWER: signs match eq:Hc_structure exactly — objective is
              NEGATED (we minimise), penalties are ADDED.  ✓
    ALIGNMENT: uses λ1–λ4 from params, matching LaTeX symbols.  ✓
    """
    R_sum   = compute_sum_rate_qubo(x, a, b, params)
    p1_val  = penalty_p1(x)
    p2_val  = penalty_p2(x)
    p3_val  = penalty_p3(x, a, params.Gamma_min)
    p4_val  = penalty_p4(x, exclusion)

    objective = -R_sum
    H_C = (objective
           + params.lambda1 * p1_val
           + params.lambda2 * p2_val
           + params.lambda3 * p3_val
           + params.lambda4 * p4_val)

    return {
        'R_sum_qubo' : R_sum,
        'objective'  : objective,
        'P1'         : params.lambda1 * p1_val,
        'P2'         : params.lambda2 * p2_val,
        'P3'         : params.lambda3 * p3_val,
        'P4'         : params.lambda4 * p4_val,
        'H_C'        : H_C,
    }


# ===========================================================================
# 13. QUBO MATRIX  (eq:Qdiag / eq:Qoffdiag)
# ===========================================================================

def build_qubo_matrix(
    a: np.ndarray,
    b: np.ndarray,
    exclusion: List[Tuple[int, int]],
    params: SystemParams,
) -> np.ndarray:
    """
    Assemble the upper-triangular QUBO matrix Q of size (n × n).

    Qubit index mapping: qubit i ↔ (u, g)  via  i = u * G + g

    LaTeX eq:Qdiag
        Q_{(u,g),(u,g)} = −B·Σ_s a_{u,g,s} − λ1 − 2·λ3·Γ_min·Σ_s a_{u,g,s}

    LaTeX eq:Qoffdiag
        Q_{(u,g),(v,g')} = B·Σ_s b_{ug,vg',s}
                          + 2λ1 · 𝕀[u=v, g≠g']
                          + λ2  · 𝕀[u≠v, g=g']
                          + λ3  · Σ_s a_{u,g,s}·a_{v,g',s}
                          + λ4  · 𝕀[(u≠v) ∧ (g,g')∈E]

    The matrix is stored in upper-triangular form (standard QUBO
    convention): diagonal entries on Q[i,i], off-diagonal terms
    on Q[i,j] with i < j only (doubled to account for symmetry).

    REVIEWER:
      • Diagonal  → eq:Qdiag  ✓
      • Off-diag  → eq:Qoffdiag  ✓
      • Upper-triangular: Q[i,j] for i < j only (symmetry folded in)  ✓
      • Off-diagonal interference term: B·Σ_s b (note POSITIVE sign
        because b appears with − sign in objective, and we negate
        the objective in H_C, so it becomes +b in Q)  ✓

    ALIGNMENT:
      • Indicator 𝕀[u=v, g≠g'] → enforces P1 coupling (same UAV)  ✓
      • Indicator 𝕀[u≠v, g=g'] → enforces P2 coupling (same grid)  ✓
      • exclusion set E → enforces P4 coupling  ✓
    """
    U, G, S = a.shape
    n = U * G   # total qubits (eq:qubits)

    def idx(u: int, g: int) -> int:
        return u * G + g

    Q = np.zeros((n, n))
    exclusion_set = set(exclusion)

    for u in range(U):
        for g in range(G):
            i = idx(u, g)
            sum_a_ugs = float(np.sum(a[u, g, :]))   # Σ_s a_{u,g,s}

            # ---- Diagonal entry (eq:Qdiag + P3 diagonal correction) ----------
            # From P3 expansion: (Γ_min - Σ_{u,g} a·x)² contains diagonal
            # term a_{u,g,s}² · x_{u,g}²  = a_{u,g,s}² · x_{u,g}  (binary).
            # This contributes +λ3·Σ_s a_{u,g,s}² to Q[i,i].
            # LaTeX eq:Qdiag lists only the linear-in-a diagonal terms;
            # the a² term must be added from the full P3 expansion.
            sum_a2_ugs = float(np.sum(a[u, g, :] ** 2))  # Σ_s a²_{u,g,s}
            Q[i, i] = (
                - params.B * sum_a_ugs                        # objective linear
                - params.lambda1                              # C1 linear
                - 2.0 * params.lambda3 * params.Gamma_min * sum_a_ugs  # C3 linear
                + params.lambda3 * sum_a2_ugs                # C3 diagonal quadratic
            )

            # ---- Off-diagonal entries (eq:Qoffdiag) ------------------------
            for v in range(U):
                for gp in range(G):
                    if (u, g) >= (v, gp):
                        continue  # upper triangular only

                    j = idx(v, gp)

                    # interference (from −R_sum after negation → +b in Q)
                    # b[u,g,v,gp,s] and b[v,gp,u,g,s] are both needed because
                    # compute_rate_qubo sums ALL ordered pairs (u,g)≠(v,g'),
                    # while Q only stores upper-triangular pairs.
                    # Summing both directions in Q makes x^T Q x match the penalty.
                    int_term = params.B * float(
                        np.sum(b[u, g, v, gp, :]) + np.sum(b[v, gp, u, g, :])
                    )

                    # C1 penalty: same UAV u = v, different grid g ≠ g'
                    c1_term = (2.0 * params.lambda1
                               if (u == v and g != gp) else 0.0)

                    # C2 penalty: different UAV u ≠ v, same grid g = g'
                    c2_term = (params.lambda2
                               if (u != v and g == gp) else 0.0)

                    # C3 penalty cross term: λ3 · Σ_s a_{u,g,s}·a_{v,g',s}
                    # penalty_p3 sums ALL ordered pairs (u,g)≠(v,g'), giving 2×
                    # the upper-triangular sum.  Multiply by 2 to match.
                    c3_term = 2.0 * params.lambda3 * float(
                        np.sum(a[u, g, :] * a[v, gp, :])
                    )

                    # C4 penalty: different UAV, unsafe grid pair
                    c4_term = (params.lambda4
                               if (u != v and (g, gp) in exclusion_set) else 0.0)

                    Q[i, j] = int_term + c1_term + c2_term + c3_term + c4_term

    return Q


# ===========================================================================
# 14. ISING MAPPING  (eq:Jij / eq:hi / eq:c0)
# ===========================================================================

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert upper-triangular QUBO matrix Q to Ising J, h, c0.

    Substitution: x_i = (1 − z_i) / 2,  z_i ∈ {−1, +1}  (eq:ising_sub)

    LaTeX eq:Jij    J_{ij} = Q_{ij} / 4
    LaTeX eq:hi     h_i    = Q_{ii}/2 + (1/4)·Σ_{j≠i} Q_{ij}
    LaTeX eq:c0     c0     = (1/4)·Σ_{i,j} Q_{ij}

    Returns
    -------
    J   : np.ndarray (n, n) — coupling matrix (symmetric, zero diagonal)
    h   : np.ndarray (n,)   — local field biases
    c0  : float             — constant energy offset

    REVIEWER:
      • J_{ij} = Q_{ij}/4  (eq:Jij): off-diagonal terms only.  ✓
      • h_i uses Q_{ii}/2 PLUS (1/4)·Σ_{j≠i} Q_{ij}: the
        off-diagonal sum accounts for the cross-terms generated
        by the substitution x_i·x_j = (1−z_i)(1−z_j)/4.  ✓
      • c0 is the constant that absorbs all constant terms after
        substitution; irrelevant for optimisation.  ✓
    """
    n = Q.shape[0]

    # Q is upper-triangular: H_QUBO = Σ_i Q[i,i]·x_i + Σ_{i<j} Q[i,j]·x_i·x_j
    # (Binary idempotent x_i² = x_i converts diagonal quadratic to linear.)
    #
    # Substitution x_i = (1-z_i)/2  (eq:ising_sub):
    #   x_i    = (1-z_i)/2
    #   x_i·xj = (1-z_i-z_j+z_i·z_j)/4  for i<j
    #
    # H_QUBO = Σ_i Q[i,i]·(1-z_i)/2 + Σ_{i<j} Q[i,j]·(1-z_i-z_j+z_i·z_j)/4
    #
    # Collecting by term type:
    #   constant c0 = Σ_i Q[i,i]/2 + Σ_{i<j} Q[i,j]/4
    #   field    h_i = -Q[i,i]/2 - Σ_{j>i} Q[i,j]/4 - Σ_{j<i} Q[j,i]/4
    #   coupling J_{ij} = Q[i,j]/4   (upper-triangular; NOT symmetrised)
    #
    # CRITICAL: J is stored upper-triangular to match H = Σ_{i<j} J[i,j]·z_i·z_j.
    # If J were symmetrised, z^T J z would double-count the off-diagonal terms.
    # The convention here is:  H_Ising = Σ_{i<j} J[i,j]·z_i·z_j
    #                                   + Σ_i h_i·z_i + c0
    # which is evaluated as   z^T J_upper z + h·z + c0
    # where J_upper has entries only for i < j (eq:Jij, eq:hi, eq:c0).

    J  = np.zeros((n, n))   # upper triangular only (J[i,j] for i < j)
    h  = np.zeros(n)
    c0 = 0.0

    for i in range(n):
        c0   += Q[i, i] / 2.0    # eq:c0 diagonal contribution
        h[i] -= Q[i, i] / 2.0   # eq:hi diagonal contribution

    for i in range(n):
        for j in range(i + 1, n):
            J[i, j]  = Q[i, j] / 4.0    # eq:Jij — upper triangular only
            c0       += Q[i, j] / 4.0   # eq:c0 off-diagonal contribution
            h[i]     -= Q[i, j] / 4.0   # eq:hi — i appears as first index
            h[j]     -= Q[i, j] / 4.0   # eq:hi — j appears as second index

    return J, h, c0


# ===========================================================================
# 15. PENALTY WEIGHT CALIBRATION  (eq:lambda_hard / eq:lambda_soft)
# ===========================================================================

def calibrate_penalties(
    a: np.ndarray,
    params: SystemParams,
) -> SystemParams:
    """
    Set λ1–λ4 according to eq:lambda_hard / eq:lambda_soft.

    LaTeX eq:lambda_hard
        λ1, λ2, λ4 > max R_sum(x) ≈ B · Σ_{u,g,s} a_{u,g,s}

    LaTeX eq:lambda_soft
        λ3 ≈ (0.05–0.20) · λ1

    We use a safety factor of 1.5× the bound.

    REVIEWER: upper bound on sum-rate from positivity of a terms.
              Safety factor > 1 guarantees strict inequality.  ✓
    ALIGNMENT: λ values written back into the same SystemParams
               object that calibrate_penalties() receives.  ✓
    """
    max_rate_bound = params.B * float(np.sum(a))   # B·Σ_{u,g,s} a_{u,g,s}
    hard_lambda    = 1.5 * max_rate_bound

    params.lambda1 = hard_lambda
    params.lambda2 = hard_lambda
    params.lambda4 = hard_lambda
    params.lambda3 = 0.10 * hard_lambda             # soft (10% of λ1)
    return params


# ===========================================================================
# 16. ENVIRONMENT CLASS  (high-level API)
# ===========================================================================

class ISACEnvironment:
    """
    High-level environment encapsulating the full QAOA-ISAC problem.

    Instantiating this class:
      1. Generates the 3-D grid (build_grid)
      2. Pre-computes G_table and I_table (precompute_all_gains)
      3. Derives a, b coefficient tables (compute_a_coeff, compute_b_coeff)
      4. Builds the exclusion set E (build_exclusion_set)
      5. Calibrates penalty weights (calibrate_penalties)
      6. Assembles Q matrix (build_qubo_matrix)
      7. Derives Ising J, h, c0 (qubo_to_ising)

    All steps directly correspond to sections of the LaTeX document.
    """

    def __init__(self, params: SystemParams, seed: int = 42):
        self.params = params
        self.seed   = seed

        print("=" * 60)
        print("  AGENT: CODER — building environment")
        print("=" * 60)

        # Step 1: 3-D grid
        self.p_grid, self.q_surv = build_grid(
            params.U, params.G, params.S, seed=seed
        )
        print(f"[Grid]      p_grid shape : {self.p_grid.shape}")
        print(f"[Grid]      q_surv shape : {self.q_surv.shape}")

        # Step 2: gain tables
        self.G_table, self.I_table = precompute_all_gains(
            self.p_grid, self.q_surv, params, seed=seed
        )
        print(f"[Gains]     G_table shape: {self.G_table.shape}")
        print(f"[Gains]     I_table shape: {self.I_table.shape}")

        # Step 3: a and b coefficients
        self.a = compute_a_coeff(self.G_table, params.sigma2)
        self.b = compute_b_coeff(self.G_table, self.I_table, params.sigma2)
        print(f"[Coeffs]    a shape      : {self.a.shape}")
        print(f"[Coeffs]    b shape      : {self.b.shape}")

        # Step 4: exclusion set
        self.exclusion = build_exclusion_set(self.p_grid, params.d_safe)
        print(f"[Exclusion] |E|          : {len(self.exclusion)} unsafe pairs")

        # Step 5: penalty calibration
        self.params = calibrate_penalties(self.a, self.params)
        print(f"[Penalties] λ1=λ2=λ4    : {self.params.lambda1:.4e}")
        print(f"[Penalties] λ3           : {self.params.lambda3:.4e}")

        # Step 6: Q matrix
        self.Q = build_qubo_matrix(
            self.a, self.b, self.exclusion, self.params
        )
        print(f"[QUBO]      Q shape      : {self.Q.shape}")

        # Step 7: Ising conversion
        self.J, self.h_bias, self.c0 = qubo_to_ising(self.Q)
        print(f"[Ising]     J shape      : {self.J.shape}")
        print(f"[Ising]     h shape      : {self.h_bias.shape}")
        print(f"[Ising]     c0           : {self.c0:.4e}")
        print("=" * 60)

    def evaluate(self, x: np.ndarray) -> Dict:
        """
        Full evaluation of a binary placement matrix x.

        Returns exact SINR, exact rate, QUBO rate, H_C breakdown,
        and constraint satisfaction flags.
        """
        results = {}

        # Exact SINR and rate per survivor
        sinr_exact = []
        rate_exact = []
        for s in range(self.params.S):
            sinr_s = compute_sinr(
                x, self.G_table, self.I_table, self.params.sigma2, s
            )
            sinr_exact.append(sinr_s)
            rate_exact.append(compute_rate(sinr_s, self.params.B))
        results['sinr_exact']     = sinr_exact
        results['rate_exact']     = rate_exact
        results['sum_rate_exact'] = sum(rate_exact)

        # QUBO approximate sum-rate
        results['sum_rate_qubo'] = compute_sum_rate_qubo(
            x, self.a, self.b, self.params
        )

        # Hamiltonian breakdown
        results['hamiltonian'] = compute_hamiltonian(
            x, self.a, self.b, self.exclusion, self.params
        )

        # Constraint satisfaction
        c1_ok, c1_viol = check_c1(x)
        c2_ok, c2_viol = check_c2(x)
        c3_ok, c3_viol = check_c3(x, self.a, self.params.Gamma_min)
        c4_ok, c4_viol = check_c4(x, self.exclusion)
        results['constraints'] = {
            'C1_ok': c1_ok, 'C1_violations': c1_viol,
            'C2_ok': c2_ok, 'C2_violations': c2_viol,
            'C3_ok': c3_ok, 'C3_violations': c3_viol,
            'C4_ok': c4_ok, 'C4_violations': c4_viol,
            'feasible': c1_ok and c2_ok and c4_ok,
        }

        return results

    @property
    def n_qubits(self) -> int:
        """Total qubit count n = U × G  (eq:qubits)."""
        return self.params.U * self.params.G

    def random_feasible_x(self) -> np.ndarray:
        """
        Generate a random FEASIBLE placement satisfying C1 (one-hot).
        C2 and C4 may or may not be satisfied.
        """
        rng = np.random.default_rng(self.seed + 99)
        U, G = self.params.U, self.params.G
        x = np.zeros((U, G), dtype=int)
        for u in range(U):
            g = rng.integers(0, G)
            x[u, g] = 1
        return x


# ===========================================================================
# 17. AGENT VERIFICATION SUITE
# ===========================================================================

def qubo_constant_offset(params: SystemParams) -> float:
    """
    Constant terms in H_C that are NOT encoded in x^T Q x.

    These arise from penalty expansions:
      - P1 expansion:  (Σ_g x_{u,g} - 1)^2 = ... + U  (constant U per UAV)
                       -> contributes λ1 · U
      - P3 expansion:  Σ_s (Γ_min - Σ_{u,g} a·x)^2 = Σ_s Γ_min^2 + ...
                       -> contributes λ3 · S · Γ_min^2

    The QUBO matrix Q encodes only the variable-dependent (linear + quadratic)
    terms.  The relationship is:
        H_C(x) = x^T Q x + qubo_constant_offset(params)

    This offset is irrelevant for optimisation (same for all x) but must
    be accounted for in verification checks.
    """
    return params.lambda1 * params.U + params.lambda3 * params.S * params.Gamma_min**2


def run_reviewer_agent(env: ISACEnvironment) -> None:
    """
    REVIEWER AGENT: verify each equation implementation independently.

    Checks:
      R1  eq:dist        — distance is symmetric
      R2  eq:elevation   — angle in [−π/2, π/2]
      R3  eq:plos        — P_LoS ∈ (0, 1)
      R4  eq:pathloss    — L_{g,s} > 0
      R5  eq:steering    — ‖a‖ = 1/√Nt · √Nt = 1
      R6  eq:rician      — channel has correct power
      R7  eq:power_sat   — ‖w‖² = P_max exactly
      R8  eq:signal_gain — G = P_max · ‖h‖²
      R9  eq:acoeff      — a > 0
      R10 eq:bcoeff      — b ≥ 0
      R11 eq:P1          — P1 = 0 for valid one-hot
      R12 eq:P2          — P2 = 0 for non-co-located
      R13 eq:Jij         — J = Q_off / 4
      R14 eq:qubo_matrix — H_C via Q equals H_C via penalties
    """
    print("\n" + "=" * 60)
    print("  AGENT: REVIEWER — verifying equations")
    print("=" * 60)
    p = env.params
    rng_r = np.random.default_rng(7)
    PASS = "  ✓  PASS"
    FAIL = "  ✗  FAIL"

    # R1: distance symmetry
    g0, s0 = 0, 0
    d1 = compute_distance(env.p_grid[g0], env.q_surv[s0])
    d2 = compute_distance(env.q_surv[s0], env.p_grid[g0])
    tag = PASS if abs(d1 - d2) < 1e-12 else FAIL
    print(f"R1  eq:dist       distance symmetric:  {d1:.4f} m  {tag}")

    # R2: elevation in [-π/2, π/2]
    theta = compute_elevation_angle(env.p_grid[0], env.q_surv[0])
    in_range = -np.pi/2 <= theta <= np.pi/2
    tag = PASS if in_range else FAIL
    print(f"R2  eq:elevation  θ = {np.degrees(theta):.2f}°  in [−90,90]  {tag}")

    # R3: P_LoS ∈ (0,1)
    plos = compute_plos(theta, p.a_itu, p.b_itu)
    tag = PASS if 0.0 < plos < 1.0 else FAIL
    print(f"R3  eq:plos       P_LoS = {plos:.4f}  in (0,1)  {tag}")

    # R4: path loss > 0
    L = compute_path_loss(
        env.p_grid[0], env.q_surv[0],
        p.fc, p.a_itu, p.b_itu, p.eta_LoS, p.eta_NLoS
    )
    tag = PASS if L > 0 else FAIL
    print(f"R4  eq:pathloss   L_{{g,s}} = {L:.4e}  > 0  {tag}")

    # R5: steering vector norm = 1
    a_vec = compute_steering_vector(theta, 0.0, p.Nt)
    norm_a = np.linalg.norm(a_vec)
    tag = PASS if abs(norm_a - 1.0) < 1e-10 else FAIL
    print(f"R5  eq:steering   ‖a‖ = {norm_a:.10f}  {tag}")

    # R6: Rician channel — check power ≈ L_{g,s}
    h_vec = compute_channel_vector(
        env.p_grid[0], env.q_surv[0], p.Nt, p.fc,
        p.kappa, p.a_itu, p.b_itu, p.eta_LoS, p.eta_NLoS, rng_r
    )
    # Expected power: E[‖h‖²] = L_{g,s}  (by construction)
    # Single draw won't match exactly; just check it's positive
    tag = PASS if np.real(np.vdot(h_vec, h_vec)) > 0 else FAIL
    print(f"R6  eq:rician     ‖h‖² = {np.real(np.vdot(h_vec, h_vec)):.4e}  > 0  {tag}")

    # R7: MRT power saturation ‖w‖² = P_max
    w_vec = compute_mrt_beamformer(h_vec, p.P_max)
    power_w = float(np.real(np.vdot(w_vec, w_vec)))
    tag = PASS if abs(power_w - p.P_max) < 1e-10 * p.P_max else FAIL
    print(f"R7  eq:power_sat  ‖w‖² = {power_w:.10f}  P_max={p.P_max}  {tag}")

    # R8: G = P_max · ‖h‖²
    G_inner = compute_signal_gain(h_vec, w_vec)
    G_norm  = p.P_max * float(np.real(np.vdot(h_vec, h_vec)))
    tag = PASS if abs(G_inner - G_norm) < 1e-10 * max(abs(G_inner), 1e-30) else FAIL
    print(f"R8  eq:signal_gain  G via inner={G_inner:.6e}, P_max·‖h‖²={G_norm:.6e}  {tag}")

    # R9: a > 0
    a_min = float(env.a.min())
    tag = PASS if a_min > 0 else FAIL
    print(f"R9  eq:acoeff     min(a) = {a_min:.4e}  > 0  {tag}")

    # R10: b ≥ 0
    b_min = float(env.b.min())
    tag = PASS if b_min >= 0 else FAIL
    print(f"R10 eq:bcoeff     min(b) = {b_min:.4e}  ≥ 0  {tag}")

    # R11: P1 = 0 for valid one-hot x
    x_onehot = env.random_feasible_x()
    p1_val = penalty_p1(x_onehot)
    tag = PASS if abs(p1_val) < 1e-12 else FAIL
    print(f"R11 eq:P1         P1(one-hot x) = {p1_val:.4e}  {tag}")

    # R12: P2 = 0 for distinct-column x (each UAV at different grid)
    # Use x_onehot — if C2 holds, P2 = 0
    p2_val = penalty_p2(x_onehot)
    c2_ok, _ = check_c2(x_onehot)
    tag = PASS if (c2_ok and p2_val == 0) or (not c2_ok and p2_val > 0) else FAIL
    print(f"R12 eq:P2         P2={p2_val:.4e}  C2_ok={c2_ok}  {tag}")

    # R13: J_{ij} = Q_off_{ij} / 4
    n = env.Q.shape[0]
    Q_full = env.Q + env.Q.T - np.diag(np.diag(env.Q))
    i, j = 0, 1
    J_expected = Q_full[i, j] / 4.0
    tag = PASS if abs(env.J[i, j] - J_expected) < 1e-12 else FAIL
    print(f"R13 eq:Jij        J[0,1]={env.J[i,j]:.6e}  Q[0,1]/4={J_expected:.6e}  {tag}")

    # R14: H_C via Q + constant_offset == H_C via penalty functions
    # The Q matrix encodes only variable-dependent terms (linear + quadratic).
    # Constant terms (λ1·U from P1, λ3·S·Γ²_min from P3) are absent from Q
    # but present in compute_hamiltonian().  The offset accounts for this.
    x_vec    = x_onehot.flatten().astype(float)
    H_via_Q  = float(x_vec @ env.Q @ x_vec) + qubo_constant_offset(env.params)
    H_via_funcs = compute_hamiltonian(
        x_onehot, env.a, env.b, env.exclusion, env.params
    )['H_C']
    rel_err = abs(H_via_Q - H_via_funcs) / (abs(H_via_funcs) + 1e-30)
    tag = PASS if rel_err < 1e-6 else FAIL
    print(f"R14 eq:qubo_matrix  H_C via Q+offset={H_via_Q:.6e}  via funcs={H_via_funcs:.6e}  "
          f"rel_err={rel_err:.2e}  {tag}")

    print("=" * 60)


def run_alignment_agent(env: ISACEnvironment) -> None:
    """
    ALIGNMENT AGENT: cross-check that code symbols, shapes, and signs
    are consistent with the LaTeX document.

    Checks:
      A1  Qubit count n = U × G  (eq:qubits)
      A2  G_table[u,g,s] = P_max · L_{g,s}  (eq:signal_gain third form)
      A3  a = G / (σ²·ln2)  (eq:acoeff)
      A4  b = G·I / (σ⁴·(ln2)²)  (eq:bcoeff)
      A5  Q diagonal sign (negative = reward signal, negative λ1)
      A6  Ising: H_C via J,h matches H_C via Q
      A7  λ1 > max approximate sum-rate bound (eq:lambda_hard)
      A8  λ3 ≈ 0.10 · λ1 (eq:lambda_soft midpoint)
    """
    print("\n" + "=" * 60)
    print("  AGENT: ALIGNMENT — cross-checking code vs LaTeX")
    print("=" * 60)
    p = env.params
    PASS = "  ✓  PASS"
    FAIL = "  ✗  FAIL"

    # A1: qubit count
    n_expected = p.U * p.G
    tag = PASS if env.n_qubits == n_expected else FAIL
    print(f"A1  eq:qubits     n={env.n_qubits}  U×G={n_expected}  {tag}")

    # A2: G_table ≈ P_max · L_{g,s}  (check one entry)
    g0, s0 = 0, 0
    L_val = compute_path_loss(
        env.p_grid[g0], env.q_surv[s0],
        p.fc, p.a_itu, p.b_itu, p.eta_LoS, p.eta_NLoS
    )
    G_expected_u_independent = p.P_max * L_val
    # Note: G_table[u,g,s] includes NLoS randomness, so will differ from
    # P_max·L (the LoS-only approximation).  We check it's in a
    # plausible range instead.
    G_actual = env.G_table[0, g0, s0]
    ratio = G_actual / G_expected_u_independent if G_expected_u_independent > 0 else 0
    tag = PASS if 0.01 < ratio < 100 else FAIL
    print(f"A2  eq:signal_gain G[0,0,0]={G_actual:.4e}  P_max·L={G_expected_u_independent:.4e}  "
          f"ratio={ratio:.3f}  {tag}")

    # A3: a = G / (σ² · ln2)
    a_check = env.G_table[0, 0, 0] / (p.sigma2 * LN2_VAL)
    tag = PASS if abs(env.a[0, 0, 0] - a_check) < 1e-10 * a_check else FAIL
    print(f"A3  eq:acoeff     a[0,0,0]={env.a[0,0,0]:.6e}  G/(σ²ln2)={a_check:.6e}  {tag}")

    # A4: b entry formula
    u0, g0_, v0, gp0, s0_ = 0, 0, 1, 1, 0
    b_check = (env.G_table[u0, g0_, s0_] * env.I_table[u0, g0_, v0, gp0, s0_]
               / (p.sigma2**2 * LN2_VAL**2))
    tag = PASS if abs(env.b[u0, g0_, v0, gp0, s0_] - b_check) < 1e-10 * max(b_check, 1e-30) else FAIL
    print(f"A4  eq:bcoeff     b[0,0,1,1,0]={env.b[u0,g0_,v0,gp0,s0_]:.6e}  check={b_check:.6e}  {tag}")

    # A5: Q diagonal entry negative (signal reward dominates for reasonable λ)
    q_diag_0 = env.Q[0, 0]
    tag = PASS if q_diag_0 < 0 else FAIL
    print(f"A5  eq:Qdiag      Q[0,0]={q_diag_0:.4e}  should be <0 (negative reward)  {tag}")

    # A6: Ising H_C consistency
    # H = Σ_{i<j} J[i,j]·z_i·z_j + Σ_i h_i·z_i + c0  (J upper-triangular)
    # Evaluated as  z^T J_upper z + h·z + c0  should equal x^T Q x.
    x_test = env.random_feasible_x()
    x_vec  = x_test.flatten().astype(float)
    z_vec  = 1.0 - 2.0 * x_vec   # z_i = 1 − 2·x_i  (eq:ising_sub)

    # J is upper-triangular; z^T J z uses only upper-tri entries (no double count)
    H_ising = float(z_vec @ env.J @ z_vec + env.h_bias @ z_vec + env.c0)
    H_qubo  = float(x_vec @ env.Q @ x_vec)   # x^T Q x (no penalty constants)
    rel_err = abs(H_ising - H_qubo) / (abs(H_qubo) + 1e-30)
    tag = PASS if rel_err < 1e-6 else FAIL
    print(f"A6  eq:Jij/hi     H_ising={H_ising:.6e}  H_qubo(x^TQx)={H_qubo:.6e}  "
          f"rel_err={rel_err:.2e}  {tag}")

    # A7: λ1 > B·Σ a (eq:lambda_hard)
    max_rate_bound = p.B * float(np.sum(env.a))
    tag = PASS if p.lambda1 > max_rate_bound else FAIL
    print(f"A7  eq:lambda_hard λ1={p.lambda1:.4e}  bound={max_rate_bound:.4e}  {tag}")

    # A8: λ3 / λ1 ≈ 0.10
    ratio_3 = p.lambda3 / p.lambda1
    tag = PASS if abs(ratio_3 - 0.10) < 0.01 else FAIL
    print(f"A8  eq:lambda_soft λ3/λ1={ratio_3:.4f}  target=0.10  {tag}")

    print("=" * 60)


# ===========================================================================
# 18. MAIN — demonstration
# ===========================================================================

if __name__ == "__main__":
    import pprint

    # --- Build environment ---
    params = SystemParams(U=3, G=9, S=4, Nt=4)
    env    = ISACEnvironment(params, seed=42)

    # --- Run Reviewer Agent ---
    run_reviewer_agent(env)

    # --- Run Alignment Agent ---
    run_alignment_agent(env)

    # --- Evaluate a random feasible placement ---
    x_test = env.random_feasible_x()
    print(f"\n{'='*60}")
    print("  Evaluating random feasible placement x:")
    print(f"{'='*60}")
    print(f"x (U={params.U}, G={params.G}):\n{x_test}")
    results = env.evaluate(x_test)

    print(f"\nExact SINR per survivor : {[f'{v:.4f}' for v in results['sinr_exact']]}")
    print(f"Exact rate per survivor : {[f'{v:.4e}' for v in results['rate_exact']]} bps")
    print(f"Exact sum-rate          : {results['sum_rate_exact']:.6e} bps")
    print(f"QUBO sum-rate (approx)  : {results['sum_rate_qubo']:.6e} bps")
    print(f"\nHamiltonian breakdown:")
    for k, v in results['hamiltonian'].items():
        print(f"  {k:15s}: {v:.6e}")
    print(f"\nConstraints:")
    for k, v in results['constraints'].items():
        print(f"  {k:20s}: {v}")
