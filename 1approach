import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

c = 299792458.0
hbar = 1.054571817e-34

# =============================================================================
# 0. PARAMETERS -- each with the source that justifies its range
# =============================================================================
P = OrderedDict(
    width_nm     = (400, 400, 400),
    height_nm    = (220, 220, 220),
    A_dB_per_cm  = (0.5, 0.5, 0.5),
    FSR_GHz      = (25.0, 100.0, 600.0),
    K1_percent   = (0.5, 1.8, 22.0),
    L_couple_um  = (2.0, 5.0, 15.0),
    pump_nm      = (1580.0, 1580.0, 1580.0),
)
# NOTE: FSR and K1/K2/K3 optimization are deliberately retained to reproduce
# the previously presented First-Approach slides and figures.
typ = {k: v[1] for k, v in P.items()}
pump_nm = typ["pump_nm"]

print("=" * 78)
print("0. PARAMETERS (min, typical, max)")
print("=" * 78)
for k, (lo, t, hi) in P.items():
    print(f"  {k:14s}: [{lo:>7g} , {t:>7g} , {hi:>7g}]")


# =============================================================================
# [UPSTREAM INPUTS] GVD OPTIMISATION IS NOT RECOMPUTED HERE
# =============================================================================
# [SOURCE: teammate GVD optimisation]
# The preceding team study varied the SOI cross-section and wavelength looking
# for a slightly positive / near-zero GVD operating region.  Its selected point
# for the next simulations was:
#       width = 400 nm, height = 220 nm, pump = 1580 nm.
#
# [SOURCE: teammate coupled-microring design note]
# The circuit note adopts n_g = 4.2 as its reference group index.  We use that
# same n_g here so that the First-Approach FSR/radius optimisation is consistent
# with the teammate's circuit-level reference value.
#
# IMPORTANT:
# - GVD D and beta2 are NOT recalculated in this Task-4 script.
# - No numerical D or beta2 is invented here.
# - n_eff(pump)=2.18533 is kept as the fixed phase-index input already used by
#   the First-Approach resonance model.
# - n_g=4.2 is treated as a fixed circuit input.  The teammate note quotes it at
#   its 1.55-um reference point; using it at 1580 nm here is therefore an
#   explicit modelling choice made to keep one consistent group-index value.
#
# To avoid reintroducing an unsupported GVD curve, n_eff(lambda) is represented
# only by the FIRST-ORDER local slope required to satisfy n_g at the pump:
#       n_g = n_eff - lambda * dn_eff/dlambda.
# This makes n_g constant (=4.2) in the narrow filter window and does not claim
# a second-order dispersion model.

NEFF_PUMP = 2.18533
NG_PUMP = 4.2

lambda0_m = pump_nm * 1e-9
dn_dlambda0 = (NEFF_PUMP - NG_PUMP) / lambda0_m

def n_eff(l_nm):
    """Local first-order phase-index model; GVD is not recalculated here."""
    l_m = np.asarray(l_nm, dtype=float) * 1e-9
    return NEFF_PUMP + dn_dlambda0 * (l_m - lambda0_m)

def n_g(l_nm):
    """Fixed group index adopted from the teammate circuit-design note."""
    arr = np.asarray(l_nm, dtype=float)
    return np.full_like(arr, NG_PUMP, dtype=float) if arr.ndim else float(NG_PUMP)

print("\n" + "=" * 78)
print("[UPSTREAM] Geometry from teammate GVD study; n_g from teammate circuit note")
print("=" * 78)
print(f"  w = {typ['width_nm']:.0f} nm, h = {typ['height_nm']:.0f} nm, pump = {pump_nm:.0f} nm")
print("  GVD optimisation is NOT repeated in this Task-4 script.")
print("  upstream GVD conclusion: selected operating point is in the slightly-positive / near-zero-GVD region")
print(f"  n_eff(pump) = {NEFF_PUMP:.5f}  [fixed phase-index input]")
print(f"  n_g          = {NG_PUMP:.4f}  [adopted from teammate circuit-design note]")
print("  D and beta2 are intentionally not assigned numerical values here.")

# Material and FEMWELL helpers are retained only for the physical coupling-gap
# extraction and the fabrication-width sensitivity study later in the script.
def n_Si(l):
    """[SALZ57] Sellmeier fit, crystalline Si, valid 1.357-11.04 um."""
    l2 = l ** 2
    return math.sqrt(1 + 10.6684293 * l2 / (l2 - 0.301516485 ** 2)
                     + 0.0030434748 * l2 / (l2 - 1.13475115 ** 2)
                     + 1.54133408 * l2 / (l2 - 1104 ** 2))

def n_SiO2(l):
    """[MALIT65] Sellmeier fit, fused silica."""
    l2 = l ** 2
    return math.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                     + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                     + 0.8974794 * l2 / (l2 - 9.896161 ** 2))

def strip_mesh(w, h):
    core = shapely.geometry.box(-w / 2, 0, w / 2, h)
    clad = shapely.geometry.box(-4 * w, 0, 4 * w, 6 * h)
    box = shapely.geometry.box(-4 * w, -6 * h, 4 * w, 0)
    res = dict(core={"resolution": 0.02, "distance": 0.3},
               clad={"resolution": 0.05, "distance": 0.3},
               box={"resolution": 0.05, "distance": 0.3})
    return from_meshio(mesh_from_OrderedDict(
        OrderedDict(core=core, clad=clad, box=box), res, default_resolution_max=2))

def te0(mesh, lam_um):
    """TE0 FEMWELL helper used only for fabrication sensitivity."""
    b = Basis(mesh, ElementTriP0())
    eps = b.zeros()
    for dom, nf in {"core": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um) ** 2
    return compute_modes(b, eps, wavelength=lam_um, num_modes=3,
                         order=1).sorted(key=lambda m: -np.real(m.te_fraction))[0]

w_um, h_um = typ["width_nm"] * 1e-3, typ["height_nm"] * 1e-3


# =============================================================================
# [T4-3] THE RING: radius from the FSR target, then Q
# =============================================================================
# The FSR target fixes the radius [RABUS07 Eq. 2.21 solved for R; MEDINA24 sec
# 3.3.2 sizes its rings from FSR.  The radius is therefore not an independent
# variable: FSR is the swept geometry parameter.  The nominal point remains
# 100 GHz for continuity with the previously presented First Approach.
R1_um = c / (n_g(pump_nm) * 2 * np.pi * typ["FSR_GHz"] * 1e9) * 1e6
L1 = 2 * np.pi * R1_um * 1e-6
m1 = round(n_eff(pump_nm) * L1 / (pump_nm * 1e-9))
if m1 % 2 == 0:
    m1 += 1                                   # odd order at the pump -> parity argument
L1 = m1 * pump_nm * 1e-9 / n_eff(pump_nm)     # exact resonance [RABUS07 Eq. 2.71]
L2 = L1 / 2                                   # Vernier 2:1 [RABUS07 2.77-2.78; ASSUMPTION]
R1_um = L1 / (2 * np.pi) * 1e6
# FSR is computed only AFTER L1 is final: the order-rounding above changes L1 by
# ~1 part in m1, and FSR feeds the resonance brackets and the tolerance report.
FSR_nm = pump_nm ** 2 / (n_g(pump_nm) * L1 * 1e9)

def B_field_per_m(A_dB_cm):
    """Field attenuation coefficient B [1/m].

    [SUPERVISOR BOARD]
        A [dB/cm] -> B [1/cm], with field attenuation exp(-B*Delta z).
    Since 20 log10(E_out/E_in) = -A*Delta z_cm,
        B = A ln(10)/20  [1/cm].
    """
    return A_dB_cm * np.log(10) / 20.0 * 100.0


def alpha_power_per_m(A_dB_cm):
    """Power attenuation coefficient alpha [1/m] used in SAVANIER16 Eq. (4).

    alpha_power = 2*B_field = A ln(10)/10 converted from 1/cm to 1/m.
    """
    return A_dB_cm * np.log(10) / 10.0 * 100.0


def alpha_rt_field(A_dB_cm, L_m):
    """Round-trip FIELD attenuation 0 < a <= 1: a = exp(-B L).

    [SUPERVISOR BOARD / SAVANIER16 Eq. 3 notation]
    """
    return np.exp(-B_field_per_m(A_dB_cm) * L_m)


# Backward-compatible alias used below. It is a FULL-round-trip field attenuation.
alpha_of = alpha_rt_field


def Q_factors(L_m, A_dB_cm, K_power):
    """Intrinsic and coupling quality factors.

    [SAVANIER16 Eqs. (4)-(5)]
        Q_U   = 2*pi*n_g / (lambda * alpha_power)
        Q_cpl = 2*pi*n_g*L / (lambda * |kappa|^2)

    Here K_power = |kappa|^2. Eq. (5) is the low-|kappa|^2 approximation.
    """
    lam = pump_nm * 1e-9
    ng = n_g(pump_nm)
    alpha_p = alpha_power_per_m(A_dB_cm)
    Q_o = 2 * np.pi * ng / (lam * alpha_p)
    Q_e = 2 * np.pi * ng * L_m / (lam * K_power)
    return Q_o, Q_e


print("\n" + "=" * 78)
print("[T4-3] Ring resonator")
print("=" * 78)
print(f"FSR target {typ['FSR_GHz']:.0f} GHz  ->  R1 = {R1_um:.1f} um (m1 = {m1}), "
      f"FSR = {FSR_nm:.3f} nm; R2 = R1/2 = {R1_um/2:.1f} um")
print("  nominal 100-GHz point retained for continuity with the previously presented First Approach.")
Q_o, Q_e = Q_factors(L1, typ["A_dB_per_cm"], typ["K1_percent"] / 100)
Q_L = 1 / (1 / Q_o + 1 / Q_e)
print(f"Q_o = {Q_o:.2e}  Q_e = {Q_e:.2e}  Q_L = {Q_L:.2e}  "
      f"(FWHM = {pump_nm/Q_L*1e3:.0f} pm = {c/(pump_nm*1e-9)/Q_L*1e-9:.2f} GHz)")
print(f"  MA17 measures Q_i = 9e5, Q_L = 9.2e4, FWHM 2.1 GHz at 1 dB/cm -- same order.")
for A_r, QU_r in ((0.74, 9.2e5), (1.23, 5.6e5)):
    QU = 2 * np.pi * 4.2 / (pump_nm * 1e-9 * A_r * 100 / 4.3429)
    print(f"  VALIDATION [SAVANIER16 cutback]: {A_r} dB/cm -> Q_U {QU:.2e} vs {QU_r:.1e} "
          f"({(QU/QU_r-1)*100:+.0f} %)")

# -----------------------------------------------------------------------------
# Pair-generation/GVD calculations are intentionally not repeated here.
# They belong to the upstream source-design stage.  This script is now focused
# on the Task-4 two-ring FILTER optimization and physical coupler implementation.
# -----------------------------------------------------------------------------


# =============================================================================
# [T4-4] VERNIER DOUBLE RING: couplings, T(omega), extinction
# =============================================================================
# Two rings in series [RABUS07 Fig. 2.8, Eqs. 2.58-2.63], solved as the 2x2 linear
# system they reduce to. |t|^2+|k|^2 = 1 [Eq. 2.2].
# Butterworth (maximally flat) from RABUS07 Eqs. 2.74-2.76 in their GENERAL form
# (r1 != r2):  mu_2^2 = 0.250 mu_1^4  with  mu_1^2 = k1^2 v_g/(2 pi r1),
# mu_2^2 = k2^2 v_g^2/(4 pi^2 r1 r2)  =>  k2 = (k1^2/2) sqrt(r2/r1).
# For r2 = r1/2 this is k2 = k1^2/(2 sqrt 2); at r1 = r2 it reduces exactly to
# Eq. 2.76, k2 = 0.5 k1^2 -- checked numerically.
# k3: RABUS07 sets k1 = k3. We keep that here (the lean choice): at r2 = r1/2 it
# is NOT rate-matched, but it is what the book prescribes and it costs ~13 % of
# peak T_drop, which we simply report.

def couplers(K1):
    k1, t1 = np.sqrt(K1), np.sqrt(1 - K1)
    k3, t3 = k1, t1                                   # [RABUS07 sec 2.2.1: k1 = k3]
    k2 = (k1 ** 2 / 2) * np.sqrt(0.5)                 # [RABUS07 2.74-2.75, r2=r1/2]
    return k1, t1, k2, np.sqrt(1 - k2 ** 2), k3, t3


def couplers_independent(K1, K2, K3):
    """Independent ideal couplers for the circuit-level optimization.

    Ki are POWER coupling coefficients: Ki = |kappa_i|^2.

    [MODELLING ASSUMPTION -- explicitly requested to be stated]
    Each directional coupler is reciprocal and lossless:
        |t_i|^2 + |kappa_i|^2 = 1.

    The Rabus/Butterworth relations are NOT imposed here; they remain the
    reference/baseline design in couplers(K1).
    """
    if not (0.0 <= K1 < 1.0 and 0.0 <= K2 < 1.0 and 0.0 <= K3 < 1.0):
        raise ValueError("K1, K2, K3 must be power couplings in [0,1).")
    k1, k2, k3 = np.sqrt(K1), np.sqrt(K2), np.sqrt(K3)
    t1, t2, t3 = np.sqrt(1-K1), np.sqrt(1-K2), np.sqrt(1-K3)
    return k1, t1, k2, t2, k3, t3


theta = lambda l, L: 2 * np.pi * n_eff(l) * L / (l * 1e-9)      # [RABUS07 Eq. 2.5]


def series_rings(l, K1, A, dn2=0.0):
    """[RABUS07 Eqs. 2.58-2.63] as a 2x2 system. Energy conservation is checked."""
    k1, t1, k2, t2, k3, t3 = couplers(K1)
    # alpha_of(A,L) is the FIELD attenuation of a FULL round trip.
    # a1/a2 in the Rabus two-ring equations propagate over HALF a round trip,
    # therefore their attenuation is sqrt(alpha_rt).
    a1 = np.sqrt(alpha_of(A, L1)) * np.exp(1j * theta(l, L1) / 2)
    a2 = np.sqrt(alpha_of(A, L2)) * np.exp(
        1j * 2 * np.pi * (n_eff(l) + dn2) * L2 / (l * 1e-9) / 2)
    M = np.array([[1 - t1 * t2 * a1 ** 2, t1 * a1 * k2 * a2],
                  [-t3 * a2 * k2 * a1, 1 - t3 * t2 * a2 ** 2]])
    E1a, E2b = np.linalg.solve(M, [-k1, 0.0])
    E1b = t2 * a1 * E1a - k2 * a2 * E2b
    E2a = k2 * a1 * E1a + t2 * a2 * E2b
    return t1 + k1 * a1 * E1b, k3 * a2 * E2a, E1a, E2a




def series_rings_3K(l, K1, K2, K3, A, dn2=0.0):
    """RABUS07 Eqs. 2.58-2.63 with K1,K2,K3 independent.

    This uses exactly the same two-ring circuit equations as series_rings().
    Only the Butterworth constraint between the three couplers is removed.
    """
    k1_, t1_, k2_, t2_, k3_, t3_ = couplers_independent(K1, K2, K3)

    # alpha_of(A,L) = full-round-trip FIELD attenuation.
    # a1/a2 propagate one half round trip.
    a1_ = np.sqrt(alpha_of(A, L1)) * np.exp(1j * theta(l, L1) / 2)
    a2_ = np.sqrt(alpha_of(A, L2)) * np.exp(
        1j * 2*np.pi*(n_eff(l) + dn2)*L2/(l*1e-9)/2)

    M_ = np.array([
        [1 - t1_*t2_*a1_**2,       t1_*a1_*k2_*a2_],
        [-t3_*a2_*k2_*a1_,         1 - t3_*t2_*a2_**2]
    ])

    E1a_, E2b_ = np.linalg.solve(M_, [-k1_, 0.0])
    E1b_ = t2_*a1_*E1a_ - k2_*a2_*E2b_
    E2a_ = k2_*a1_*E1a_ + t2_*a2_*E2b_

    E_through_ = t1_ + k1_*a1_*E1b_
    E_drop_ = k3_*a2_*E2a_
    return E_through_, E_drop_, E1a_, E2a_


def resonance(L, m, guess):
    """[RABUS07 Eq. 2.71] n_eff(lambda) L/lambda = m, by root finding."""
    f = lambda l: n_eff(l) * L / (l * 1e-9) - m
    fsr = guess ** 2 / (n_g(guess) * L * 1e9)
    return brentq(f, guess - 0.6 * fsr, guess + 0.6 * fsr)


lam_s = resonance(L1, m1 + 1, pump_nm - FSR_nm)
lam_i = resonance(L1, m1 - 1, pump_nm + FSR_nm)


def kpis(K1_pct, A, dn2=0.0):
    K1 = K1_pct / 100
    Tp = abs(series_rings(pump_nm, K1, A, dn2)[1]) ** 2
    Ts = abs(series_rings(lam_s, K1, A, dn2)[1]) ** 2
    Ti = abs(series_rings(lam_i, K1, A, dn2)[1]) ** 2
    return dict(Tp=Tp, Ts=Ts, Ti=Ti, rej=-10 * np.log10(max(Tp, 1e-300)),
                ER=10 * np.log10(max(Ts, 1e-300) / max(Tp, 1e-300)))




def kpis_3K(K1, K2, K3, A, dn2=0.0):
    """Circuit KPIs with all three POWER couplings independent."""
    try:
        Tp_ = abs(series_rings_3K(pump_nm, K1, K2, K3, A, dn2)[1])**2
        Ts_ = abs(series_rings_3K(lam_s,   K1, K2, K3, A, dn2)[1])**2
        Ti_ = abs(series_rings_3K(lam_i,   K1, K2, K3, A, dn2)[1])**2
    except np.linalg.LinAlgError:
        return dict(Tp=np.nan, Ts=np.nan, Ti=np.nan, rej=np.nan,
                    ER=np.nan, pair_pass=np.nan)

    rej_ = -10*np.log10(max(Tp_, 1e-300))
    return dict(
        Tp=Tp_, Ts=Ts_, Ti=Ti_,
        rej=rej_,
        ER=10*np.log10(max(Ts_, 1e-300)/max(Tp_, 1e-300)),
        pair_pass=Ts_*Ti_
    )


print("\n" + "=" * 78)
print("[T4-4] Vernier double ring")
print("=" * 78)
k1, t1, k2, t2, k3, t3 = couplers(typ["K1_percent"] / 100)
print(f"six couplings: (t1,k1)=({t1:.4f},{k1:.4f})  (t2,k2)=({t2:.5f},{k2:.5f})  "
      f"(t3,k3)=({t3:.4f},{k3:.4f})")
print(f"pump {pump_nm:.3f} / signal {lam_s:.3f} / idler {lam_i:.3f} nm")
# parity check: L2 = L1/2 and the same index -> theta2 = theta1/2 identically
for nm_, l_ in (("pump", pump_nm), ("signal", lam_s), ("idler", lam_i)):
    fr = (2 * np.pi * n_eff(l_) * L2 / (l_ * 1e-9) / (2 * np.pi)) % 1.0
    det_pm = min(fr, 1 - fr) * (pump_nm ** 2 / (n_g(pump_nm) * L2 * 1e9)) * 1e3
    st = "RESONANT" if det_pm < pump_nm / Q_L * 1e3 / 2 else (
        "ANTI-RESONANT" if abs(fr - .5) < .02 else "detuned")
    print(f"  theta2/2pi at {nm_:6s}: frac = {fr:.4f} ({det_pm:6.0f} pm) -> {st}")
KP = kpis(typ["K1_percent"], typ["A_dB_per_cm"])
print(f"T_drop(pump) = {KP['Tp']:.2e} -> rejection {KP['rej']:.1f} dB")
print(f"T_drop(signal) = {KP['Ts']:.3f}, T_drop(idler) = {KP['Ti']:.3f}, "
      f"extinction = {KP['ER']:.1f} dB")
print("Route to the 100-120 dB target [minutes]:")
for N in (1, 2, 3):
    signal_kept = KP['Ts']**N
    pair_kept = (KP['Ts']*KP['Ti'])**N
    print(f"  {N} stage(s): {N*KP['rej']:.0f} dB, "
          f"signal kept {signal_kept:.3f}, pair survival {pair_kept:.3e}")
print("  WANG24: CROW filters reach >96 dB on-chip, 'exceeding the performance of")
print("  commercially available filters'. MA17, the CAR>12000 record, still used")
print("  EXTERNAL benchtop filters -- on-chip 100 dB is not yet routine.")


# =============================================================================
# [T4-OPT] SINGLE-DEVICE SWEEP OF K1 -- EXACTLY TWO RINGS
# =============================================================================
# The physical architecture is fixed to ONE double-ring device (R1 + R2).
# Cascading additional double-ring stages is NOT an optimization variable.
#
# K1 range:
#   [SAVANIER16] |kappa|^2 ~ 0.005 and 0.018 experimentally inferred.
#   [STRAIN15] tunable Si microring coupling spans ~0.22 down to <0.005.
#
# The 100 dB value from MEDINA24 is treated as a REQUIREMENT TO TEST, not
# something the code is allowed to satisfy by duplicating the device.
#
# We report three transparent single-device quantities:
#   A) maximum pump rejection;
#   B) maximum useful joint transmission Ts*Ti;
#   C) a relative source-aware KPI ~ (QL^3/R^2)*(Ts*Ti).
#
# C is only a comparative baseline metric while K2,K3 remain tied to the
# Butterworth prescription. It is not used as the final 3-K objective.

print("\n" + "=" * 78)
print("[T4-OPT] Single two-ring sweep of K1")
print("=" * 78)

REJ_TARGET_DB = 100.0                         # [SOURCE: MEDINA24 Sec. 3.4]
REJ_OPERATING_DB = 40.0                       # engineering constraint for max-TsTi operating point
K1_grid = np.geomspace(0.005, 0.22, 320)     # [SOURCE RANGE: SAVANIER16 + STRAIN15]

opt_K1 = []
R1_m = L1 / (2*np.pi)

for K1_power in K1_grid:
    kp_ = kpis(100.0 * K1_power, typ["A_dB_per_cm"])
    Qo_, Qe_ = Q_factors(L1, typ["A_dB_per_cm"], K1_power)
    QL_ = 1.0 / (1.0/Qo_ + 1.0/Qe_)

    pair_pass_ = kp_["Ts"] * kp_["Ti"]
    source_rel_ = QL_**3 / R1_m**2                       # [SAVANIER16 relative scaling]
    system_rel_single_ = source_rel_ * pair_pass_       # [DERIVED single-device KPI]

    opt_K1.append(dict(
        K1=K1_power, Qo=Qo_, Qe=Qe_, QL=QL_,
        Tp=kp_["Tp"], Ts=kp_["Ts"], Ti=kp_["Ti"],
        rej_stage=kp_["rej"],
        pair_pass=pair_pass_,
        source_rel=source_rel_,
        system_rel=system_rel_single_,
    ))

best_rej_K1 = max(opt_K1, key=lambda r: r["rej_stage"])
best_trans = max(opt_K1, key=lambda r: r["pair_pass"])
best_system = max(opt_K1, key=lambda r: r["system_rel"])

single_feasible = [r for r in opt_K1 if r["rej_stage"] >= REJ_TARGET_DB]
best_feasible_K1 = max(single_feasible, key=lambda r: r["pair_pass"]) if single_feasible else None

# Critical coupling predicted by the same loss/Q equations.
Kcrit = alpha_power_per_m(typ["A_dB_per_cm"]) * L1

print("\nDerived critical-coupling benchmark:")
print(f"  Kcrit = alpha_power*L = {100*Kcrit:.3f} %")
print("  [derived from SAVANIER16 Eqs. 4-5 by setting Qcpl = QU]")

print("\nOBJECTIVE A -- maximum pump rejection of ONE double-ring device")
print(f"  K1 = {100*best_rej_K1['K1']:.3f} %")
print(f"  rejection = {best_rej_K1['rej_stage']:.2f} dB")
print(f"  Ts = {best_rej_K1['Ts']:.4f}, Ti = {best_rej_K1['Ti']:.4f}")
print(f"  Ts*Ti = {best_rej_K1['pair_pass']:.4e}")

print("\nOBJECTIVE B -- maximum useful joint transmission of ONE device")
print(f"  K1 = {100*best_trans['K1']:.3f} %")
print(f"  rejection = {best_trans['rej_stage']:.2f} dB")
print(f"  Ts = {best_trans['Ts']:.4f}, Ti = {best_trans['Ti']:.4f}")
print(f"  Ts*Ti = {best_trans['pair_pass']:.4f}")

print("\nOBJECTIVE C -- source-aware relative SINGLE-device KPI")
print("  metric ~ (QL^3/R^2)*(Ts*Ti)")
print(f"  K1 = {100*best_system['K1']:.3f} %")
print(f"  rejection = {best_system['rej_stage']:.2f} dB")
print(f"  Ts*Ti = {best_system['pair_pass']:.4e}")
print(f"  QL = {best_system['QL']:.3e}")

if best_feasible_K1 is None:
    print("\n100 dB FEASIBILITY:")
    print("  NO K1 in the documented 0.5%-22% interval makes this one")
    print("  two-ring Butterworth device reach the 100 dB requirement.")
else:
    print("\n100 dB FEASIBILITY:")
    print("  A feasible single-device region exists.")
    print(f"  best feasible K1 = {100*best_feasible_K1['K1']:.3f}%")
    print(f"  rejection = {best_feasible_K1['rej_stage']:.2f} dB")
    print(f"  Ts*Ti = {best_feasible_K1['pair_pass']:.4e}")

# Plot: only the two directly interpretable quantities used in the First Approach.
xK = np.array([r["K1"] for r in opt_K1]) * 100
rejK = np.array([r["rej_stage"] for r in opt_K1])
pairK = np.array([r["pair_pass"] for r in opt_K1])

fopt, aopt = plt.subplots(1, 2, figsize=(10.5, 4.5))

aopt[0].semilogx(xK, rejK)
aopt[0].axhline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
aopt[0].axhline(REJ_OPERATING_DB, ls=":", label="40 dB operating constraint")
aopt[0].axvline(100*Kcrit, ls="-.", label="derived critical coupling")
aopt[0].scatter([100*best_rej_K1["K1"]], [best_rej_K1["rej_stage"]],
                marker="*", s=100, label="max rejection")
aopt[0].set_xlabel("K1 power coupling (%)")
aopt[0].set_ylabel("pump rejection (dB)")
aopt[0].set_title("(a) single-device rejection")
aopt[0].grid(alpha=.3)
aopt[0].legend(fontsize=7)

aopt[1].semilogx(xK, pairK)
aopt[1].scatter([100*best_trans["K1"]], [best_trans["pair_pass"]],
                marker="*", s=100, label="max TsTi")
aopt[1].set_xlabel("K1 power coupling (%)")
aopt[1].set_ylabel(r"$T_sT_i$")
aopt[1].set_title("(b) useful pair transmission")
aopt[1].grid(alpha=.3)
aopt[1].legend(fontsize=7)

fopt.suptitle("[T4-OPT] Exactly two rings: K1 trade-offs without cascaded stages")
fopt.tight_layout()
fopt.savefig("lean_fig_OPT_K1.png", dpi=170)
plt.close(fopt)



# =============================================================================
# [T4-4b] THE COUPLINGS FROM THE GAP -- CHROST15 Eqs. 4.3 / 4.5 / 4.8
# =============================================================================
def coupled_mesh(w, h, gap):
    c1 = shapely.geometry.box(-gap / 2 - w, 0, -gap / 2, h)
    c2 = shapely.geometry.box(gap / 2, 0, gap / 2 + w, h)
    sp = 2 * w + gap
    clad = shapely.geometry.box(-2 * sp, 0, 2 * sp, 6 * h)
    box = shapely.geometry.box(-2 * sp, -6 * h, 2 * sp, 0)
    res = {k: {"resolution": 0.02 if k.startswith("core") else 0.05, "distance": 0.3}
           for k in ("core1", "core2", "clad", "box")}
    return from_meshio(mesh_from_OrderedDict(
        OrderedDict(core1=c1, core2=c2, clad=clad, box=box), res, default_resolution_max=2))


def kappa_z_for_geometry(gap_nm, lam_um, w_local_um, h_local_um):
    """[CHROST15 Eq. 4.3] kappa_z = pi (n_even-n_odd)/lambda.

    The geometry is explicit so the CHROST15 benchmark can be evaluated on the
    reference 500x220-nm cross-section independently of the actual device.
    """
    m = coupled_mesh(w_local_um, h_local_um, gap_nm * 1e-3)
    b = Basis(m, ElementTriP0()); eps = b.zeros()
    for dom, nf in {"core1": n_Si, "core2": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um) ** 2
    md = compute_modes(b, eps, wavelength=lam_um, num_modes=2, order=1)
    ne, no = sorted((np.real(md[0].n_eff), np.real(md[1].n_eff)), reverse=True)
    return np.pi / lam_um * (ne - no)


def kappa_z(gap_nm, lam_um):
    """Actual-device coupling for the fixed upstream 400x220-nm geometry."""
    return kappa_z_for_geometry(gap_nm, lam_um, w_um, h_um)


print("\n" + "=" * 78)
print("[T4-4b] Couplings from the gap (Femwell supermodes + CHROST15 Eq. 4.8)")
print("=" * 78)

# [MODELLING CHOICE: NUMERICAL DOMAIN, not a fabrication claim]
# Earlier versions solved only 150-350 nm and then extrapolated to ~600 nm for
# the very weak inter-ring coupling.  That is not acceptable for the final
# optimization.  We therefore solve FEMWELL explicitly over 100-700 nm.
# The endpoints are chosen only to CONTAIN the couplings required by the
# circuit study; no result outside this domain is used.
gaps = np.array([100, 125, 150, 175, 200, 250, 300, 350,
                 400, 450, 500, 550, 600, 650, 700], dtype=float)
kzv = np.array([kappa_z(g, pump_nm * 1e-3) for g in gaps])

print("FEMWELL supermode samples:")
for g_, kz_ in zip(gaps, kzv):
    print(f"  gap {g_:4.0f} nm -> kappa_z = {kz_:.6e} rad/um")

_i200 = int(np.where(gaps == 200)[0][0])
Lx200_device = np.pi / (2*kzv[_i200])
print(f"\nACTUAL DEVICE (400x220 nm @ {pump_nm:.0f} nm):")
print(f"  L_x(200 nm gap) = {Lx200_device:.1f} um")
print("  This is NOT compared directly to the 37.5-um CHROST15 value because")
print("  the waveguide width and wavelength are different.")

# Independent METHOD benchmark on the geometry of CHROST15 Fig. 4.14b.
kz_chrost_benchmark = kappa_z_for_geometry(
    gap_nm=200.0, lam_um=1.550, w_local_um=0.500, h_local_um=0.220
)
Lx_chrost_benchmark = np.pi / (2*kz_chrost_benchmark)
chrost_err_pct = (Lx_chrost_benchmark/37.5 - 1.0)*100

print("\nMETHOD BENCHMARK [CHROST15 Fig. 4.14b]:")
print("  auxiliary geometry = 500x220 nm, lambda = 1550 nm, gap = 200 nm")
print(f"  FEMWELL kappa_z = {kz_chrost_benchmark:.6f} rad/um")
print(f"  FEMWELL L_x = {Lx_chrost_benchmark:.2f} um vs ~37.5 um "
      f"({chrost_err_pct:+.2f} %)")
print("  -> this validates the supermode METHOD on the reference geometry;")
print("     it does not force the actual 400x220-nm device to have the same coupling.")

# [NUMERICAL INTERPOLATION]
# PCHIP of log(kappa_z) is used ONLY between the FEMWELL-simulated points.
# No extrapolation is allowed.
_log_kz_interp = PchipInterpolator(gaps, np.log(kzv), extrapolate=False)

def kappa_z_interp(gap_nm):
    g = np.asarray(gap_nm, dtype=float)
    if np.any(g < gaps.min()) or np.any(g > gaps.max()):
        raise ValueError(
            f"gap={gap_nm} nm outside FEMWELL-simulated domain "
            f"[{gaps.min():.0f},{gaps.max():.0f}] nm"
        )
    val = np.exp(_log_kz_interp(g))
    return float(val) if val.ndim == 0 else val


L_c = typ["L_couple_um"]

def K_of_gap(gap_nm):
    """POWER coupling from physical gap, without extrapolation.

    [CHROST15 Eq. 4.8]
        K = |kappa|^2 = sin^2(kappa_z * L_c)

    kappa_z comes from FEMWELL supermodes plus interpolation between
    physically simulated gaps.
    """
    return np.sin(kappa_z_interp(gap_nm)*L_c)**2


GAP_MIN_NM = float(gaps.min())
GAP_MAX_NM = float(gaps.max())
K_AT_GMIN = float(K_of_gap(GAP_MIN_NM))
K_AT_GMAX = float(K_of_gap(GAP_MAX_NM))
K_PHYS_MIN = min(K_AT_GMIN, K_AT_GMAX)
K_PHYS_MAX = max(K_AT_GMIN, K_AT_GMAX)

print(f"\nCHROST15 Eq. 4.8 with L_c = {L_c:.1f} um:")
print(f"  physically simulated coupling interval over {GAP_MIN_NM:.0f}-{GAP_MAX_NM:.0f} nm:")
print(f"  {K_PHYS_MIN:.3e} <= K <= {K_PHYS_MAX:.3e}")


def gap_for_K(K_target):
    """Invert K(g) ONLY inside the FEMWELL-simulated gap interval."""
    if not (K_PHYS_MIN <= K_target <= K_PHYS_MAX):
        return np.nan
    return brentq(lambda g: K_of_gap(g) - K_target, GAP_MIN_NM, GAP_MAX_NM)


print("\nNominal Rabus/Butterworth design mapped to physical gaps:")
gaps_design = {}
for nm_, Ktarget_ in (("bus<->R1", k1**2),
                      ("R1<->R2", k2**2),
                      ("R2<->drop", k3**2)):
    g_ = gap_for_K(Ktarget_)
    gaps_design[nm_] = g_
    if np.isfinite(g_):
        print(f"  {nm_:10s}: K = {Ktarget_:.6e} -> gap = {g_:.1f} nm")
    else:
        print(f"  {nm_:10s}: K = {Ktarget_:.6e} -> OUTSIDE simulated physical range")

for label_, result_ in (("filter-only K1 optimum", best_trans),
                        ("source-aware K1 optimum", best_system)):
    g_ = gap_for_K(result_["K1"])
    if np.isfinite(g_):
        print(f"{label_}: K1={100*result_['K1']:.3f}% -> gap={g_:.1f} nm")
    else:
        print(f"{label_}: K1={100*result_['K1']:.3f}% is outside simulated K(g).")


# =============================================================================
# [T4-OPT3] INDEPENDENT K1/K2/K3 OPTIMIZATION -- EXACTLY TWO RINGS
# =============================================================================
# The Rabus/Butterworth design remains the BASELINE.
# K1,K2,K3 are then independently varied in the same two-ring circuit equations.
#
# HARD ARCHITECTURAL CONSTRAINT:
#     exactly one R1 + one R2 device.
# The optimizer is NOT allowed to duplicate the double-ring stage.
#
# Search domains:
#   K1,K3: intersection of the documented 0.5%-22% bus-ring range and the
#          physically simulated FEMWELL K(g) interval.
#   K2:    directly from the physically simulated FEMWELL K(g) interval.
#
# Objectives:
#   A) maximize single-device pump rejection;
#   B) maximize single-device useful joint transmission Ts*Ti;
#   C) test feasibility of rejection >= 100 dB.  If feasible, maximize Ts*Ti
#      inside that feasible set.  If not feasible, report the architecture as
#      infeasible for the stated target within the investigated domain.
#
# The Pareto frontier is therefore the central result:
#          pump rejection  <->  Ts*Ti

print("\n" + "="*78)
print("[T4-OPT3] Independent K1 / K2 / K3 -- EXACTLY TWO RINGS")
print("="*78)

K13_MIN = max(0.005, K_PHYS_MIN)
K13_MAX = min(0.22,  K_PHYS_MAX)

if not (K13_MIN < K13_MAX):
    raise RuntimeError("FEMWELL K(g) range does not overlap the documented K1/K3 range.")

K1_grid_3 = np.geomspace(K13_MIN, K13_MAX, 22)
K3_grid_3 = np.geomspace(K13_MIN, K13_MAX, 22)
K2_grid_3 = np.geomspace(max(K_PHYS_MIN, 1e-12), K_PHYS_MAX, 26)

results_3K = []

for K1_ in K1_grid_3:
    for K2_ in K2_grid_3:
        for K3_ in K3_grid_3:
            r_ = kpis_3K(K1_, K2_, K3_, typ["A_dB_per_cm"])
            if not np.isfinite(r_["rej"] + r_["Ts"] + r_["Ti"]):
                continue

            results_3K.append(dict(
                K1=K1_, K2=K2_, K3=K3_,
                Tp=r_["Tp"], Ts=r_["Ts"], Ti=r_["Ti"],
                rej_stage=r_["rej"],
                pair_pass=r_["Ts"]*r_["Ti"],
            ))

best_rejection_3K = max(results_3K, key=lambda r: r["rej_stage"])
best_pass_3K = max(results_3K, key=lambda r: r["pair_pass"])

feasible_3K = [r for r in results_3K if r["rej_stage"] >= REJ_TARGET_DB]
best_feasible_3K = max(feasible_3K, key=lambda r: r["pair_pass"]) if feasible_3K else None

# Engineering operating point: maximize useful signal-idler transmission
# while satisfying a minimum pump-rejection requirement.
feasible_40_3K = [r for r in results_3K if r["rej_stage"] >= REJ_OPERATING_DB]
best_40_3K = max(feasible_40_3K, key=lambda r: r["pair_pass"]) if feasible_40_3K else None

K1_BW = typ["K1_percent"]/100.0
_k1bw, _t1bw, _k2bw, _t2bw, _k3bw, _t3bw = couplers(K1_BW)
baseline_3K = kpis_3K(K1_BW, _k2bw**2, _k3bw**2, typ["A_dB_per_cm"])

def _print_3k_result(title, r):
    print(f"\n{title}")
    print(f"  K1 = {100*r['K1']:.6f} %")
    print(f"  K2 = {100*r['K2']:.6f} %")
    print(f"  K3 = {100*r['K3']:.6f} %")
    print(f"  rejection = {r['rej_stage']:.2f} dB")
    print(f"  Ts = {r['Ts']:.5f}, Ti = {r['Ti']:.5f}, Ts*Ti = {r['pair_pass']:.5f}")
    for nm_, kval_ in (("g1", r["K1"]), ("g12", r["K2"]), ("g3", r["K3"])):
        gv_ = gap_for_K(kval_)
        print(f"  {nm_:3s} = {gv_:7.2f} nm" if np.isfinite(gv_)
              else f"  {nm_:3s} = outside simulated K(g)")

print("\nBASELINE -- Rabus/Butterworth")
print(f"  K1={100*K1_BW:.6f} %, K2={100*_k2bw**2:.6f} %, K3={100*_k3bw**2:.6f} %")
print(f"  rejection={baseline_3K['rej']:.2f} dB")
print(f"  Ts={baseline_3K['Ts']:.5f}, Ti={baseline_3K['Ti']:.5f}, "
      f"Ts*Ti={baseline_3K['Ts']*baseline_3K['Ti']:.5f}")

_print_3k_result("OBJECTIVE A -- maximum single-device pump rejection",
                 best_rejection_3K)
_print_3k_result("OBJECTIVE B -- maximum single-device Ts*Ti",
                 best_pass_3K)

print("\nOBJECTIVE C -- 100 dB single-device feasibility")
if best_feasible_3K is None:
    print("  INFEASIBLE in the investigated physical domain.")
    print(f"  Maximum sampled rejection = {best_rejection_3K['rej_stage']:.2f} dB")
    print("  -> the two-ring topology does not meet the 100 dB target in this model.")
else:
    _print_3k_result("  best feasible design (max Ts*Ti with rejection >=100 dB)",
                     best_feasible_3K)

print(f"\nOPERATING POINT -- maximum Ts*Ti with rejection >= {REJ_OPERATING_DB:.0f} dB")
if best_40_3K is None:
    print("  No fixed-radius 3K design satisfies the operating constraint.")
else:
    _print_3k_result("  best constrained fixed-radius design", best_40_3K)

# Pareto frontier: keep a point only if no sampled design has both greater
# rejection and greater/equal pair transmission.
_sorted = sorted(results_3K, key=lambda r: r["rej_stage"], reverse=True)
pareto = []
best_pair_seen = -np.inf
for r_ in _sorted:
    if r_["pair_pass"] > best_pair_seen:
        pareto.append(r_)
        best_pair_seen = r_["pair_pass"]

f3k, a3k = plt.subplots(1, 2, figsize=(11.5, 4.5))

a3k[0].scatter([r["rej_stage"] for r in results_3K],
               [r["pair_pass"] for r in results_3K],
               s=5, alpha=.15, label="physical-grid designs")
a3k[0].plot([r["rej_stage"] for r in pareto],
            [r["pair_pass"] for r in pareto],
            linewidth=2, label="Pareto frontier")
a3k[0].axvline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
a3k[0].axvline(REJ_OPERATING_DB, ls=":", label="40 dB operating constraint")
a3k[0].scatter([baseline_3K["rej"]],
               [baseline_3K["Ts"]*baseline_3K["Ti"]],
               marker="x", s=70, label="Butterworth baseline")
a3k[0].scatter([best_rejection_3K["rej_stage"]],
               [best_rejection_3K["pair_pass"]],
               marker="*", s=120, label="max rejection")
if best_40_3K is not None:
    a3k[0].scatter([best_40_3K["rej_stage"]],
                   [best_40_3K["pair_pass"]],
                   marker="D", s=75, label="max TsTi, rej >= 40 dB")
a3k[0].set_xlabel("pump rejection of one double-ring (dB)")
a3k[0].set_ylabel(r"$T_sT_i$")
a3k[0].set_title("(a) rejection / transmission Pareto trade-off")
a3k[0].grid(alpha=.3)
a3k[0].legend(fontsize=7)

_labels = ["K1", "K2", "K3"]
_bw_vals = np.array([K1_BW, _k2bw**2, _k3bw**2])*100
_rej_vals = np.array([
    best_rejection_3K["K1"],
    best_rejection_3K["K2"],
    best_rejection_3K["K3"]
])*100
_x = np.arange(3)

a3k[1].semilogy(_x, _bw_vals, "o-", label="Butterworth")
a3k[1].semilogy(_x, _rej_vals, "s-", label="max-rejection design")
if best_40_3K is not None:
    _op_vals = np.array([best_40_3K["K1"], best_40_3K["K2"], best_40_3K["K3"]])*100
    a3k[1].semilogy(_x, _op_vals, "d-", label="max TsTi, rej >= 40 dB")
a3k[1].set_xticks(_x, _labels)
a3k[1].set_ylabel("POWER coupling K (%)")
a3k[1].set_title("(b) coupling configurations")
a3k[1].grid(alpha=.3, which="both")
a3k[1].legend(fontsize=7)

f3k.suptitle("[T4-OPT3] Exactly two rings: independent K1/K2/K3 optimization")
f3k.tight_layout()
f3k.savefig("lean_fig_OPT_3K.png", dpi=170)
plt.close(f3k)



# =============================================================================
# [T4-OPTR] RADIUS + COUPLING OPTIMIZATION -- EXACTLY TWO RINGS
# =============================================================================
# Architectural constraint:
#       R2 = R1/2
# and there is exactly ONE R1+R2 double-ring device.
#
# [MODELLING CHOICE] We now broaden the First-Approach FSR sensitivity study to
# 25-600 GHz.  The nominal point remains 100 GHz so the previously presented
# baseline remains directly comparable.  This broader interval is exploratory;
# it is not claimed as a literature-prescribed fabrication range.
#
# The pump is snapped to the nearest ODD longitudinal order so the 2:1 parity
# argument is preserved.  At the high-FSR end the resulting radii become small,
# so the straight-waveguide n_eff and straight-coupler K(g) approximations are
# only first-order circuit-level approximations; curvature-dependent coupler
# extraction would be required for a mask-final design.
#
# The joint search asks:
#   A) What is the maximum single-device pump rejection?
#   B) What is the maximum single-device Ts*Ti?
#   C) Does ANY sampled two-ring design reach >=100 dB?
#   D) Among designs with rejection >=40 dB, maximize Ts*Ti.
#
# No cascaded-stage multiplication is used.

print("\n" + "="*78)
print("[T4-OPTR] Radius optimization -- EXACTLY TWO RINGS")
print("="*78)


def geometry_from_fsr_target(fsr_target_GHz):
    """Self-consistent pump-resonant geometry from target FSR."""
    L_guess = c / (n_g(pump_nm) * fsr_target_GHz * 1e9)
    R_guess_um = L_guess / (2*np.pi) * 1e6

    m1_ = round(n_eff(pump_nm) * L_guess / (pump_nm * 1e-9))
    if m1_ % 2 == 0:
        m_lo = m1_ - 1
        m_hi = m1_ + 1
        L_lo = m_lo * pump_nm * 1e-9 / n_eff(pump_nm)
        L_hi = m_hi * pump_nm * 1e-9 / n_eff(pump_nm)
        m1_ = m_lo if abs(L_lo - L_guess) <= abs(L_hi - L_guess) else m_hi

    L1_ = m1_ * pump_nm * 1e-9 / n_eff(pump_nm)
    L2_ = L1_ / 2.0
    R1_ = L1_ / (2*np.pi) * 1e6
    R2_ = L2_ / (2*np.pi) * 1e6

    fsr_GHz_actual_ = c / (n_g(pump_nm) * L1_) * 1e-9
    fsr_nm_ = pump_nm**2 / (n_g(pump_nm) * L1_ * 1e9)

    ls_ = resonance(L1_, m1_ + 1, pump_nm - fsr_nm_)
    li_ = resonance(L1_, m1_ - 1, pump_nm + fsr_nm_)

    return dict(
        fsr_target_GHz=float(fsr_target_GHz),
        fsr_GHz=float(fsr_GHz_actual_),
        FSR_nm=float(fsr_nm_),
        m1=int(m1_),
        L1=float(L1_),
        L2=float(L2_),
        R1_um=float(R1_),
        R2_um=float(R2_),
        ls=float(ls_),
        li=float(li_),
        R_guess_um=float(R_guess_um)
    )


def series_rings_geom(l, geom, K1, K2, K3, A):
    """Same Rabus two-ring equations with local L1,L2."""
    k1_, t1_, k2_, t2_, k3_, t3_ = couplers_independent(K1, K2, K3)
    L1_, L2_ = geom["L1"], geom["L2"]

    a1_ = np.sqrt(alpha_of(A, L1_)) * np.exp(1j * theta(l, L1_) / 2)
    a2_ = np.sqrt(alpha_of(A, L2_)) * np.exp(1j * theta(l, L2_) / 2)

    M_ = np.array([
        [1 - t1_*t2_*a1_**2,       t1_*a1_*k2_*a2_],
        [-t3_*a2_*k2_*a1_,         1 - t3_*t2_*a2_**2]
    ])
    E1a_, E2b_ = np.linalg.solve(M_, [-k1_, 0.0])
    E1b_ = t2_*a1_*E1a_ - k2_*a2_*E2b_
    E2a_ = k2_*a1_*E1a_ + t2_*a2_*E2b_

    E_through_ = t1_ + k1_*a1_*E1b_
    E_drop_ = k3_*a2_*E2a_
    return E_through_, E_drop_, E1a_, E2a_


def kpis_radius(geom, K1, K2, K3, A):
    """Single-device KPIs for one radius and one K1,K2,K3 triplet."""
    try:
        Tp_ = abs(series_rings_geom(pump_nm, geom, K1, K2, K3, A)[1])**2
        Ts_ = abs(series_rings_geom(geom["ls"], geom, K1, K2, K3, A)[1])**2
        Ti_ = abs(series_rings_geom(geom["li"], geom, K1, K2, K3, A)[1])**2
    except np.linalg.LinAlgError:
        return dict(Tp=np.nan, Ts=np.nan, Ti=np.nan, rej=np.nan, pair_pass=np.nan)

    rej_ = -10*np.log10(max(Tp_, 1e-300))
    return dict(
        Tp=Tp_, Ts=Ts_, Ti=Ti_,
        rej=rej_,
        pair_pass=Ts_*Ti_
    )


# [NUMERICAL CHOICE]
# Sample every 25 GHz over 25-600 GHz so the historical 100- and 200-GHz points
# remain explicitly present.  We also insert the FSR corresponding to R1=20 um
# at n_g=4.2, enabling a direct comparison with the teammate geometry.
FSR_TEAM_R20_GHz = c / (NG_PUMP * 2*np.pi * 20e-6) * 1e-9
FSR_grid_R = np.unique(np.concatenate([
    np.arange(P["FSR_GHz"][0], P["FSR_GHz"][2] + 0.1, 25.0),
    [FSR_TEAM_R20_GHz]
]))

R1_nominal_min_um = c / (NG_PUMP * 2*np.pi * P["FSR_GHz"][2] * 1e9) * 1e6
R1_nominal_max_um = c / (NG_PUMP * 2*np.pi * P["FSR_GHz"][0] * 1e9) * 1e6
print(f"  exploratory FSR grid: {P['FSR_GHz'][0]:.0f}-{P['FSR_GHz'][2]:.0f} GHz")
print(f"  nominal radius span before pump-order snapping: R1 ~ {R1_nominal_min_um:.2f}-{R1_nominal_max_um:.2f} um")
print(f"  corresponding R2 span: ~ {R1_nominal_min_um/2:.2f}-{R1_nominal_max_um/2:.2f} um")
print(f"  teammate R1=20 um corresponds to FSR ~ {FSR_TEAM_R20_GHz:.2f} GHz at n_g={NG_PUMP:.1f}")
print("  WARNING: small-radius/high-FSR points reuse the straight-coupler K(g) map;")
print("           interpret them as exploratory circuit-level results until curved-coupler validation.")

# -------------------------------------------------------------------------
# (1) Radius-only sweep with K fixed at the maximum-rejection 3K design.
# -------------------------------------------------------------------------
radius_sweep = []
for fsr_target_ in FSR_grid_R:
    geom_ = geometry_from_fsr_target(fsr_target_)
    r_ = kpis_radius(
        geom_,
        best_rejection_3K["K1"],
        best_rejection_3K["K2"],
        best_rejection_3K["K3"],
        typ["A_dB_per_cm"]
    )
    radius_sweep.append({**geom_, **r_})

best_radius_rejection_fixedK = max(radius_sweep, key=lambda r: r["rej"])
best_radius_pair_fixedK = max(radius_sweep, key=lambda r: r["pair_pass"])

print("\nRADIUS-ONLY SWEEP -- K1,K2,K3 fixed at the max-rejection 3K point")
print(f"  FSR interval = {P['FSR_GHz'][0]:.0f}-{P['FSR_GHz'][2]:.0f} GHz")
print("  maximum rejection over radius:")
print(f"    target/actual FSR = {best_radius_rejection_fixedK['fsr_target_GHz']:.2f} / "
      f"{best_radius_rejection_fixedK['fsr_GHz']:.3f} GHz")
print(f"    R1 = {best_radius_rejection_fixedK['R1_um']:.3f} um")
print(f"    R2 = {best_radius_rejection_fixedK['R2_um']:.3f} um")
print(f"    rejection = {best_radius_rejection_fixedK['rej']:.2f} dB")
print(f"    Ts*Ti = {best_radius_rejection_fixedK['pair_pass']:.5f}")

# -------------------------------------------------------------------------
# (2) Joint search over radius + K1 + K2 + K3.
# -------------------------------------------------------------------------
best_rej_per_radius = []
best_pair_per_radius = []
best_feasible_per_radius = []
best_40_per_radius = []

for fsr_target_ in FSR_grid_R:
    geom_ = geometry_from_fsr_target(fsr_target_)
    local_best_rej = None
    local_best_pair = None
    local_best_feasible = None
    local_best_40 = None

    for K1_ in K1_grid_3:
        for K2_ in K2_grid_3:
            for K3_ in K3_grid_3:
                r_ = kpis_radius(geom_, K1_, K2_, K3_, typ["A_dB_per_cm"])
                if not np.isfinite(r_["rej"] + r_["pair_pass"]):
                    continue

                cand_ = {
                    **geom_, **r_,
                    "K1": float(K1_), "K2": float(K2_), "K3": float(K3_)
                }

                if local_best_rej is None or cand_["rej"] > local_best_rej["rej"]:
                    local_best_rej = cand_

                if local_best_pair is None or cand_["pair_pass"] > local_best_pair["pair_pass"]:
                    local_best_pair = cand_

                if cand_["rej"] >= REJ_TARGET_DB:
                    if (local_best_feasible is None or
                            cand_["pair_pass"] > local_best_feasible["pair_pass"]):
                        local_best_feasible = cand_

                if cand_["rej"] >= REJ_OPERATING_DB:
                    if (local_best_40 is None or
                            cand_["pair_pass"] > local_best_40["pair_pass"]):
                        local_best_40 = cand_

    best_rej_per_radius.append(local_best_rej)
    best_pair_per_radius.append(local_best_pair)
    best_feasible_per_radius.append(local_best_feasible)
    best_40_per_radius.append(local_best_40)

best_joint_rejection = max(best_rej_per_radius, key=lambda r: r["rej"])
best_joint_pair = max(best_pair_per_radius, key=lambda r: r["pair_pass"])

_feasible_joint = [r for r in best_feasible_per_radius if r is not None]
best_joint_feasible = (max(_feasible_joint, key=lambda r: r["pair_pass"])
                       if _feasible_joint else None)

_feasible_40_joint = [r for r in best_40_per_radius if r is not None]
best_joint_40 = (max(_feasible_40_joint, key=lambda r: r["pair_pass"])
                 if _feasible_40_joint else None)


def _print_joint_design(title, r):
    print(f"\n{title}")
    print(f"  target FSR = {r['fsr_target_GHz']:.2f} GHz")
    print(f"  actual FSR = {r['fsr_GHz']:.3f} GHz")
    print(f"  R1 = {r['R1_um']:.3f} um, R2 = {r['R2_um']:.3f} um")
    print(f"  K1 = {100*r['K1']:.6f} %")
    print(f"  K2 = {100*r['K2']:.6f} %")
    print(f"  K3 = {100*r['K3']:.6f} %")
    print(f"  rejection = {r['rej']:.2f} dB")
    print(f"  Ts = {r['Ts']:.5f}, Ti = {r['Ti']:.5f}, Ts*Ti = {r['pair_pass']:.5f}")
    print("  physical gaps:")
    for nm_, kval_ in (("g1", r["K1"]), ("g12", r["K2"]), ("g3", r["K3"])):
        gv_ = gap_for_K(kval_)
        print(f"    {nm_:3s} = {gv_:7.2f} nm" if np.isfinite(gv_)
              else f"    {nm_:3s} = outside simulated K(g)")


_print_joint_design("JOINT OBJECTIVE A -- maximum rejection of ONE double-ring",
                    best_joint_rejection)
_print_joint_design("JOINT OBJECTIVE B -- maximum Ts*Ti of ONE double-ring",
                    best_joint_pair)

print("\nJOINT OBJECTIVE C -- 100 dB feasibility with EXACTLY TWO RINGS")
if best_joint_feasible is None:
    print("  INFEASIBLE in the sampled radius/coupling domain.")
    print(f"  Maximum sampled rejection = {best_joint_rejection['rej']:.2f} dB")
    print("  -> radius optimization does not remove the two-ring limitation.")
else:
    _print_joint_design("  best feasible design", best_joint_feasible)

print(f"\nJOINT OPERATING POINT -- maximum Ts*Ti with rejection >= {REJ_OPERATING_DB:.0f} dB")
if best_joint_40 is None:
    print("  No sampled radius/coupling design satisfies the 40 dB constraint.")
else:
    _print_joint_design("  best constrained design", best_joint_40)

# Boundary diagnostics for the maximum-rejection solution.
fsr_boundary_ = (
    abs(best_joint_rejection["fsr_target_GHz"] - P["FSR_GHz"][0]) < 1e-9 or
    abs(best_joint_rejection["fsr_target_GHz"] - P["FSR_GHz"][2]) < 1e-9
)
K1_boundary_ = (
    abs(best_joint_rejection["K1"] - K13_MIN) < 1e-12 or
    abs(best_joint_rejection["K1"] - K13_MAX) < 1e-12
)
K2_boundary_ = (
    abs(best_joint_rejection["K2"] - K2_grid_3[0]) < 1e-12 or
    abs(best_joint_rejection["K2"] - K2_grid_3[-1]) < 1e-12
)
K3_boundary_ = (
    abs(best_joint_rejection["K3"] - K13_MIN) < 1e-12 or
    abs(best_joint_rejection["K3"] - K13_MAX) < 1e-12
)

if fsr_boundary_ or K1_boundary_ or K2_boundary_ or K3_boundary_:
    print("\n  WARNING: maximum-rejection solution lies on at least one search boundary.")
    print("  -> interpret it as a CONSTRAINED maximum within the investigated domain.")


# -------------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------------
fR, aR = plt.subplots(1, 3, figsize=(15, 4.5))

aR[0].plot([r["fsr_GHz"] for r in radius_sweep],
           [r["R1_um"] for r in radius_sweep], "o-", label="R1")
aR[0].plot([r["fsr_GHz"] for r in radius_sweep],
           [r["R2_um"] for r in radius_sweep], "s-", label="R2 = R1/2")
aR[0].scatter([best_joint_rejection["fsr_GHz"]],
              [best_joint_rejection["R1_um"]],
              marker="*", s=120, label="max-rejection R1")
aR[0].axvline(FSR_TEAM_R20_GHz, ls=":", label="teammate R1=20 um equivalent")
aR[0].set_xlabel("actual FSR (GHz)")
aR[0].set_ylabel("equivalent radius (um)")
aR[0].set_title("(a) radius from FSR + exact pump order")
aR[0].grid(alpha=.3)
aR[0].legend(fontsize=7)

aR[1].plot([r["R1_um"] for r in radius_sweep],
           [r["rej"] for r in radius_sweep], "o-", label="rejection")
aR[1].axhline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
aR[1].axvline(20.0, ls=":", label="teammate R1=20 um")
aR[1].set_xlabel("R1 (um)")
aR[1].set_ylabel("pump rejection (dB)")
aR[1].set_title("(b) radius-only; K fixed at max-rejection 3K point")
aR[1].grid(alpha=.3)
aRb = aR[1].twinx()
aRb.plot([r["R1_um"] for r in radius_sweep],
         [r["pair_pass"] for r in radius_sweep], "s--",
         label=r"$T_sT_i$")
aRb.set_ylabel(r"$T_sT_i$")
h1, l1_ = aR[1].get_legend_handles_labels()
h2, l2_ = aRb.get_legend_handles_labels()
aR[1].legend(h1+h2, l1_+l2_, fontsize=7, loc="best")

aR[2].plot([r["R1_um"] for r in best_rej_per_radius],
           [r["rej"] for r in best_rej_per_radius],
           "o-", label="max rejection at each radius")
aR[2].axhline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
aR[2].axhline(REJ_OPERATING_DB, ls=":", label="40 dB operating constraint")
if best_joint_40 is not None:
    aR[2].scatter([best_joint_40["R1_um"]], [best_joint_40["rej"]],
                  marker="D", s=70, label="max TsTi, rej >= 40 dB")
aR[2].axvline(20.0, ls=":", label="teammate R1=20 um")
aR[2].set_xlabel("R1 (um)")
aR[2].set_ylabel("max pump rejection (dB)")
aR[2].set_title("(c) joint radius + K1/K2/K3 search")
aR[2].grid(alpha=.3)
aRc = aR[2].twinx()
aRc.plot([r["R1_um"] for r in best_rej_per_radius],
         [r["pair_pass"] for r in best_rej_per_radius],
         "s--", label=r"$T_sT_i$ at max rejection")
_valid_40_radius = [r for r in best_40_per_radius if r is not None]
if _valid_40_radius:
    aRc.plot([r["R1_um"] for r in _valid_40_radius],
             [r["pair_pass"] for r in _valid_40_radius],
             "d-.", label=r"max $T_sT_i$ with rej >= 40 dB")
aRc.set_ylabel(r"$T_sT_i$")
h1, l1_ = aR[2].get_legend_handles_labels()
h2, l2_ = aRc.get_legend_handles_labels()
aR[2].legend(h1+h2, l1_+l2_, fontsize=7, loc="best")

fR.suptitle("[T4-OPTR] Exactly two rings: radius/coupling feasibility study")
fR.tight_layout()
fR.savefig("lean_fig_OPT_Radii.png", dpi=170)
plt.close(fR)


# =============================================================================
# FIGURES -- one per consigne
# =============================================================================
# IMPORTANT PLOTTING CHOICE:
#   the transmission/loss figures below are generated with the OPTIMIZED
#   two-ring design, not with the nominal Butterworth point.
#   We use the joint radius + K1/K2/K3 design that maximizes single-device
#   pump rejection within the investigated physical domain.
plot_design = best_joint_rejection
plot_geom = best_joint_rejection
plot_FSR_nm = plot_geom["FSR_nm"]
plot_lam_s = plot_geom["ls"]
plot_lam_i = plot_geom["li"]
plot_K1 = plot_design["K1"]
plot_K2 = plot_design["K2"]
plot_K3 = plot_design["K3"]
plot_A = typ["A_dB_per_cm"]
plot_KP = dict(Tp=plot_design["Tp"], Ts=plot_design["Ts"], Ti=plot_design["Ti"], rej=plot_design["rej"])
plot_Qo, plot_Qe = Q_factors(plot_geom["L1"], plot_A, plot_K1)
plot_QL = 1.0 / (1.0/plot_Qo + 1.0/plot_Qe)

print("\n[PLOT] Figures lean_fig2_T4-4 and lean_fig3_sweeps use the optimized")
print("       two-ring design (joint radius + K1/K2/K3 max-rejection point):")
print(f"       target/actual FSR = {plot_geom['fsr_target_GHz']:.2f} / {plot_geom['fsr_GHz']:.3f} GHz")
print(f"       R1 = {plot_geom['R1_um']:.3f} um, R2 = {plot_geom['R2_um']:.3f} um")
print(f"       K1 = {100*plot_K1:.6f} %, K2 = {100*plot_K2:.6f} %, K3 = {100*plot_K3:.6f} %")
print(f"       Tp = {plot_KP['Tp']:.3e}, Ts = {plot_KP['Ts']:.5f}, Ti = {plot_KP['Ti']:.5f}, rejection = {plot_KP['rej']:.2f} dB")

# The former T4-1 GVD/phase-matching figure is intentionally omitted here:
# those quantities are accepted from the upstream teammate stage rather than
# recalculated in this filter-optimization script.

# -------------------------------------------------------------------------
# Optimized spectrum / transmission figure
# -------------------------------------------------------------------------
span = 2.6 * plot_FSR_nm
lams = np.linspace(pump_nm - span, pump_nm + span, 20001)
out = np.array([
    series_rings_geom(l, plot_geom, plot_K1, plot_K2, plot_K3, plot_A)
    for l in lams
])

f2, a = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
a[0].plot(lams - pump_nm, abs(out[:, 2]) ** 2 / max((abs(out[:, 2]) ** 2).max(), 1e-300), lw=.8)
a[0].set_ylabel("ring 1\n(generator)")
a[0].set_title(f"(a) field buildup with optimized R1 = {plot_geom['R1_um']:.1f} um")

a[1].plot(lams - pump_nm, abs(out[:, 3]) ** 2 / max((abs(out[:, 3]) ** 2).max(), 1e-300),
          lw=.8, color="tab:orange")
a[1].set_ylabel("ring 2\n(filter)")
a[1].set_title(f"(b) field buildup with optimized R2 = {plot_geom['R2_um']:.1f} um")

a[2].semilogy(lams - pump_nm, abs(out[:, 1]) ** 2, color="tab:green", label=r"$T_{drop}$")
a[2].semilogy(lams - pump_nm, abs(out[:, 0]) ** 2, color="tab:red", label=r"$T_{through}$")
a[2].axhline(1e-10, color="gray", ls="--", label="-100 dB reference")
a[2].scatter([0.0], [plot_KP["Tp"]], marker="x", s=60, label="pump")
a[2].scatter([plot_lam_s - pump_nm], [plot_KP["Ts"]], marker="o", s=35, label="signal")
a[2].scatter([plot_lam_i - pump_nm], [plot_KP["Ti"]], marker="o", s=35, label="idler")
a[2].set_ylim(1e-12, 3)
a[2].legend(fontsize=8)
a[2].set_ylabel("transmission")
a[2].set_xlabel(r"$\lambda-\lambda_p$ (nm)")
a[2].set_title(
    f"(c) optimized spectrum: rej = {plot_KP['rej']:.2f} dB, "
    f"Ts = {plot_KP['Ts']:.3f}, Ti = {plot_KP['Ti']:.3f}"
)
for ax in a:
    for l_, col in ((pump_nm, "k"), (plot_lam_s, "r"), (plot_lam_i, "b")):
        ax.axvline(l_ - pump_nm, color=col, ls=":", lw=1)
    ax.grid(alpha=.3)
f2.suptitle("[T4-4] The two combs and T(omega) for the optimized two-ring design")
f2.tight_layout(); f2.savefig("lean_fig2_T4-4.png", dpi=170); plt.close(f2)

# -------------------------------------------------------------------------
# Sweeps referenced to the optimized design
# -------------------------------------------------------------------------
K1s = np.geomspace(K13_MIN, K13_MAX, 60)
As = np.linspace(P["A_dB_per_cm"][0], P["A_dB_per_cm"][2], 40)
rK = [kpis_radius(plot_geom, K1_, plot_K2, plot_K3, plot_A) for K1_ in K1s]
rA = [kpis_radius(plot_geom, plot_K1, plot_K2, plot_K3, A_) for A_ in As]

gg = np.linspace(gaps.min(), gaps.max(), 240)
g1_opt = gap_for_K(plot_K1)
g12_opt = gap_for_K(plot_K2)
g3_opt = gap_for_K(plot_K3)

f3, a = plt.subplots(1, 3, figsize=(15, 4.2))
a[0].semilogx(100*K1s, [r["rej"] for r in rK], "o-", ms=3, label="rejection")
a[0].axvline(100*plot_K1, color="k", ls=":", label="optimized K1")
a[0].set_xlabel("K1 (%) with K2,K3 fixed at optimum")
a[0].set_ylabel("pump rejection (dB)")
a[0].grid(alpha=.3)
a[0].legend(fontsize=7)
ab = a[0].twinx()
ab.semilogx(100*K1s, [r["Ts"] for r in rK], "s--", ms=3, color="tab:orange", label=r"$T_s$")
ab.set_ylabel(r"$T_s$", color="tab:orange")
a[0].set_title("(a) K1 cut around the optimized design")

a[1].plot(As, [r["rej"] for r in rA], "o-", ms=3, label="rejection")
a[1].axvline(plot_A, color="k", ls=":", label="selected loss")
a[1].set_xlabel("loss A (dB/cm)")
a[1].set_ylabel("pump rejection (dB)")
a[1].grid(alpha=.3)
ac = a[1].twinx()
ac.plot(As, [r["Ts"] for r in rA], "s--", ms=3, color="tab:orange", label=r"$T_s$")
ac.set_ylabel(r"$T_s$", color="tab:orange")
h1, l1_ = a[1].get_legend_handles_labels()
h2, l2_ = ac.get_legend_handles_labels()
a[1].legend(h1+h2, l1_+l2_, fontsize=7, loc="best")
a[1].set_title("(b) loss sweep at the optimized couplings")

a[2].semilogy(gaps, kzv, "o", label="FEMWELL supermodes")
a[2].semilogy(gg, [kappa_z_interp(x) for x in gg], "-", label="PCHIP between FEMWELL samples")
for g_, lbl_ in ((g1_opt, "g1 opt"), (g12_opt, "g12 opt"), (g3_opt, "g3 opt")):
    if np.isfinite(g_):
        a[2].axvline(g_, ls=":", label=f"{lbl_} = {g_:.1f} nm")
a[2].set_xlabel("gap (nm)")
a[2].set_ylabel(r"$\kappa_z$ (rad/um)")
a[2].legend(fontsize=7)
a[2].grid(alpha=.3, which="both")
a[2].set_title("(c) coupling-vs-gap with optimized physical gaps")
f3.suptitle("[T4-4/4b] Sweeps referenced to the optimized two-ring design")
f3.tight_layout(); f3.savefig("lean_fig3_sweeps.png", dpi=170); plt.close(f3)
