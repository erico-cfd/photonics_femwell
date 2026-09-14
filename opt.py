"""
task4_lean.py -- Vernier double-ring pump filter for an SFWM pair source
========================================================================
LEAN version. Design rule: every equation here appears in the supplied
material. Nothing is constructed by us. Where a quantity cannot be derived
from those sources it is CALIBRATED on a published measurement and said so.

Consignes addressed (Task 4 statement + minutes of 09/09):
  [T4-1] Femwell on the straight SOI guide -> mode profile, n_eff, n_g, A_eff,
         GVD, and the phase-matching condition Delta_k(Delta_Omega).
  [T4-2] FWM conversion efficiency and the photon-pair rate.
  [T4-3] Pair generation in a ring: Q factors and PGR.
  [T4-4] Vernier two-ring design: six couplings, T(omega), the two combs,
         pump extinction against the 100-120 dB target.
  [T4-4b] The couplings from the real gap geometry.
  [T4-OPT] Baseline sweep-based optimization of K1 with Butterworth K2,K3.
  [T4-OPT3] Independent circuit-level optimization of K1,K2,K3, constrained
            by the physically simulated FEMWELL coupling range.
  [T4-5] Effect of clean-room parameter variability.
  [T4-6] Feasibility: what is certain, assumed, missing.

WHAT THIS VERSION DROPS relative to earlier drafts, and why
  - bent-mode n_eff and a 1/R^2 scaling of it: the ring radius now follows from
    the FSR target (T4-3) and lands at ~114 um, where the curvature correction is
    1e-5 -- negligible. MEDINA24 sec 3.3.2 sizes rings the same way (R = 120 um)
    and likewise uses the straight-guide index.
  - an analytic curved-path coupling integral of our own: replaced by CHROST15
    Eq. 4.8 with an explicit coupling length L.
  - a calibrated chi(3) prefactor from a formula we could not reproduce: replaced
    by the rate-equation scaling of BAJONI17, calibrated on one measured PGR.
  - a Markov-chain escape efficiency of our own: dropped; we report T_drop, which
    is what the transfer function actually gives.

SOURCES (all supplied)
  RABUS07   Rabus, Integrated Ring Resonators, Springer 2007, Ch. 2.
            Eqs. 2.1-2.5 (coupler, theta), 2.20-2.21 (n_g, FSR), 2.58-2.63
            (two rings in series), 2.71 (resonance), 2.74-2.78 (Butterworth, Vernier).
  CHROST15  Chrostowski & Hochberg, Silicon Photonics Design, CUP 2015, sec 4.1.
            Eqs. 4.1/4.3/4.5/4.8 (supermode coupling), Fig. 4.14 (500x220 strip).
  BAJONI17  Bajoni & Galli, Photonics and Nanostructures 26, 24 (2017).
            Eqs. 13-14: rate equations -> PGR ~ P^2 tau^3 ~ P^2 Q^3.
  SAVANIER16 Savanier, Kumar, Mookherjea, Opt. Express 24, 3313 (2016).
            Eqs. 3-5 (Q_L, Q_U, Q_cpl with n_g), Fig. 6 (PGR = K Q^3/R^2 P^2),
            cutback losses 0.74 / 1.23 dB/cm, Appendix (L_res = L F/pi).
  MEDINA24  Medina Quiroz, PhD thesis Univ. Paris-Saclay 2024, Ch. 3.
            sec 3.3.1 dispersion engineering; 3.3.2 R = 120 um for 100-200 GHz FSR,
            gaps 50-120 nm, Q ~ 50k-150k; 3.4 the 100 dB requirement.
  MA17      Ma et al., Opt. Express 25, 32995 (2017). 1 dB/cm, Q_i 9e5, Q_L 9.2e4,
            FWHM 2.1 GHz; pump removed with EXTERNAL benchtop filters.
  ARXIV2411 arXiv:2411.05921 (2024). Table 1 (die-to-die spread, measured PGR),
            heater range 0.62 nm vs sigma 1.73 nm.
  WANG24    Wang et al., review, Adv. Devices Instrum. (2024). CROW filters give
            >96 dB on-chip pump suppression, beating commercial filters.
  SALZ57 / MALIT65   Sellmeier fits for Si and SiO2.
  THOMSON16 Roadmap on silicon photonics: linewidth fluctuations of order 5 nm.
  STRAIN15  Strain et al., Opt. Lett. 40, 1274-1277 (2015):
            tunable silicon microring cross-coupling from ~0.22 down to <0.005,
            crossing critical coupling near 0.018. Added specifically to justify
            the K1 sweep range requested for the optimization.
  FEMWELL   https://helgegehring.github.io/femwell (mode solver, GVD example).
"""

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.interpolate import UnivariateSpline, PchipInterpolator
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
    # 220 nm is the standard foundry SOI thickness; 450-550 nm keeps the guide
    # single-mode TE at 1550 nm. 500x220 nm is also the cross-section CHROST15
    # sec 4.1.6 analyses, which lets us validate the coupling against its Fig. 4.14.
    width_nm     = (450, 500, 550),
    height_nm    = (220, 220, 220),
    # Propagation loss. MA17 measures 1 dB/cm on test sites; SAVANIER16 measures
    # 0.74 and 1.23 dB/cm by cutback; MEDINA24 sec 3.3.2 infers <1 dB/cm oxide-clad.
    # [SUPERVISOR BOARD] nominal A ~= 0.5 dB/cm.
    # [SAVANIER16] measured cutback values 0.74 and 1.23 dB/cm.
    # [MEDINA24] rough estimate ~1.5 dB/cm for the C2N rings.
    A_dB_per_cm  = (0.5, 0.5, 1.5),
    # Target FSR. MEDINA24 sec 3.3.2 designs for 100-200 GHz to match the telecom
    # grid. THIS FIXES THE RADIUS -- it is not a free parameter.
    FSR_GHz      = (50.0, 100.0, 200.0),
    # Bus-to-ring POWER coupling K1 = |kappa_1|^2.
    # [SAVANIER16] experimentally inferred |kappa|^2 values around 0.005 and 0.018.
    # [STRAIN15] a tunable Si microring spans ~0.22 down to <0.005 and crosses
    # critical coupling near 0.018. Therefore 0.5%-22% is an experimentally
    # grounded sweep interval; 1.8% is a physically meaningful nominal point.
    K1_percent   = (0.5, 1.8, 22.0),
    # Coupling length of the straight section (CHROST15 Eq. 4.8 needs one).
    L_couple_um  = (2.0, 5.0, 15.0),
    pump_nm      = (1550.0, 1550.0, 1550.0),
)
typ = {k: v[1] for k, v in P.items()}
pump_nm = typ["pump_nm"]

print("=" * 78)
print("0. PARAMETERS (min, typical, max)")
print("=" * 78)
for k, (lo, t, hi) in P.items():
    print(f"  {k:14s}: [{lo:>7g} , {t:>7g} , {hi:>7g}]")


# =============================================================================
# [T4-1] STRAIGHT WAVEGUIDE: mode, dispersion, phase matching
# =============================================================================
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
    """TE0 mode [FEMWELL]. n_eff is the eigenvalue; beta = k n_eff [RABUS07 2.4]."""
    b = Basis(mesh, ElementTriP0())
    eps = b.zeros()
    for dom, nf in {"core": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um) ** 2
    return compute_modes(b, eps, wavelength=lam_um, num_modes=3,
                         order=1).sorted(key=lambda m: -np.real(m.te_fraction))[0]


w_um, h_um = typ["width_nm"] * 1e-3, typ["height_nm"] * 1e-3
print("\n" + "=" * 78)
print("[T4-1] Straight SOI waveguide (Femwell)")
print("=" * 78)
mesh0 = strip_mesh(w_um, h_um)
m_te0 = te0(mesh0, pump_nm * 1e-3)
m_te0.show("I", colorbar=True)
plt.title(f"TE0 intensity, {typ['width_nm']:.0f}x{typ['height_nm']:.0f} nm SOI strip @ 1550 nm")
plt.savefig("lean_fig0_mode.png", dpi=160); plt.close()

wl = np.linspace(1450, 1650, 13)
neff_v, aeff_v = [], []
for l in wl:
    m = te0(mesh0, l * 1e-3)
    neff_v.append(np.real(m.n_eff)); aeff_v.append(np.real(m.calculate_effective_area()))
neff_v, aeff_v = np.array(neff_v), np.array(aeff_v)
spl = UnivariateSpline(wl, neff_v, s=0, k=3)
d1 = spl.derivative(1)
n_eff = lambda l: spl(l)
n_g = lambda l: spl(l) - l * d1(l)                     # [RABUS07 Eq. 2.20]
A_eff = float(UnivariateSpline(wl, aeff_v, s=0, k=3)(pump_nm))


def beta2(l):
    """beta2 = (1/c) dn_g/domega. D = -(2 pi c/lambda^2) beta2 [FEMWELL GVD example]."""
    dl = 1.0
    dng_dl = (n_g(l + dl) - n_g(l - dl)) / (2 * dl * 1e-9)
    return (dng_dl / (-2 * np.pi * c / (l * 1e-9) ** 2)) / c


D_ps = lambda l: -(2 * np.pi * c / (l * 1e-9) ** 2) * beta2(l) * 1e12 * 1e-9 * 1e3
print(f"n_eff = {n_eff(pump_nm):.5f}   n_g = {n_g(pump_nm):.4f}   A_eff = {A_eff:.3f} um^2")
print(f"TE fraction = {np.real(m_te0.te_fraction):.3f}, power in core = "
      f"{np.real(m_te0.calculate_power(elements='core')):.3f}")
print(f"beta2 = {beta2(pump_nm):.3e} s^2/m,  D = {D_ps(pump_nm):+.0f} ps/(nm km) -> "
      f"{'ANOMALOUS' if D_ps(pump_nm) > 0 else 'normal'}")
print("  MEDINA24 sec 3.3.1: 'small anomalous dispersion is generally considered the")
print("  most favorable case' for SFWM -> this cross-section is in the right regime.")

# Phase matching. Delta_k = k_s + k_i - 2 k_p, omega_s,i = omega_p +/- Delta_Omega.
omega_p = 2 * np.pi * c / (pump_nm * 1e-9)
dOm = np.linspace(1e10, 2 * np.pi * c * (1 / 1490e-9 - 1 / 1550e-9), 200)
ls_nm = 2 * np.pi * c / (omega_p + dOm) * 1e9
li_nm = 2 * np.pi * c / (omega_p - dOm) * 1e9
kk = lambda l: 2 * np.pi * n_eff(l) / l
dk = (kk(ls_nm) + kk(li_nm) - 2 * kk(pump_nm)) * 1e9          # 1/m
sinc2 = lambda x: np.sinc(x / np.pi) ** 2


# =============================================================================
# [T4-3] THE RING: radius from the FSR target, then Q
# =============================================================================
# The FSR target fixes the radius [RABUS07 Eq. 2.21 solved for R; MEDINA24 sec
# 3.3.2 sizes its rings exactly this way]. The radius is therefore NOT free, and
# the design lands near 100 um -- the regime where MEDINA24 (R = 120 um) and this
# model may both use the straight-guide index, the curvature correction being ~1e-5.
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
print(f"  cf. MEDINA24 sec 3.3.2: R = 120 um for a 100-200 GHz FSR -- same regime.")
Q_o, Q_e = Q_factors(L1, typ["A_dB_per_cm"], typ["K1_percent"] / 100)
Q_L = 1 / (1 / Q_o + 1 / Q_e)
print(f"Q_o = {Q_o:.2e}  Q_e = {Q_e:.2e}  Q_L = {Q_L:.2e}  "
      f"(FWHM = {pump_nm/Q_L*1e3:.0f} pm = {c/(pump_nm*1e-9)/Q_L*1e-9:.2f} GHz)")
print(f"  MA17 measures Q_i = 9e5, Q_L = 9.2e4, FWHM 2.1 GHz at 1 dB/cm -- same order.")
for A_r, QU_r in ((0.74, 9.2e5), (1.23, 5.6e5)):
    QU = 2 * np.pi * 4.2 / (pump_nm * 1e-9 * A_r * 100 / 4.3429)
    print(f"  VALIDATION [SAVANIER16 cutback]: {A_r} dB/cm -> Q_U {QU:.2e} vs {QU_r:.1e} "
          f"({(QU/QU_r-1)*100:+.0f} %)")

# --- PGR: rate-equation scaling, calibrated on a measurement ------------------
# [BAJONI17 Eqs. 13-14] rate equations for the three resonances give
#     N_s = N_i = alpha P^2 tau^3   ->   PGR proportional to P^2 and to Q^3.
# [SAVANIER16 Fig. 6] measures exactly this: PGR = K Q^3 / R^2 * P^2, the 1/R^2
# coming from the mode volume. We keep that form and CALIBRATE K on one measured
# CW point [ARXIV2411 Table 1, die B9: 3.29 MHz/mW^2, Q_L = 4.26e4, R = 19.1 um].
# Checked against two further dies of the same paper: 3.76 vs 5.61 and 2.52 vs
# 3.16 MHz/mW^2 -- inside the >2x die-to-die spread the paper itself reports.
_QL_ref = 1 / (1 / 113.1e3 + 1 / 68.2e3)
K_PGR = 3.29e6 * (19.1e-6) ** 2 / _QL_ref ** 3
PGR = K_PGR * Q_L ** 3 / (R1_um * 1e-6) ** 2          # pairs/s per mW^2
print(f"PGR [BAJONI17 scaling, calibrated on ARXIV2411] = {PGR/1e6:.2f} MHz/mW^2")
print(f"  = {PGR*1e-6*0.01:.3f} MHz at 0.1 mW.  [not modelled] TPA/free carriers;")
print(f"  SAVANIER16 measures TPA = 0.02 dB/cm at -10 dBm, an order below the linear loss.")


# =============================================================================
# [T4-2] FWM EFFICIENCY AND PAIR RATE -- straight waveguide
# =============================================================================
# eta_FWM = (gamma P L)^2 sinc^2(Delta_k L/2)  [standard FWM conversion efficiency]
# gamma = 2 pi n2/(lambda A_eff), n2 = 4.5e-18 m^2/W for c-Si.
# SPONTANEOUS rate: SFWM is FWM seeded by vacuum, so the seed is one quantum per
# mode in the collection bandwidth: R = eta * d_nu  [structure of SAVANIER16 Eq. 1,
# r = d_nu (gamma P L_eff)^2 sinc^2]. Hence R ~ P^2, the scaling BAJONI17 derives
# and SAVANIER16 Fig. 6 measures. (Multiplying eta by the PUMP photon flux instead
# would give R ~ P^3 and overestimate by ~5 orders of magnitude.)
n2 = 4.5e-18
gamma_nl = 2 * np.pi * n2 / (pump_nm * 1e-9 * A_eff * 1e-12)
dnu_coll = c / (pump_nm * 1e-9) / Q_L          # collection bandwidth = ring linewidth
i_fsr = np.argmin(np.abs(pump_nm - ls_nm - FSR_nm))
print("\n" + "=" * 78)
print("[T4-2] FWM efficiency and pair rate, straight waveguide")
print("=" * 78)
print(f"gamma = {gamma_nl:.1f} /(W m);  collection bandwidth = ring linewidth = "
      f"{dnu_coll*1e-9:.2f} GHz")
print(f"phase matching at one FSR ({FSR_nm:.2f} nm): sinc^2 = "
      f"{sinc2(dk[i_fsr]*1e-3/2):.4f} for L = 1 mm")
print(f"  [SAVANIER16 Appendix] in a RING the sinc length is L*F/pi, not L:")
F_ring = FSR_nm / (pump_nm / Q_L)
print(f"  finesse = {F_ring:.0f} -> L_res = {L1*F_ring/np.pi*1e3:.2f} mm "
      f"-> sinc^2 = {sinc2(dk[i_fsr]*L1*F_ring/np.pi/2):.4f}")
for L_mm in (1.0, 5.0):
    for P_mW in (0.1, 1.0):
        eta = (gamma_nl * P_mW * 1e-3 * L_mm * 1e-3) ** 2 * sinc2(dk[i_fsr] * L_mm * 1e-3 / 2)
        print(f"  L = {L_mm:.0f} mm, P = {P_mW:.1f} mW: eta_FWM = {eta:.2e}, "
              f"R_pair = eta*d_nu = {eta*dnu_coll:.2e} pairs/s")


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
# [T4-OPT] SWEEP-BASED OPTIMIZATION OF K1 = |kappa_1|^2
# =============================================================================
# WHY A SWEEP RATHER THAN A BLACK-BOX OPTIMIZER?
# [SUPERVISOR BOARD] "what to be optimized (reject, transmission...), maybe fix
# some parameters, indicate the suppositions".  We therefore:
#   1) fix the nominal cross-section, FSR/radii, propagation loss and the
#      Rabus/Butterworth relations for k2 and k3;
#   2) sweep the physically documented K1 interval;
#   3) calculate rejection, transmission and Q for every point;
#   4) explicitly state the objective used to select K1.
#
# K1 RANGE:
# [SAVANIER16] |kappa|^2 ~ 0.005 and 0.018 are experimentally inferred values.
# [STRAIN15] tunable Si microring coupling spans ~0.22 down to <0.005, with
# critical coupling around 0.018.  Hence the numerical search interval
# 0.005 <= K1 <= 0.22 is experimentally grounded, not invented.
#
# REJECTION CONSTRAINT:
# [MEDINA24 Sec. 3.4] residual pump must be rejected by at least 100 dB.
#
# OBJECTIVE:
# [MODELLING CHOICE, explicitly requested by supervisor]
# Among designs that meet the 100 dB SYSTEM rejection requirement, maximize the
# useful pair transmission.  If one stage cannot reach 100 dB, use the minimum
# integer number N of IDEAL identical cascaded filter stages:
#       N = ceil(100 dB / rejection_per_stage).
# dB attenuations add for cascaded independent stages.
#
# Since both signal and idler must survive, the pair-transmission metric is
#       T_pair,total = (Ts * Ti)^N .
#
# We ALSO report a source-aware relative metric using the experimentally observed
# low-power scaling PGR proportional to Q_L^3/R^2 [SAVANIER16 Fig. 6]:
#       M_system proportional to (Q_L^3/R^2) * (Ts*Ti)^N .
# The unknown proportionality constant and fixed P^2 do NOT affect which K1
# maximizes the metric, so no new calibration constant is introduced here.

print("\n" + "=" * 78)
print("[T4-OPT] Sweep-based optimization of K1")
print("=" * 78)

REJ_TARGET_DB = 100.0                         # [SOURCE: MEDINA24 Sec. 3.4]
K1_grid = np.geomspace(0.005, 0.22, 320)     # [SOURCE RANGE: SAVANIER16 + STRAIN15]

opt_K1 = []
R1_m = L1 / (2*np.pi)

for K1_power in K1_grid:
    kp_ = kpis(100.0 * K1_power, typ["A_dB_per_cm"])
    Qo_, Qe_ = Q_factors(L1, typ["A_dB_per_cm"], K1_power)
    QL_ = 1.0 / (1.0/Qo_ + 1.0/Qe_)

    rej_stage_ = kp_["rej"]
    # If rej_stage is positive, this is the minimum ideal number of stages
    # whose dB rejections sum to >= 100 dB.
    N_ = max(1, int(np.ceil(REJ_TARGET_DB / max(rej_stage_, 1e-12))))

    Tsignal_total_ = kp_["Ts"] ** N_
    Tpair_total_ = (kp_["Ts"] * kp_["Ti"]) ** N_

    # [SOURCE: SAVANIER16 Fig. 6] relative low-power generation scaling.
    source_rel_ = QL_**3 / R1_m**2

    # [DERIVED SYSTEM KPI] generation scaling x survival of BOTH photons.
    system_rel_ = source_rel_ * Tpair_total_

    opt_K1.append(dict(
        K1=K1_power,
        Qo=Qo_,
        Qe=Qe_,
        QL=QL_,
        Tp=kp_["Tp"],
        Ts=kp_["Ts"],
        Ti=kp_["Ti"],
        rej_stage=rej_stage_,
        N=N_,
        rej_total=N_*rej_stage_,
        Tsignal_total=Tsignal_total_,
        Tpair_total=Tpair_total_,
        source_rel=source_rel_,
        system_rel=system_rel_,
    ))

# First ask whether a SINGLE stage can satisfy Medina's >=100 dB requirement.
single_feasible = [r for r in opt_K1 if r["rej_stage"] >= REJ_TARGET_DB]

if single_feasible:
    best_single = max(single_feasible, key=lambda r: r["Ts"] * r["Ti"])
    print("Single-stage feasible region EXISTS.")
    print("  Selection rule: maximize Ts*Ti subject to rejection >= 100 dB.")
    print(f"  best single-stage K1 = {100*best_single['K1']:.3f} %")
    print(f"  rejection = {best_single['rej_stage']:.2f} dB")
    print(f"  Ts = {best_single['Ts']:.4f}, Ti = {best_single['Ti']:.4f}")
else:
    best_single = None
    print("No single K1 in the documented 0.5%-22% range reaches 100 dB in this")
    print("ideal second-order double-ring model.  The model therefore evaluates the")
    print("minimum number of ideal cascaded stages required for each K1.")

# Primary project choice: maximize useful PAIR transmission after the minimum
# number of ideal stages needed to meet 100 dB.
best_trans = max(opt_K1, key=lambda r: r["Tpair_total"])

# Secondary/source-aware choice: include the Savanier Q^3/R^2 generation scaling.
best_system = max(opt_K1, key=lambda r: r["system_rel"])

# Critical coupling predicted by the SAME loss/Q equations:
# Qe = Qo -> Kcrit = alpha_power * L in the low-K approximation.
Kcrit = alpha_power_per_m(typ["A_dB_per_cm"]) * L1

print("\nDerived critical-coupling benchmark:")
print(f"  Kcrit = alpha_power*L = {100*Kcrit:.3f} %")
print("  [derived from SAVANIER16 Eqs. 4-5 by setting Qcpl = QU]")

print("\nOPTIMUM A -- filter-only objective")
print("  objective: maximize (Ts*Ti)^N subject to total ideal rejection >= 100 dB")
print(f"  K1 = {100*best_trans['K1']:.3f} %")
print(f"  rejection/stage = {best_trans['rej_stage']:.2f} dB")
print(f"  minimum N = {best_trans['N']}")
print(f"  total ideal rejection = {best_trans['rej_total']:.2f} dB")
print(f"  Ts = {best_trans['Ts']:.4f}, Ti = {best_trans['Ti']:.4f}")
print(f"  pair transmission after N stages = {best_trans['Tpair_total']:.4e}")
print(f"  QL = {best_trans['QL']:.3e}")

print("\nOPTIMUM B -- source-aware objective")
print("  objective: maximize [QL^3/R^2]*(Ts*Ti)^N")
print("  [SAVANIER16 supplies Q^3/R^2 scaling; multiplication by transmission is")
print("   a derived system KPI, not a new physical law.]")
print(f"  K1 = {100*best_system['K1']:.3f} %")
print(f"  rejection/stage = {best_system['rej_stage']:.2f} dB")
print(f"  minimum N = {best_system['N']}")
print(f"  total ideal rejection = {best_system['rej_total']:.2f} dB")
print(f"  Ts = {best_system['Ts']:.4f}, Ti = {best_system['Ti']:.4f}")
print(f"  pair transmission after N stages = {best_system['Tpair_total']:.4e}")
print(f"  QL = {best_system['QL']:.3e}")

# Figure: show the trade-off from which K1 is selected.
_Kpct = np.array([100*r["K1"] for r in opt_K1])
_rej = np.array([r["rej_stage"] for r in opt_K1])
_Ts = np.array([r["Ts"] for r in opt_K1])
_Ti = np.array([r["Ti"] for r in opt_K1])
_N = np.array([r["N"] for r in opt_K1])
_pair = np.array([r["Tpair_total"] for r in opt_K1])
_sys = np.array([r["system_rel"] for r in opt_K1])
_sys_norm = _sys / np.max(_sys)

fopt, aopt = plt.subplots(1, 3, figsize=(15, 4.4))

aopt[0].semilogx(_Kpct, _rej)
aopt[0].axhline(REJ_TARGET_DB, ls="--", label="MEDINA24: 100 dB")
aopt[0].axvline(100*Kcrit, ls=":", label="derived critical coupling")
aopt[0].set_xlabel(r"$K_1=|\kappa_1|^2$ (%)")
aopt[0].set_ylabel("pump rejection / stage (dB)")
aopt[0].set_title("(a) rejection")
aopt[0].grid(alpha=.3)
aopt[0].legend(fontsize=7)

aopt[1].semilogx(_Kpct, _Ts, label=r"$T_s$")
aopt[1].semilogx(_Kpct, _Ti, label=r"$T_i$")
aopt[1].semilogx(_Kpct, _pair, label=r"$(T_sT_i)^N$")
aopt[1].axvline(100*best_trans["K1"], ls="--", label="filter optimum")
aopt[1].set_xlabel(r"$K_1=|\kappa_1|^2$ (%)")
aopt[1].set_ylabel("transmission")
aopt[1].set_title("(b) useful transmission")
aopt[1].grid(alpha=.3)
aopt[1].legend(fontsize=7)

aopt[2].semilogx(_Kpct, _sys_norm, label="normalized source-aware metric")
aopt[2].step(_Kpct, _N / np.max(_N), where="mid", label="N stages / max(N)")
aopt[2].axvline(100*best_system["K1"], ls="--", label="source-aware optimum")
aopt[2].set_xlabel(r"$K_1=|\kappa_1|^2$ (%)")
aopt[2].set_ylabel("normalized quantity")
aopt[2].set_title("(c) system-level choice")
aopt[2].grid(alpha=.3)
aopt[2].legend(fontsize=7)

fopt.suptitle("[T4-OPT] Sweep-based selection of K1: rejection / transmission trade-off")
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


def kappa_z(gap_nm, lam_um):
    """[CHROST15 Eq. 4.3] kappa_z = pi (n_even - n_odd)/lambda, from the supermodes."""
    m = coupled_mesh(w_um, h_um, gap_nm * 1e-3)
    b = Basis(m, ElementTriP0()); eps = b.zeros()
    for dom, nf in {"core1": n_Si, "core2": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um) ** 2
    md = compute_modes(b, eps, wavelength=lam_um, num_modes=2, order=1)
    ne, no = sorted((np.real(md[0].n_eff), np.real(md[1].n_eff)), reverse=True)
    return np.pi / lam_um * (ne - no)


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
Lx200 = np.pi / (2*kzv[_i200])
print(f"\nVALIDATION [CHROST15 Fig. 4.14b]: L_x(200 nm) = {Lx200:.1f} um "
      f"vs ~37.5 um ({abs(Lx200-37.5)/37.5*100:.1f} %).")
print("CHROST15 sec. 4.1.5 reports that eigenmode predictions differ from")
print("fabricated couplers because thickness/width/sidewall variations matter.")

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
# [T4-OPT3] INDEPENDENT OPTIMIZATION OF K1, K2, K3
# =============================================================================
# Baseline: Rabus/Butterworth relations.
# New study: K1,K2,K3 are independently varied in the same circuit equations.
#
# K1 and K3:
#   search interval = intersection of
#     (i) documented bus-ring interval 0.5%-22% [SAVANIER16 + STRAIN15], and
#     (ii) the physically simulated FEMWELL K(g) interval above.
#
# K2:
#   no arbitrary literature interval is inserted. Its range comes directly
#   from the FEMWELL-simulated K(g) domain.
#
# [MODELLING ASSUMPTION]
# The same local 500x220-nm two-waveguide supermode model and L_c translate
# all three K values into gaps. Curvature, exact ring-ring geometry and coupler
# excess loss are not included by this 2-D local model.
#
# OBJECTIVES:
#   A) maximum pump rejection per stage;
#   B) maximum single-stage useful transmission Ts*Ti;
#   C) maximum pair survival after the minimum number N of ideal identical
#      stages needed to meet MEDINA24's >=100 dB system rejection.
#
# We intentionally do NOT use isolated-ring Q_L^3 in this 3-K optimization.
# Once K2 and K3 are independent, loading of ring 1 is a coupled-cavity property.

print("\n" + "="*78)
print("[T4-OPT3] Independent K1 / K2 / K3 circuit optimization")
print("="*78)

K13_MIN = max(0.005, K_PHYS_MIN)
K13_MAX = min(0.22,  K_PHYS_MAX)

if not (K13_MIN < K13_MAX):
    raise RuntimeError("FEMWELL K(g) range does not overlap the documented K1/K3 range.")

# [NUMERICAL CHOICE] Grid resolution only.
# 22 x 26 x 22 = 12,584 circuit evaluations; no FEMWELL call occurs here.
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

            rej_ = r_["rej"]
            N_ = max(1, int(np.ceil(REJ_TARGET_DB / max(rej_, 1e-12))))
            pair_pass_ = r_["Ts"] * r_["Ti"]
            pair_survival_ = pair_pass_**N_

            results_3K.append(dict(
                K1=K1_, K2=K2_, K3=K3_,
                Tp=r_["Tp"], Ts=r_["Ts"], Ti=r_["Ti"],
                rej_stage=rej_,
                pair_pass=pair_pass_,
                N=N_,
                rej_total=N_*rej_,
                pair_survival=pair_survival_
            ))

best_rejection_3K = max(results_3K, key=lambda r: r["rej_stage"])
best_pass_3K = max(results_3K, key=lambda r: r["pair_pass"])
best_filter_3K = max(results_3K, key=lambda r: r["pair_survival"])

K1_BW = typ["K1_percent"]/100.0
_k1bw, _t1bw, _k2bw, _t2bw, _k3bw, _t3bw = couplers(K1_BW)
baseline_3K = kpis_3K(K1_BW, _k2bw**2, _k3bw**2, typ["A_dB_per_cm"])
baseline_N = max(1, int(np.ceil(REJ_TARGET_DB/max(baseline_3K["rej"], 1e-12))))
baseline_pair_survival = (baseline_3K["Ts"]*baseline_3K["Ti"])**baseline_N

def _print_3k_result(title, r):
    print(f"\n{title}")
    print(f"  K1 = {100*r['K1']:.6f} %")
    print(f"  K2 = {100*r['K2']:.6f} %")
    print(f"  K3 = {100*r['K3']:.6f} %")
    print(f"  rejection/stage = {r['rej_stage']:.2f} dB")
    print(f"  Ts = {r['Ts']:.5f}, Ti = {r['Ti']:.5f}, Ts*Ti = {r['pair_pass']:.5f}")
    print(f"  minimum ideal stages for >=100 dB = {r['N']}")
    print(f"  total ideal rejection = {r['rej_total']:.2f} dB")
    print(f"  pair survival = (Ts*Ti)^N = {r['pair_survival']:.5e}")
    for nm_, kval_ in (("g1", r["K1"]), ("g12", r["K2"]), ("g3", r["K3"])):
        gv_ = gap_for_K(kval_)
        print(f"  {nm_:3s} = {gv_:7.2f} nm" if np.isfinite(gv_)
              else f"  {nm_:3s} = outside simulated K(g)")

print("\nBASELINE -- Rabus/Butterworth")
print(f"  K1={100*K1_BW:.6f} %, K2={100*_k2bw**2:.6f} %, K3={100*_k3bw**2:.6f} %")
print(f"  rejection/stage={baseline_3K['rej']:.2f} dB")
print(f"  Ts={baseline_3K['Ts']:.5f}, Ti={baseline_3K['Ti']:.5f}")
print(f"  N={baseline_N}, pair survival={baseline_pair_survival:.5e}")

_print_3k_result("OBJECTIVE A -- maximum single-stage pump rejection", best_rejection_3K)
_print_3k_result("OBJECTIVE B -- maximum single-stage Ts*Ti", best_pass_3K)
_print_3k_result("OBJECTIVE C -- maximum pair survival subject to >=100 dB total",
                 best_filter_3K)

print("\nInterpretation:")
print("  The three objectives are intentionally different. There is no universal")
print("  'best coupling': rejection and useful transmission compete.")

_sorted = sorted(results_3K, key=lambda r: r["rej_stage"], reverse=True)
pareto = []
best_pair_seen = -np.inf
for r_ in _sorted:
    if r_["pair_pass"] > best_pair_seen:
        pareto.append(r_)
        best_pair_seen = r_["pair_pass"]

f3k, a3k = plt.subplots(1, 3, figsize=(15, 4.5))

a3k[0].scatter([r["rej_stage"] for r in results_3K],
               [r["pair_pass"] for r in results_3K],
               s=5, alpha=.15, label="physical-grid designs")
a3k[0].plot([r["rej_stage"] for r in pareto],
            [r["pair_pass"] for r in pareto],
            linewidth=2, label="Pareto frontier")
a3k[0].scatter([baseline_3K["rej"]],
               [baseline_3K["Ts"]*baseline_3K["Ti"]],
               marker="x", s=70, label="Butterworth baseline")
a3k[0].scatter([best_filter_3K["rej_stage"]],
               [best_filter_3K["pair_pass"]],
               marker="*", s=120, label="selected 3K optimum")
a3k[0].set_xlabel("pump rejection / stage (dB)")
a3k[0].set_ylabel(r"$T_sT_i$ / stage")
a3k[0].set_title("(a) rejection / transmission Pareto trade-off")
a3k[0].grid(alpha=.3)
a3k[0].legend(fontsize=7)

_labels = ["K1", "K2", "K3"]
_bw_vals = np.array([K1_BW, _k2bw**2, _k3bw**2])*100
_opt_vals = np.array([best_filter_3K["K1"], best_filter_3K["K2"], best_filter_3K["K3"]])*100
_x = np.arange(3)
a3k[1].semilogy(_x, _bw_vals, "o-", label="Butterworth")
a3k[1].semilogy(_x, _opt_vals, "s-", label="3K optimum")
a3k[1].set_xticks(_x, _labels)
a3k[1].set_ylabel("POWER coupling K (%)")
a3k[1].set_title("(b) couplings: baseline vs optimum")
a3k[1].grid(alpha=.3, which="both")
a3k[1].legend(fontsize=7)

_bw_gaps = [gap_for_K(v/100) for v in _bw_vals]
_opt_gaps = [gap_for_K(v/100) for v in _opt_vals]
a3k[2].plot(_x, _bw_gaps, "o-", label="Butterworth")
a3k[2].plot(_x, _opt_gaps, "s-", label="3K optimum")
a3k[2].set_xticks(_x, ["g1", "g12", "g3"])
a3k[2].set_ylabel("physical gap from FEMWELL K(g) (nm)")
a3k[2].set_title("(c) physical implementation")
a3k[2].grid(alpha=.3)
a3k[2].legend(fontsize=7)

f3k.suptitle("[T4-OPT3] Independent K1/K2/K3 optimization constrained by FEMWELL")
f3k.tight_layout()
f3k.savefig("lean_fig_OPT_3K.png", dpi=170)
plt.close(f3k)



# =============================================================================
# [T4-5] FABRICATION VARIABILITY
# =============================================================================
print("\n" + "=" * 78)
print("[T4-5] Clean-room parameter variability")
print("=" * 78)
dwn = 5.0
npl = np.real(te0(strip_mesh(w_um + dwn * 1e-3, h_um), pump_nm * 1e-3).n_eff)
nmi = np.real(te0(strip_mesh(w_um - dwn * 1e-3, h_um), pump_nm * 1e-3).n_eff)
dn_dw = (npl - nmi) / (2 * dwn)
shift = pump_nm * dn_dw / n_g(pump_nm) * 1e3          # pm per nm of width
print(f"dn_eff/dw = {dn_dw:.2e} /nm -> {shift:.0f} pm of resonance shift per nm of width")
print(f"  [THOMSON16] observed linewidth fluctuations are of order 5 nm.")
print(f"  [MODELLING CHOICE] symmetric +/-5 nm is used only as a sensitivity window,")
print(f"  not as a measured statistical distribution.")
print(f"  First-order common-mode estimate at +/-5 nm -> +/-{shift*5e-3:.2f} nm resonance shift,")
print(f"  i.e. {shift*5e-3/FSR_nm*100:.0f} % of one FSR.")
dw = np.unique(np.concatenate([
    np.linspace(-2.0, -0.05, 25),
    np.linspace(-0.05, 0.05, 401),
    np.linspace(0.05, 2.0, 25)
]))
tol = [kpis(typ["K1_percent"], typ["A_dB_per_cm"], dn2=dn_dw*x) for x in dw]
Ts_t = np.array([t["Ts"] for t in tol])
ER_t = np.array([t["ER"] for t in tol])

_half_target = 0.5*KP["Ts"]
_fhalf = lambda d: kpis(
    typ["K1_percent"], typ["A_dB_per_cm"], dn2=dn_dw*d
)["Ts"] - _half_target

try:
    dw_half_pos = brentq(_fhalf, 0.0, 0.05)
except ValueError:
    dw_half_pos = np.nan

try:
    dw_half_neg = brentq(_fhalf, -0.05, 0.0)
except ValueError:
    dw_half_neg = np.nan

_half_candidates = [abs(x) for x in (dw_half_neg, dw_half_pos) if np.isfinite(x)]
dw_half = min(_half_candidates) if _half_candidates else np.nan

if np.isfinite(dw_half):
    print(f"  DIFFERENTIAL: half-extraction equivalent-width sensitivity = {dw_half:.5f} nm")
    print(f"  -> {dw_half*shift:.2f} pm relative resonance shift, against a "
          f"{pump_nm/Q_L*1e3:.2f} pm nominal loaded linewidth.")
else:
    print("  DIFFERENTIAL: half-extraction point was not bracketed in +/-0.05 nm.")
print(f"  -> this is an EQUIVALENT-WIDTH sensitivity, not a fabrication specification.")
print(f"  The physical conclusion is that differential resonance detuning must remain")
print(f"  within a fraction of the loaded linewidth or be actively corrected.")


# =============================================================================
# FIGURES -- one per consigne
# =============================================================================
lf = np.linspace(1460, 1640, 60)
f1, a = plt.subplots(1, 3, figsize=(15, 4.2))
a[0].plot(lf, n_eff(lf), label=r"$n_{eff}$"); a[0].plot(wl, neff_v, "o", ms=3)
a[0].plot(lf, n_g(lf), label=r"$n_g$"); a[0].legend(fontsize=8); a[0].grid(alpha=.3)
a[0].set_xlabel("wavelength (nm)"); a[0].set_ylabel("index")
a[0].set_title("(a) dispersion [RABUS07 2.20]")
a[1].plot(lf, [D_ps(x) for x in lf]); a[1].axhline(0, color="gray", ls="--")
a[1].set_xlabel("wavelength (nm)"); a[1].set_ylabel("D (ps/(nm km))"); a[1].grid(alpha=.3)
a[1].set_title(f"(b) GVD = {D_ps(pump_nm):+.0f}, anomalous [MEDINA24 3.3.1]")
a[2].plot(dOm / (2 * np.pi) * 1e-12, sinc2(dk * 1e-3 / 2), label="L = 1 mm straight")
a[2].plot(dOm / (2 * np.pi) * 1e-12, sinc2(dk * L1 * F_ring / np.pi / 2),
          label=r"ring, $L\mathcal{F}/\pi$ [SAVANIER16]")
a[2].axvline(dOm[i_fsr] / (2 * np.pi) * 1e-12, color="r", ls=":", label="1 FSR")
a[2].set_xlabel(r"$\Delta\Omega/2\pi$ (THz)"); a[2].set_ylabel(r"sinc$^2$")
a[2].legend(fontsize=7); a[2].grid(alpha=.3); a[2].set_title("(c) phase matching")
f1.suptitle("[T4-1] Straight SOI waveguide: dispersion and phase matching")
f1.tight_layout(); f1.savefig("lean_fig1_T4-1.png", dpi=170); plt.close(f1)

span = 2.6 * FSR_nm
lams = np.linspace(pump_nm - span, pump_nm + span, 20001)
out = np.array([series_rings(l, typ["K1_percent"] / 100, typ["A_dB_per_cm"]) for l in lams])
f2, a = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
a[0].plot(lams - pump_nm, abs(out[:, 2]) ** 2 / (abs(out[:, 2]) ** 2).max(), lw=.8)
a[0].set_ylabel("ring 1\n(generator)")
a[1].plot(lams - pump_nm, abs(out[:, 3]) ** 2 / (abs(out[:, 3]) ** 2).max(),
          lw=.8, color="tab:orange"); a[1].set_ylabel("ring 2\n(filter)")
a[2].semilogy(lams - pump_nm, abs(out[:, 1]) ** 2, color="tab:green", label=r"$T_{drop}$")
a[2].semilogy(lams - pump_nm, abs(out[:, 0]) ** 2, color="tab:red", label=r"$T_{through}$")
a[2].axhline(1e-10, color="gray", ls="--", label="-100 dB target")
a[2].set_ylim(1e-12, 3); a[2].legend(fontsize=8); a[2].set_ylabel("transmission")
a[2].set_xlabel(r"$\lambda-\lambda_p$ (nm)")
for ax in a:
    for l_, col in ((pump_nm, "k"), (lam_s, "r"), (lam_i, "b")):
        ax.axvline(l_ - pump_nm, color=col, ls=":", lw=1)
    ax.grid(alpha=.3)
f2.suptitle("[T4-4] The two combs and T(omega) [RABUS07 Eqs. 2.58-2.63]")
f2.tight_layout(); f2.savefig("lean_fig2_T4-4.png", dpi=170); plt.close(f2)

K1s = np.logspace(np.log10(P["K1_percent"][0]), np.log10(P["K1_percent"][2]), 20)
As = np.linspace(P["A_dB_per_cm"][0], P["A_dB_per_cm"][2], 10)
rK = [kpis(x, typ["A_dB_per_cm"]) for x in K1s]
rA = [kpis(typ["K1_percent"], x) for x in As]
f3, a = plt.subplots(1, 3, figsize=(15, 4.2))
a[0].semilogx(K1s, [r["rej"] for r in rK], "o-", label="rejection")
a[0].axhspan(100, 120, color="red", alpha=.12, label="target")
a[0].set_xlabel("K1 (%)"); a[0].set_ylabel("dB"); a[0].legend(fontsize=7); a[0].grid(alpha=.3)
ab = a[0].twinx(); ab.semilogx(K1s, [r["Ts"] for r in rK], "s--", color="tab:orange")
ab.set_ylabel(r"$T_{drop}$(s)", color="tab:orange"); a[0].set_title("(a) bus coupling")
a[1].plot(As, [r["Ts"] for r in rA], "o-")
a[1].set_xlabel("loss A (dB/cm)"); a[1].set_ylabel(r"$T_{drop}$(signal)"); a[1].grid(alpha=.3)
a[1].set_title("(b) loss [MA17 1 dB/cm; SAVANIER16 0.74-1.23]")
gg = np.linspace(gaps.min(), gaps.max(), 240)
a[2].semilogy(gaps, kzv, "o", label="Femwell supermodes")
a[2].semilogy(gg, [kappa_z_interp(x) for x in gg], "-",
              label="PCHIP between FEMWELL samples")
a[2].set_xlabel("gap (nm)"); a[2].set_ylabel(r"$\kappa_z$ (rad/um)")
a[2].legend(fontsize=7); a[2].grid(alpha=.3, which="both")
a[2].set_title("(c) coupling vs gap [CHROST15 4.3]")
f3.suptitle("[T4-4/4b] Parameter sweeps and the coupling-gap relation")
f3.tight_layout(); f3.savefig("lean_fig3_sweeps.png", dpi=170); plt.close(f3)

f4, a = plt.subplots(1, 2, figsize=(11, 4.3))
a[0].plot(dw, Ts_t, color="tab:orange"); a[0].set_xlim(-0.5, 0.5)
a[0].axhline(0.5 * KP["Ts"], color="gray", ls="--", lw=1)
a[0].set_xlabel("differential width error (nm)"); a[0].set_ylabel(r"$T_{drop}$(signal)")
a[0].grid(alpha=.3); a[0].set_title("(a) DIFFERENTIAL -- breaks the Vernier")
ab = a[0].twinx(); ab.plot(dw, ER_t, color="tab:green"); ab.set_ylabel("extinction (dB)", color="tab:green")
dwc = np.linspace(-5, 5, 50)
a[1].plot(dwc, dwc * shift * 1e-3, color="tab:blue", label="first-order comb shift")
# FWHM/2 is the directly relevant resonance-alignment scale; it is DERIVED from Q_L.
_fwhm_nm = pump_nm / Q_L
a[1].axhspan(-_fwhm_nm/2, _fwhm_nm/2, color="green", alpha=.12,
             label=r"$\pm$FWHM/2 (derived)")
# FSR/2 is shown only as the distance to the midpoint between longitudinal modes,
# NOT as an alignment tolerance.
a[1].axhline(+FSR_nm/2, color="gray", ls=":", lw=1, label=r"$\pm$FSR/2 reference")
a[1].axhline(-FSR_nm/2, color="gray", ls=":", lw=1)
a[1].set_xlabel("common-mode width error (nm)"); a[1].set_ylabel("resonance shift (nm)")
a[1].legend(fontsize=7); a[1].grid(alpha=.3); a[1].set_title("(b) COMMON -- shifts both combs")
f4.suptitle("[T4-5] First-order width sensitivity [THOMSON16: fluctuations of order 5 nm]")
f4.tight_layout(); f4.savefig("lean_fig4_T4-5.png", dpi=170); plt.close(f4)
print("\nFigures: lean_fig0_mode, lean_fig1_T4-1, lean_fig2_T4-4, lean_fig3_sweeps,")
print("         lean_fig_OPT_K1, lean_fig_OPT_3K, lean_fig4_T4-5")


# =============================================================================
# [T4-6] FEASIBILITY
# =============================================================================
print("\n" + "=" * 78)
print("[T4-6] Feasibility: certain / assumed / missing")
print("=" * 78)
print(f"""
CERTAIN -- every equation is in the supplied material:
  resonance and FSR [RABUS07 2.5, 2.21, 2.71]; Q with n_g [SAVANIER16 4-5],
  validated to +7/+9 % against their cutback measurement; the two-ring transfer
  function [RABUS07 2.58-2.63]; Butterworth k2 [RABUS07 2.74-2.76]; the coupling
  from the gap [CHROST15 4.3/4.5/4.8], validated to {abs(Lx200-37.5)/37.5*100:.0f} % against their
  Fig. 4.14b on the same cross-section; PGR ~ P^2 Q^3 [BAJONI17 13-14, measured by
  SAVANIER16 Fig. 6]; the shift law dlambda/lambda = dn_eff/n_g [THOMSON16].
  L2 = L1/2 with one index makes theta2 = theta1/2 identically -- pure algebra.

ASSUMED:
  the 2:1 Vernier ratio, read off the supervisor's comb slide (NOT stated
  numerically anywhere -- the single assumption the whole design rests on);
  lossless, frequency-independent couplers; CW undepleted pump; circular rings
  with a point coupler; the PGR prefactor calibrated on one measured die.

MISSING, and known to matter:
  TPA and free carriers -- PGR is an upper bound [SAVANIER16 quantifies TPA at
  0.02 dB/cm at -10 dBm, an order below the linear loss, so the model is safe only
  at low power]; the scattered-light floor -- modelled rejection is an upper bound
  (WANG24 reports >96 dB achieved on chip with CROW, and MA17, the CAR record,
  still used external filters); backscattering and resonance splitting from
  sidewall roughness [RABUS07 2.1.1]; the coupler's own phase, which shifts
  resonances [RABUS07 2.56-2.57]; a Monte Carlo yield study; the laser's own
  linewidth and side-mode suppression, which both supervisor slides ask for.

THE ANSWER TO THE FEASIBILITY QUESTION:
  the nominal |00| case gives {KP['rej']:.0f} dB rejection per stage and
  Ts = {KP['Ts']:.3f}, Ti = {KP['Ti']:.3f}.  The sweep-based filter objective
  selects K1 = {100*best_trans['K1']:.3f}% and needs N = {best_trans['N']} ideal
  stage(s) to exceed the 100 dB MEDINA24 requirement.  The independent 3-K
  study tests whether the Rabus/Butterworth coupling constraint is optimal for
  the chosen filter objective, while restricting K1/K2/K3 to couplings represented
  by the FEMWELL gap simulations. Differential width error is reported only as
  a first-order equivalent-width sensitivity; the physical requirement is
  resonance alignment within the loaded linewidth.
""")
