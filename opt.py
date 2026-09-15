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
  [T4-OPT] Baseline SINGLE-DEVICE sweep of K1 with Butterworth K2,K3.
  [T4-OPT3] Independent SINGLE-DEVICE optimization of K1,K2,K3, constrained
            by the physically simulated FEMWELL coupling range.
  [T4-OPTR] Radius optimization under the 2:1 Vernier constraint:
            R2 = R1/2, with the documented 100-200 GHz FSR design interval.
            Exactly TWO rings are retained throughout; cascaded double-ring
            stages are NOT used to satisfy the 100 dB target.
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
    # [SOURCE: UPSTREAM TEAM GVD-DESIGN STAGE]
    # These are FIXED inputs inherited from the preceding task, not optimized here.
    # That stage selected w=400 nm, h=220 nm and lambda_p=1580 nm for subsequent
    # circuit simulations.  Task 4 therefore accepts them as architectural inputs.
    width_nm     = (400, 400, 400),
    height_nm    = (220, 220, 220),
    # Propagation loss. MA17 measures 1 dB/cm on test sites; SAVANIER16 measures
    # 0.74 and 1.23 dB/cm by cutback; MEDINA24 sec 3.3.2 infers <1 dB/cm oxide-clad.
    # [SUPERVISOR BOARD] nominal A ~= 0.5 dB/cm.
    # [SAVANIER16] measured cutback values 0.74 and 1.23 dB/cm.
    # [MEDINA24] rough estimate ~1.5 dB/cm for the C2N rings.
    A_dB_per_cm  = (0.5, 0.5, 1.5),
    # Target FSR / radius-design interval.
    # [MEDINA24 sec 3.3.2] uses the 100-200 GHz regime.  The nominal case is
    # 100 GHz; [T4-OPTR] sweeps only this documented interval rather than
    # inventing an independent radius range.
    FSR_GHz      = (100.0, 100.0, 200.0),
    # Bus-to-ring POWER coupling K1 = |kappa_1|^2.
    # [SAVANIER16] experimentally inferred |kappa|^2 values around 0.005 and 0.018.
    # [STRAIN15] a tunable Si microring spans ~0.22 down to <0.005 and crosses
    # critical coupling near 0.018. Therefore 0.5%-22% is an experimentally
    # grounded sweep interval; 1.8% is a physically meaningful nominal point.
    K1_percent   = (0.5, 1.8, 22.0),
    # Coupling length of the straight section (CHROST15 Eq. 4.8 needs one).
    L_couple_um  = (2.0, 5.0, 15.0),
    # [SOURCE: UPSTREAM TEAM GVD-DESIGN STAGE] fixed pump wavelength.
    pump_nm      = (1580.0, 1580.0, 1580.0),
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
plt.title(f"TE0 intensity, {typ['width_nm']:.0f}x{typ['height_nm']:.0f} nm SOI strip @ {pump_nm:.0f} nm")
plt.savefig("lean_fig0_mode.png", dpi=160); plt.close()

# -------------------------------------------------------------------------
# GVD -- STRICTLY FOLLOW THE OFFICIAL FEMWELL "Calculate GVD of waveguide"
# EXAMPLE:
# https://helgegehring.github.io/femwell/photonics/examples/calculate_GVD.html
#
# The official example:
#   1) sweeps wavelength on ONE fixed mesh;
#   2) selects the mode with the largest TE fraction;
#   3) fits n_eff(lambda_nm) with UnivariateSpline(s=0, k=3);
#   4) differentiates THAT spline twice directly;
#   5) evaluates
#          D = -lambda/c * d^2 n_eff / d lambda^2
#      with lambda in nm and c = 2.99792e-7 km/ps,
#      giving D directly in ps/(nm km).
#
# We intentionally do NOT calculate D through a finite difference of n_g here.
# The geometry/materials are ours; the numerical GVD procedure is FEMWELL's.
# -------------------------------------------------------------------------
wavelength_range = [1500.0, 1600.0]  # nm: spectral window of our GVD study
wavelength_points = 50               # same number of samples as FEMWELL example
wl = np.linspace(wavelength_range[0], wavelength_range[1], wavelength_points)

neff_v, aeff_v = [], []
for l in wl:
    m = te0(mesh0, l * 1e-3)
    neff_v.append(np.real(m.n_eff))
    aeff_v.append(np.real(m.calculate_effective_area()))

neff_v = np.asarray(neff_v)
aeff_v = np.asarray(aeff_v)

# Exact FEMWELL-example spline construction.
spl = UnivariateSpline(wl, neff_v, s=0, k=3, ext=2)
spl_d1 = spl.derivative(n=1)
spl_d2 = spl.derivative(n=2)

n_eff = lambda l: spl(l)

# Group index is still needed later for FSR/Q. This follows directly from
# n_g = n_eff - lambda dn_eff/dlambda, using the SAME spline.
n_g = lambda l: spl(l) - l * spl_d1(l)

A_eff = float(UnivariateSpline(wl, aeff_v, s=0, k=3)(pump_nm))

# EXACT unit convention used by the official FEMWELL example:
# c = 2.99792e-7 km/ps, wavelength is in nm, spline second derivative is 1/nm^2.
c_km_per_ps_femwell = 2.99792e-7

def D_ps(l_nm):
    """GVD D [ps/(nm km)] -- direct FEMWELL-example formula."""
    return -l_nm / c_km_per_ps_femwell * spl_d2(l_nm)

def beta2(l_nm):
    """beta2 [s^2/m], derived AFTER D; GVD itself is computed only by FEMWELL's method."""
    lam_m = l_nm * 1e-9
    D_SI = D_ps(l_nm) * 1e-6  # 1 ps/(nm km) = 1e-6 s/m^2
    return -D_SI * lam_m**2 / (2*np.pi*c)

print(f"n_eff = {n_eff(pump_nm):.5f}   n_g = {n_g(pump_nm):.4f}   "
      f"A_eff = {A_eff:.3f} um^2")
print(f"TE fraction = {np.real(m_te0.te_fraction):.3f}, power in core = "
      f"{np.real(m_te0.calculate_power(elements='core')):.3f}")
print(f"beta2 = {beta2(pump_nm):.3e} s^2/m,  "
      f"D = {D_ps(pump_nm):+.3f} ps/(nm km) -> "
      f"{'ANOMALOUS' if D_ps(pump_nm) > 0 else 'normal'}")

if D_ps(pump_nm) > 0 and abs(D_ps(pump_nm)) < 100:
    print("  This point is in a small-anomalous-D regime.")
elif D_ps(pump_nm) > 0:
    print("  D is anomalous, but not close to zero.")
else:
    print("  D is normal; this point is NOT in the small-anomalous-D regime.")

# Phase matching. Delta_k = k_s + k_i - 2 k_p,
# omega_s,i = omega_p +/- Delta_Omega.
#
# IMPORTANT: n_eff(lambda) is valid ONLY on the FEMWELL-simulated spline domain.
# The previous script inherited a hard-coded 1490/1550-nm detuning limit and
# silently extrapolated the cubic spline for much of the idler branch.
# Here BOTH signal and idler are forced to remain inside [wl.min(), wl.max()].
omega_p = 2 * np.pi * c / (pump_nm * 1e-9)

lam_min_m = wl.min() * 1e-9
lam_max_m = wl.max() * 1e-9
lam_p_m = pump_nm * 1e-9

dOm_max_signal = 2*np.pi*c * (1/lam_min_m - 1/lam_p_m)
dOm_max_idler  = 2*np.pi*c * (1/lam_p_m - 1/lam_max_m)
dOm_max = min(dOm_max_signal, dOm_max_idler)

# 0.98 leaves a small numerical margin from the spline boundary.
dOm = np.linspace(1e10, 0.98*dOm_max, 200)
ls_nm = 2 * np.pi * c / (omega_p + dOm) * 1e9
li_nm = 2 * np.pi * c / (omega_p - dOm) * 1e9

# Hard guards: phase matching may never use extrapolated n_eff values.
assert np.min(ls_nm) >= wl.min() - 1e-9
assert np.max(ls_nm) <= wl.max() + 1e-9
assert np.min(li_nm) >= wl.min() - 1e-9
assert np.max(li_nm) <= wl.max() + 1e-9

print(f"  phase-matching wavelength domain: "
      f"signal {ls_nm.min():.1f}-{ls_nm.max():.1f} nm, "
      f"idler {li_nm.min():.1f}-{li_nm.max():.1f} nm")
print("  n_eff spline extrapolation is disabled (UnivariateSpline ext=2).")
print(f"  all dispersion plots are restricted to {wl.min():.0f}-{wl.max():.0f} nm.")

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

# Plot: rejection, useful pair transmission, source-aware baseline metric.
xK = np.array([r["K1"] for r in opt_K1]) * 100
rejK = np.array([r["rej_stage"] for r in opt_K1])
pairK = np.array([r["pair_pass"] for r in opt_K1])
sysK = np.array([r["system_rel"] for r in opt_K1])
sysK_norm = sysK / np.nanmax(sysK)

fopt, aopt = plt.subplots(1, 3, figsize=(15, 4.5))

aopt[0].semilogx(xK, rejK)
aopt[0].axhline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
aopt[0].axvline(100*Kcrit, ls=":", label="derived critical coupling")
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

aopt[2].semilogx(xK, sysK_norm)
aopt[2].scatter([100*best_system["K1"]],
                [best_system["system_rel"]/np.nanmax(sysK)],
                marker="*", s=100, label="source-aware max")
aopt[2].set_xlabel("K1 power coupling (%)")
aopt[2].set_ylabel("normalized relative KPI")
aopt[2].set_title("(c) source-aware single-device comparison")
aopt[2].grid(alpha=.3)
aopt[2].legend(fontsize=7)

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

# Pareto frontier: keep a point only if no sampled design has both greater
# rejection and greater/equal pair transmission.
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
a3k[0].axvline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
a3k[0].scatter([baseline_3K["rej"]],
               [baseline_3K["Ts"]*baseline_3K["Ti"]],
               marker="x", s=70, label="Butterworth baseline")
a3k[0].scatter([best_rejection_3K["rej_stage"]],
               [best_rejection_3K["pair_pass"]],
               marker="*", s=120, label="max rejection")
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
a3k[1].set_xticks(_x, _labels)
a3k[1].set_ylabel("POWER coupling K (%)")
a3k[1].set_title("(b) baseline vs max-rejection couplings")
a3k[1].grid(alpha=.3, which="both")
a3k[1].legend(fontsize=7)

_bw_gaps = [gap_for_K(v/100) for v in _bw_vals]
_rej_gaps = [gap_for_K(v/100) for v in _rej_vals]
a3k[2].plot(_x, _bw_gaps, "o-", label="Butterworth")
a3k[2].plot(_x, _rej_gaps, "s-", label="max-rejection design")
a3k[2].set_xticks(_x, ["g1", "g12", "g3"])
a3k[2].set_ylabel("physical gap from FEMWELL K(g) (nm)")
a3k[2].set_title("(c) physical implementation")
a3k[2].grid(alpha=.3)
a3k[2].legend(fontsize=7)

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
# The radius interval is generated from the documented 100-200 GHz FSR regime
# instead of inventing an independent radius range.  The pump is snapped to the
# nearest ODD longitudinal order so the 2:1 parity argument is preserved.
#
# The joint search asks:
#   A) What is the maximum single-device pump rejection?
#   B) What is the maximum single-device Ts*Ti?
#   C) Does ANY sampled two-ring design reach >=100 dB?
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


def series_rings_geom(l, geom, K1, K2, K3, A, dn2=0.0):
    """Same Rabus two-ring equations with local L1,L2.

    dn2 is an optional differential effective-index perturbation applied only
    to ring 2, used later for the fabrication/detuning sensitivity plot.
    """
    k1_, t1_, k2_, t2_, k3_, t3_ = couplers_independent(K1, K2, K3)
    L1_, L2_ = geom["L1"], geom["L2"]

    a1_ = np.sqrt(alpha_of(A, L1_)) * np.exp(1j * theta(l, L1_) / 2)
    a2_ = np.sqrt(alpha_of(A, L2_)) * np.exp(
        1j * 2*np.pi*(n_eff(l) + dn2)*L2_/(l*1e-9)/2
    )

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


def kpis_radius(geom, K1, K2, K3, A, dn2=0.0):
    """Single-device KPIs for one radius and one K1,K2,K3 triplet."""
    try:
        Tp_ = abs(series_rings_geom(pump_nm, geom, K1, K2, K3, A, dn2)[1])**2
        Ts_ = abs(series_rings_geom(geom["ls"], geom, K1, K2, K3, A, dn2)[1])**2
        Ti_ = abs(series_rings_geom(geom["li"], geom, K1, K2, K3, A, dn2)[1])**2
    except np.linalg.LinAlgError:
        return dict(Tp=np.nan, Ts=np.nan, Ti=np.nan, rej=np.nan, pair_pass=np.nan)

    rej_ = -10*np.log10(max(Tp_, 1e-300))
    return dict(
        Tp=Tp_, Ts=Ts_, Ti=Ti_,
        rej=rej_,
        ER=10*np.log10(max(Ts_, 1e-300)/max(Tp_, 1e-300)),
        pair_pass=Ts_*Ti_
    )


# [NUMERICAL CHOICE] 13 target-FSR points over the documented 100-200 GHz range.
FSR_grid_R = np.linspace(P["FSR_GHz"][0], P["FSR_GHz"][2], 13)

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

for fsr_target_ in FSR_grid_R:
    geom_ = geometry_from_fsr_target(fsr_target_)
    local_best_rej = None
    local_best_pair = None
    local_best_feasible = None

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

    best_rej_per_radius.append(local_best_rej)
    best_pair_per_radius.append(local_best_pair)
    best_feasible_per_radius.append(local_best_feasible)

best_joint_rejection = max(best_rej_per_radius, key=lambda r: r["rej"])
best_joint_pair = max(best_pair_per_radius, key=lambda r: r["pair_pass"])

_feasible_joint = [r for r in best_feasible_per_radius if r is not None]
best_joint_feasible = (max(_feasible_joint, key=lambda r: r["pair_pass"])
                       if _feasible_joint else None)


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
aR[0].set_xlabel("actual FSR (GHz)")
aR[0].set_ylabel("equivalent radius (um)")
aR[0].set_title("(a) radius from FSR + exact pump order")
aR[0].grid(alpha=.3)
aR[0].legend(fontsize=7)

aR[1].plot([r["R1_um"] for r in radius_sweep],
           [r["rej"] for r in radius_sweep], "o-", label="rejection")
aR[1].axhline(REJ_TARGET_DB, ls="--", label="100 dB requirement")
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
aR[2].set_xlabel("R1 (um)")
aR[2].set_ylabel("max pump rejection (dB)")
aR[2].set_title("(c) joint radius + K1/K2/K3 search")
aR[2].grid(alpha=.3)
aRc = aR[2].twinx()
aRc.plot([r["R1_um"] for r in best_rej_per_radius],
         [r["pair_pass"] for r in best_rej_per_radius],
         "s--", label=r"$T_sT_i$ at max rejection")
aRc.set_ylabel(r"$T_sT_i$")
h1, l1_ = aR[2].get_legend_handles_labels()
h2, l2_ = aRc.get_legend_handles_labels()
aR[2].legend(h1+h2, l1_+l2_, fontsize=7, loc="best")

fR.suptitle("[T4-OPTR] Exactly two rings: radius/coupling feasibility study")
fR.tight_layout()
fR.savefig("lean_fig_OPT_Radii.png", dpi=170)
plt.close(fR)



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
# Plot ONLY inside the physically simulated FEMWELL wavelength interval.
# UnivariateSpline(ext=2) intentionally forbids extrapolation, so every
# diagnostic/figure must respect the same domain as the phase-matching model.
#
# IMPORTANT PLOTTING CHOICE:
#   The optimizer produces continuous target couplings K1,K2,K3.  Those are
#   first inverted through the FEMWELL-derived K(g) map to continuous gaps.
#   A mask/fabrication grid of 10 nm is then imposed: 220 nm and 230 nm are
#   allowed, for example, but 222 nm is not.  AFTER snapping the geometry, the
#   coupling is recalculated from the snapped gap.  Every final spectrum and
#   loss/transmission sweep below uses these REALISED mask couplings.
#
#   continuous optimum K -> ideal g -> 10-nm mask g -> realised kappa_z,K
#   -> transfer function / figures.

MASK_GRID_NM = 10.0                       # [MODELLING / MASK QUANTISATION CHOICE]
plot_design_ideal = best_joint_rejection
plot_geom = best_joint_rejection
plot_FSR_nm = plot_geom["FSR_nm"]
plot_lam_s = plot_geom["ls"]
plot_lam_i = plot_geom["li"]
plot_A = typ["A_dB_per_cm"]


def snap_gap_to_mask(g_nm, step_nm=MASK_GRID_NM):
    """Round a continuous gap to the nearest allowed mask value.

    Uses half-up rounding rather than Python's bankers-rounding.  The result is
    clipped to the FEMWELL-simulated interval, because K(g) is never extrapolated.
    """
    gq = np.floor(float(g_nm)/step_nm + 0.5) * step_nm
    return float(np.clip(gq, GAP_MIN_NM, GAP_MAX_NM))


ideal_K = np.array([
    plot_design_ideal["K1"],
    plot_design_ideal["K2"],
    plot_design_ideal["K3"],
], dtype=float)
ideal_gaps_nm = np.array([gap_for_K(K_) for K_ in ideal_K], dtype=float)
if not np.all(np.isfinite(ideal_gaps_nm)):
    raise RuntimeError("At least one optimized K cannot be mapped inside the FEMWELL gap domain.")

mask_gaps_nm = np.array([snap_gap_to_mask(g_) for g_ in ideal_gaps_nm], dtype=float)
real_kappa_z = np.array([kappa_z_interp(g_) for g_ in mask_gaps_nm], dtype=float)
real_K = np.array([K_of_gap(g_) for g_ in mask_gaps_nm], dtype=float)
real_kappa_amp = np.sqrt(real_K)
real_t_amp = np.sqrt(1.0-real_K)

plot_K1, plot_K2, plot_K3 = real_K
plot_KP = kpis_radius(plot_geom, plot_K1, plot_K2, plot_K3, plot_A)
ideal_KP = kpis_radius(
    plot_geom,
    plot_design_ideal["K1"], plot_design_ideal["K2"], plot_design_ideal["K3"],
    plot_A
)

plot_Qo, plot_Qe = Q_factors(plot_geom["L1"], plot_A, plot_K1)
plot_QL = 1.0 / (1.0/plot_Qo + 1.0/plot_Qe)
plot_F_ring = plot_FSR_nm / (pump_nm / plot_QL)
plot_i_fsr = np.argmin(np.abs(pump_nm - ls_nm - plot_FSR_nm))

print("\n" + "="*78)
print("[T4-MASK] CONTINUOUS OPTIMUM -> 10-nm MASK -> REALISED COUPLINGS")
print("="*78)
print(f"mask grid = {MASK_GRID_NM:.0f} nm; FEMWELL K(g) domain = "
      f"{GAP_MIN_NM:.0f}-{GAP_MAX_NM:.0f} nm; Lc = {L_c:.1f} um")
print(f"geometry: actual FSR={plot_geom['fsr_GHz']:.3f} GHz, "
      f"R1={plot_geom['R1_um']:.3f} um, R2={plot_geom['R2_um']:.3f} um")
print(f"{'coupler':8s} {'K target (%)':>13s} {'g ideal (nm)':>14s} "
      f"{'g mask (nm)':>12s} {'kappa_z':>12s} {'|kappa|':>11s} "
      f"{'K real (%)':>12s} {'dK/K':>9s}")
for nm_, Kt_, gi_, gm_, kz_, ka_, Kr_ in zip(
        ("K1", "K2", "K3"), ideal_K, ideal_gaps_nm, mask_gaps_nm,
        real_kappa_z, real_kappa_amp, real_K):
    print(f"{nm_:8s} {100*Kt_:13.6f} {gi_:14.2f} {gm_:12.0f} "
          f"{kz_:12.6e} {ka_:11.6f} {100*Kr_:12.6f} "
          f"{100*(Kr_/Kt_-1):+8.2f}%")

print("\nTransfer-function impact of 10-nm mask quantisation:")
print(f"  continuous optimum : rejection={ideal_KP['rej']:.2f} dB, "
      f"Ts={ideal_KP['Ts']:.5f}, Ti={ideal_KP['Ti']:.5f}, "
      f"TsTi={ideal_KP['pair_pass']:.5f}")
print(f"  realised mask      : rejection={plot_KP['rej']:.2f} dB, "
      f"Ts={plot_KP['Ts']:.5f}, Ti={plot_KP['Ti']:.5f}, "
      f"TsTi={plot_KP['pair_pass']:.5f}")
print(f"  delta              : rejection={plot_KP['rej']-ideal_KP['rej']:+.2f} dB, "
      f"Ts={100*(plot_KP['Ts']/ideal_KP['Ts']-1):+.2f} %, "
      f"TsTi={100*(plot_KP['pair_pass']/ideal_KP['pair_pass']-1):+.2f} %")
print("  NOTE: these are MASK-REALISED gaps, not post-fabrication measured gaps.")

lf = np.linspace(wl.min(), wl.max(), 200)

f1, a = plt.subplots(1, 3, figsize=(15, 4.2))
a[0].plot(lf, n_eff(lf), label=r"$n_{eff}$")
a[0].plot(wl, neff_v, "o", ms=3, label="FEMWELL samples")
a[0].plot(lf, n_g(lf), label=r"$n_g$")
a[0].legend(fontsize=8); a[0].grid(alpha=.3)
a[0].set_xlabel("wavelength (nm)"); a[0].set_ylabel("index")
a[0].set_title("(a) dispersion [RABUS07 2.20]")

a[1].plot(lf, [D_ps(x) for x in lf])
a[1].axhline(0, color="gray", ls="--")
a[1].axvline(pump_nm, color="k", ls=":", lw=1, label="pump")
a[1].set_xlabel("wavelength (nm)")
a[1].set_ylabel("D (ps/(nm km))")
a[1].grid(alpha=.3)
a[1].legend(fontsize=8)

_dispersion_label = "anomalous" if D_ps(pump_nm) > 0 else "normal"
a[1].set_title(
    f"(b) GVD @ pump = {D_ps(pump_nm):+.0f} ps/(nm km), {_dispersion_label}"
)
a[2].plot(dOm / (2 * np.pi) * 1e-12, sinc2(dk * 1e-3 / 2), label="L = 1 mm straight")
a[2].plot(dOm / (2 * np.pi) * 1e-12, sinc2(dk * plot_geom['L1'] * plot_F_ring / np.pi / 2),
          label=r"ring, $L\mathcal{F}/\pi$ @ plotted design")
a[2].axvline(dOm[plot_i_fsr] / (2 * np.pi) * 1e-12, color="r", ls=":", label="1 plotted FSR")
a[2].set_xlabel(r"$\Delta\Omega/2\pi$ (THz)"); a[2].set_ylabel(r"sinc$^2$")
a[2].legend(fontsize=7); a[2].grid(alpha=.3); a[2].set_title("(c) phase matching")
f1.suptitle("[T4-1] Straight SOI waveguide: dispersion and phase matching")
f1.tight_layout(); f1.savefig("lean_fig1_T4-1.png", dpi=170); plt.close(f1)

# -------------------------------------------------------------------------
# Mask-realised spectrum / transmission figure
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
a[0].set_title(f"(a) field buildup with R1 = {plot_geom['R1_um']:.1f} um")

a[1].plot(lams - pump_nm, abs(out[:, 3]) ** 2 / max((abs(out[:, 3]) ** 2).max(), 1e-300),
          lw=.8, color="tab:orange")
a[1].set_ylabel("ring 2\n(filter)")
a[1].set_title(f"(b) field buildup with R2 = {plot_geom['R2_um']:.1f} um")

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
    f"(c) 10-nm-mask spectrum: rej = {plot_KP['rej']:.2f} dB, "
    f"Ts = {plot_KP['Ts']:.3f}, Ti = {plot_KP['Ti']:.3f}"
)
for ax in a:
    for l_, col in ((pump_nm, "k"), (plot_lam_s, "r"), (plot_lam_i, "b")):
        ax.axvline(l_ - pump_nm, color=col, ls=":", lw=1)
    ax.grid(alpha=.3)
f2.suptitle("[T4-4] T(omega) for the 10-nm-mask-realised two-ring design")
f2.tight_layout(); f2.savefig("lean_fig2_T4-4.png", dpi=170); plt.close(f2)

# -------------------------------------------------------------------------
# Sweeps referenced to the mask-realised design
# -------------------------------------------------------------------------
K1s = np.geomspace(K13_MIN, K13_MAX, 60)
As = np.linspace(P["A_dB_per_cm"][0], P["A_dB_per_cm"][2], 40)
rK = [kpis_radius(plot_geom, K1_, plot_K2, plot_K3, plot_A) for K1_ in K1s]
rA = [kpis_radius(plot_geom, plot_K1, plot_K2, plot_K3, A_) for A_ in As]

gg = np.linspace(gaps.min(), gaps.max(), 240)
# ideal_gaps_nm are the continuous optimizer values; mask_gaps_nm are the
# actual 10-nm layout values whose couplings are used in every final figure.

f3, a = plt.subplots(1, 3, figsize=(15, 4.2))
a[0].semilogx(100*K1s, [r["rej"] for r in rK], "o-", ms=3, label="rejection")
a[0].axvline(100*plot_K1, color="k", ls=":", label="mask-realised K1")
a[0].set_xlabel("K1 (%) with K2,K3 fixed at mask-realised values")
a[0].set_ylabel("pump rejection (dB)")
a[0].grid(alpha=.3)
a[0].legend(fontsize=7)
ab = a[0].twinx()
ab.semilogx(100*K1s, [r["Ts"] for r in rK], "s--", ms=3, color="tab:orange", label=r"$T_s$")
ab.set_ylabel(r"$T_s$", color="tab:orange")
a[0].set_title("(a) K1 cut around the mask-realised design")

a[1].plot(As, [r["rej"] for r in rA], "o-", ms=3, label="rejection")
a[1].axvline(plot_A, color="k", ls=":", label="nominal loss")
a[1].set_xlabel("loss A (dB/cm)")
a[1].set_ylabel("pump rejection (dB)")
a[1].grid(alpha=.3)
ac = a[1].twinx()
ac.plot(As, [r["Ts"] for r in rA], "s--", ms=3, color="tab:orange", label=r"$T_s$")
ac.set_ylabel(r"$T_s$", color="tab:orange")
h1, l1_ = a[1].get_legend_handles_labels()
h2, l2_ = ac.get_legend_handles_labels()
a[1].legend(h1+h2, l1_+l2_, fontsize=7, loc="best")
a[1].set_title("(b) loss sweep at mask-realised couplings")

a[2].semilogy(gaps, kzv, "o", label="FEMWELL supermodes")
a[2].semilogy(gg, [kappa_z_interp(x) for x in gg], "-", label="PCHIP between FEMWELL samples")
for gi_, gm_, lbl_ in zip(ideal_gaps_nm, mask_gaps_nm, ("g1", "g12", "g3")):
    a[2].axvline(gi_, ls="--", alpha=.55, label=f"{lbl_} ideal = {gi_:.1f} nm")
    a[2].axvline(gm_, ls=":", label=f"{lbl_} mask = {gm_:.0f} nm")
a[2].set_xlabel("gap (nm)")
a[2].set_ylabel(r"$\kappa_z$ (rad/um)")
a[2].legend(fontsize=7)
a[2].grid(alpha=.3, which="both")
a[2].set_title("(c) coupling-vs-gap with ideal and 10-nm mask gaps")
f3.suptitle("[T4-4/4b] Sweeps referenced to the 10-nm-mask-realised design")
f3.tight_layout(); f3.savefig("lean_fig3_sweeps.png", dpi=170); plt.close(f3)

# -------------------------------------------------------------------------
# Dedicated implementation figure: target coupling -> continuous gap -> mask gap.
# -------------------------------------------------------------------------
fmask, am = plt.subplots(1, 3, figsize=(15, 4.2))
_xm = np.arange(3)
_lblm = ["K1", "K2", "K3"]
am[0].semilogy(_xm, 100*ideal_K, "o-", label="continuous optimum")
am[0].semilogy(_xm, 100*real_K, "s--", label="10-nm mask realised")
am[0].set_xticks(_xm, _lblm)
am[0].set_ylabel("power coupling K (%)")
am[0].set_title("(a) target vs realised coupling")
am[0].grid(alpha=.3, which="both"); am[0].legend(fontsize=7)

_w = 0.34
am[1].bar(_xm-_w/2, ideal_gaps_nm, width=_w, label="continuous gap")
am[1].bar(_xm+_w/2, mask_gaps_nm, width=_w, label="10-nm mask gap")
am[1].set_xticks(_xm, ["g1", "g12", "g3"])
am[1].set_ylabel("gap (nm)")
am[1].set_title("(b) geometry quantisation")
am[1].grid(alpha=.3, axis="y"); am[1].legend(fontsize=7)

# Compare final drop spectra using the SAME radius, loss and wavelength grid.
out_ideal = np.array([
    series_rings_geom(l, plot_geom, ideal_K[0], ideal_K[1], ideal_K[2], plot_A)
    for l in lams
])
am[2].semilogy(lams-pump_nm, np.maximum(abs(out_ideal[:,1])**2, 1e-14),
               label="continuous optimum")
am[2].semilogy(lams-pump_nm, np.maximum(abs(out[:,1])**2, 1e-14),
               ls="--", label="10-nm mask realised")
am[2].axhline(1e-10, ls=":", label="-100 dB reference")
am[2].set_ylim(1e-12, 2)
am[2].set_xlabel(r"$\lambda-\lambda_p$ (nm)")
am[2].set_ylabel(r"$T_{drop}$")
am[2].set_title("(c) transfer-function impact")
am[2].grid(alpha=.3); am[2].legend(fontsize=7)
fmask.suptitle("[T4-MASK] 10-nm gap grid: continuous optimum vs fabricable mask design")
fmask.tight_layout(); fmask.savefig("lean_fig_MASK_10nm.png", dpi=170); plt.close(fmask)

# -------------------------------------------------------------------------
# Variability figure referenced to the 10-nm-mask-realised design.
# -------------------------------------------------------------------------
plot_tol = [
    kpis_radius(plot_geom, plot_K1, plot_K2, plot_K3, plot_A, dn2=dn_dw*x)
    for x in dw
]
plot_Ts_t = np.array([t["Ts"] for t in plot_tol])
plot_ER_t = np.array([t["ER"] for t in plot_tol])

f4, a = plt.subplots(1, 2, figsize=(11, 4.3))
a[0].plot(dw, plot_Ts_t, color="tab:orange"); a[0].set_xlim(-0.5, 0.5)
a[0].axhline(0.5 * plot_KP["Ts"], color="gray", ls="--", lw=1)
a[0].set_xlabel("differential width error (nm)")
a[0].set_ylabel(r"$T_{drop}$(signal)")
a[0].grid(alpha=.3)
a[0].set_title("(a) DIFFERENTIAL -- 10-nm-mask design")
ab = a[0].twinx(); ab.plot(dw, plot_ER_t, color="tab:green"); ab.set_ylabel("extinction (dB)", color="tab:green")
dwc = np.linspace(-5, 5, 50)
a[1].plot(dwc, dwc * shift * 1e-3, color="tab:blue", label="first-order comb shift")
_plot_fwhm_nm = pump_nm / plot_QL
a[1].axhspan(-_plot_fwhm_nm/2, _plot_fwhm_nm/2, color="green", alpha=.12,
             label=r"$\pm$FWHM/2 (derived)")
a[1].axhline(+plot_FSR_nm/2, color="gray", ls=":", lw=1, label=r"$\pm$FSR/2 reference")
a[1].axhline(-plot_FSR_nm/2, color="gray", ls=":", lw=1)
a[1].set_xlabel("common-mode width error (nm)"); a[1].set_ylabel("resonance shift (nm)")
a[1].legend(fontsize=7); a[1].grid(alpha=.3); a[1].set_title("(b) COMMON -- plotted design")
f4.suptitle("[T4-5] First-order width sensitivity of the 10-nm-mask design")
f4.tight_layout(); f4.savefig("lean_fig4_T4-5.png", dpi=170); plt.close(f4)
print("\nFigures: lean_fig0_mode, lean_fig1_T4-1, lean_fig2_T4-4, lean_fig3_sweeps,")
print("         lean_fig_OPT_K1, lean_fig_OPT_3K, lean_fig_OPT_Radii, lean_fig_MASK_10nm,")
print("         lean_fig4_T4-5")


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
  from the gap [CHROST15 4.3/4.5/4.8].  The supermode METHOD is benchmarked
  separately on the CHROST15 500x220-nm, 1550-nm reference geometry, giving
  Lx = {Lx_chrost_benchmark:.2f} um vs ~37.5 um ({chrost_err_pct:+.2f} %).
  The actual 400x220-nm device is not forced to reproduce that reference value.
  PGR ~ P^2 Q^3 [BAJONI17 13-14, measured by
  SAVANIER16 Fig. 6]; the shift law dlambda/lambda = dn_eff/n_g [THOMSON16].
  L2 = L1/2 with one index makes theta2 = theta1/2 identically -- pure algebra.

FIXED UPSTREAM INPUTS:
  w=400 nm, h=220 nm and lambda_p=1580 nm are inherited from the preceding
  team GVD-design stage.  They are accepted as inputs here rather than re-optimized.

ASSUMED:
  the 2:1 Vernier ratio, read off the supervisor's comb slide (NOT stated
  numerically anywhere -- the single assumption the whole design rests on);
  lossless, frequency-independent couplers; CW undepleted pump; the 2:1 Vernier
  constraint R2 = R1/2; the local K(g) coupler map is reused across the radius
  sweep (radius-dependent coupler curvature is not simulated); the PGR prefactor
  is calibrated on one measured die.

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
  the architecture is fixed to EXACTLY TWO rings.  The nominal |00| case gives
  {KP['rej']:.0f} dB pump rejection with Ts={KP['Ts']:.3f}, Ti={KP['Ti']:.3f}.
  The independent 3-K study maximizes rejection and maps the optimized K values
  back to FEMWELL-derived physical gaps.  The radius study then searches the
  documented 100-200 GHz FSR interval while preserving R2=R1/2.
  The maximum sampled two-ring rejection after the joint radius/K search is
  {best_joint_rejection['rej']:.2f} dB before layout quantisation.  With the 10-nm
  mask grid, the realised couplings give {plot_KP['rej']:.2f} dB pump rejection and
  Ts*Ti = {plot_KP['pair_pass']:.5f}.  Therefore:
      {'THE 100 dB TARGET IS MET.' if best_joint_feasible is not None else 'THE 100 dB TARGET IS NOT MET by this two-ring topology in the investigated domain.'}
  No cascaded double-ring stages are used to manufacture feasibility.
  The fixed w=400 nm, h=220 nm and lambda_p=1580 nm inputs come from the
  upstream team GVD-design stage and are not re-optimized in Task 4.
  Differential width error is reported only as a first-order equivalent-width
  sensitivity; the physical requirement is resonance alignment within the
  loaded linewidth.
""")
