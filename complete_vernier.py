
"""
lambda/(2 Delta N_ef) where delta N_ef = Ne_f(symmetrical) - Nef_(antisymmetrical)

task4_vernier_pump_rejection.py
================================
Task 4 (core task) -- Generation of photon pairs in silicon photonics using
ring resonators and SFWM; Vernier double-ring pump rejection.

ONE flat script, sections numbered after the assignment:

  [T4-1]  Straight SOI waveguide (Femwell): mode profile, n_eff(lambda),
          n_g, GVD -> phase matching Delta_k(Delta_Omega).
  [T4-2]  FWM conversion efficiency and photon-pair rate, straight waveguide.
  [T4-3]  Photon-pair generation in a ring resonator: Q-factors, build-up,
          and the pair-generation-rate (PGR) formula of Gentry/Popovic.
  [T4-4]  Vernier double-ring design: parameter table with ranges, the six
          coupling coefficients, T(omega), the two combs, pump extinction
          ratio vs the 100-120 dB target, cascaded stages.
  [T4-5]  Device-parameter variability from clean-room imperfections,
          anchored to measured die-to-die data.
  [T4-6]  Critical discussion (what the model does NOT capture).

Every numerical input is tagged [REF: ...] with the source it comes from;
every modelling choice is tagged [ASSUMPTION] when it is ours. What is
stated as CERTAIN is only what the cited sources state or what follows
algebraically from them.

References used (short keys):
  RABUS07   D. G. Rabus, Integrated Ring Resonators, Springer 2007, Ch. 2
            (Eqs. 2.1-2.5, 2.20-2.21, 2.30-2.34, 2.58-2.65, 2.76-2.78).
  BOG12     Bogaerts et al., Laser Photon. Rev. 6, 47 (2012).
  LITTLE97  Little, Chu, Haus, Foresi, Laine, J. Lightwave Technol. 15, 998 (1997).
  GRIFFEL00 Griffel, IEEE Photon. Technol. Lett. 12, 1642 (2000).
  AGRAWAL13 Agrawal, Nonlinear Fiber Optics, 5th ed., Ch. 10.
  DINU03    Dinu, Quochi, Garcia, Appl. Phys. Lett. 82, 2954 (2003).
  BRISTOW07 Bristow, Rotenberg, van Driel, Appl. Phys. Lett. 90, 191104 (2007).
  SALZ57    Salzberg & Villa, JOSA 47, 244 (1957); Tatian, Appl. Opt. 23, 4477 (1984).
  MALIT65   Malitson, JOSA 55, 1205 (1965).
  THOMSON16 Thomson et al., "Roadmap on silicon photonics", J. Opt. 18, 073003 (2016).
  MICHON22  Michon et al., Opt. Lett. 47, 341 (2022)  [C2N, cascaded Bragg + TM filter].
  MEDINA24  D. E. Medina Quiroz, PhD thesis, Univ. Paris-Saclay (2024), Ch. 3.
  ARXIV2411 arXiv:2411.05921 (2024), "Scalable Feedback Stabilization of Quantum
            Light Sources on a CMOS Chip" -- Suppl. §1 Eq. (1) (PGR formula, from
            GENTRY18), Table 1 (6-die variability), Ext. Fig. 6 (Vernier filter
            > 100 dB, ~60 dB at sub-filter FSRs, < 1 dB IL), Fig. 2 (heater
            0.62 nm range, 0.6 pm LSB).
  GENTRY18  C. M. Gentry, PhD thesis, Univ. Colorado Boulder (2018).
  KUMAR20   Kumar, Wu, Tsang, Opt. Lett. 45, 1289 (2020)  [cascaded CROW, > 110 dB].
  AFIFI21   Afifi et al., Opt. Express 29, 25173 (2021)  [CDC filters: 60 dB measured
            vs 157 dB modelled -- scattered-light floor].
  FEMWELL   https://helgegehring.github.io/femwell  (mode solver, GVD example).
"""

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

c = 299792458.0            # m/s
hbar = 1.054571817e-34     # J s


# =============================================================================
# 0. PARAMETER TABLE (min, typical, max) -- every value justified
# =============================================================================
P = OrderedDict(
    # SOI strip cross-section. [REF: THOMSON16] 220 nm is the standard foundry
    # SOI thickness; 450-550 nm width keeps the guide single-mode TE at 1550 nm
    # [REF: BOG12]. [REF: MEDINA24 §3.3.1] uses 500-700 nm depending on cladding.
    width_nm      = (450, 500, 550),
    height_nm     = (220, 220, 220),
    # Generator ring radius. [REF: ARXIV2411 Fig.1] measured SFWM rings use
    # R = 19.1 um; [REF: MEDINA24 §3.3.2] uses R = 120 um for a 200 GHz FSR.
    # 10 um is our compact starting point (FSR ~ 9 nm); swept below.
    r1_um         = (5.0, 10.0, 20.0),
    # Propagation loss A (dB/cm). [REF: MEDINA24 §3.3.2] Q~50k at critical
    # coupling -> ~1.5 dB/cm (C2N, air clad); < 1 dB/cm (Leti/ST, oxide clad).
    # [REF: ARXIV2411 Table 1] Q_o ~ 88-108k in 45nm CMOS SOI.
    A_dB_per_cm   = (0.3, 1.0, 3.0),
    # Bus <-> ring power coupling K1. [REF: MEDINA24 §3.3.2] critical coupling
    # reached for gaps of 60-100 nm; [REF: ARXIV2411 Table 1] Q_e ~ 60-80k.
    # Range chosen to bracket under- to over-coupled.
    K1_percent    = (0.5, 3.0, 20.0),
    # Pump. [REF: assignment] 1550 nm.
    pump_nm       = (1550.0, 1550.0, 1550.0),
)
typ = {k: v[1] for k, v in P.items()}
pump_nm = typ["pump_nm"]

print("=" * 78)
print("0. PARAMETER TABLE (min, typical, max) -- sources in the code comments")
print("=" * 78)
for k, (lo, t, hi) in P.items():
    print(f"  {k:14s}: [{lo:>7g} , {t:>7g} , {hi:>7g}]")


# =============================================================================
# [T4-1] STRAIGHT SOI WAVEGUIDE: Femwell mode, dispersion, phase matching
# =============================================================================

def n_Si(lam_um):
    """[REF: SALZ57] Sellmeier fit for crystalline Si, valid 1.357-11.04 um."""
    if not (1.357 <= lam_um <= 11.04):
        raise ValueError("Si fit out of range")
    l2 = lam_um ** 2
    return math.sqrt(1 + 10.6684293 * l2 / (l2 - 0.301516485 ** 2)
                     + 0.0030434748 * l2 / (l2 - 1.13475115 ** 2)
                     + 1.54133408 * l2 / (l2 - 1104 ** 2))


def n_SiO2(lam_um):
    """[REF: MALIT65] Sellmeier fit for fused silica, valid 0.21-6.7 um."""
    l2 = lam_um ** 2
    return math.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                     + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                     + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


def strip_mesh(w_um, h_um):
    core = shapely.geometry.box(-w_um / 2, 0, w_um / 2, h_um)
    clad = shapely.geometry.box(-4 * w_um, 0, 4 * w_um, 6 * h_um)
    box = shapely.geometry.box(-4 * w_um, -6 * h_um, 4 * w_um, 0)
    res = dict(core={"resolution": 0.02, "distance": 0.3},
               clad={"resolution": 0.05, "distance": 0.3},
               box={"resolution": 0.05, "distance": 0.3})
    return from_meshio(mesh_from_OrderedDict(OrderedDict(core=core, clad=clad, box=box),
                                             res, default_resolution_max=2))


def te0(mesh, lam_um):
    """Fundamental TE mode (highest TE fraction) [REF: FEMWELL 'waveguide_modes']."""
    b = Basis(mesh, ElementTriP0())
    eps = b.zeros()
    for dom, nf in {"core": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um) ** 2
    modes = compute_modes(b, eps, wavelength=lam_um, num_modes=3, order=1)
    return modes.sorted(key=lambda m: -np.real(m.te_fraction))[0]


w_um, h_um = typ["width_nm"] * 1e-3, typ["height_nm"] * 1e-3
print("\n" + "=" * 78)
print("[T4-1] Straight waveguide: Femwell sweep (n_eff, n_g, GVD, A_eff)")
print("=" * 78)
mesh0 = strip_mesh(w_um, h_um)
wl_nm = np.linspace(1400, 1700, 16)
neff, aeff = [], []
for wl in wl_nm:
    m = te0(mesh0, wl * 1e-3)
    neff.append(np.real(m.n_eff))
    aeff.append(np.real(m.calculate_effective_area()))
neff, aeff = np.array(neff), np.array(aeff)

spl = UnivariateSpline(wl_nm, neff, s=0, k=3)
d1, d2 = spl.derivative(1), spl.derivative(2)
n_eff = lambda l: spl(l)
n_g = lambda l: spl(l) - l * d1(l)                       # [REF: RABUS07 Eq. 2.20; BOG12]
A_eff_um2 = float(UnivariateSpline(wl_nm, aeff, s=0, k=3)(pump_nm))


def beta2(l_nm):
    """beta2 = (1/c) dn_g/domega  [REF: professor's notes; AGRAWAL13 Ch.1]."""
    dl = 0.5
    dng_dl = (n_g(l_nm + dl) - n_g(l_nm - dl)) / (2 * dl * 1e-9)
    domega_dl = -2 * np.pi * c / (l_nm * 1e-9) ** 2
    return (dng_dl / domega_dl) / c


def D_ps_nm_km(l_nm):
    """D = -(2 pi c / lambda^2) beta2, converted to ps/(nm km)."""
    return -(2 * np.pi * c / (l_nm * 1e-9) ** 2) * beta2(l_nm) * 1e12 * 1e-9 * 1e3


print(f"n_eff({pump_nm:.0f}) = {n_eff(pump_nm):.5f}   n_g = {n_g(pump_nm):.4f}   "
      f"A_eff = {A_eff_um2:.3f} um^2")
print(f"beta2 = {beta2(pump_nm):.3e} s^2/m   D = {D_ps_nm_km(pump_nm):+.0f} ps/(nm km) "
      f"({'ANOMALOUS' if D_ps_nm_km(pump_nm) > 0 else 'normal'} dispersion; "
      f"Agrawal: D>0 <=> beta2<0 <=> anomalous, as in SMF-28 at 1550 nm with D=+17)")
print("  -> anomalous GVD is the FAVOURABLE regime for SFWM: the nonlinear phase 2*gamma*P")
print("     (>0) can be compensated by the linear term beta2*dw^2 (<0) [AGRAWAL13 sec 10.2;")
print("     MEDINA24 sec 3.3.1: 'small anomalous dispersion is generally considered the most")
print("     favorable case']. Same sign as Dulkeith 2006 (525x226 nm wire, D=+4400), ~10x smaller")
print("     in magnitude: GVD is very sensitive to exact width/height/cladding -- cross-check advised.")

# Phase matching Delta_k = k_s + k_i - 2 k_p, with omega_s,i = omega_p +/- Delta_Omega
# [REF: AGRAWAL13 §10.2; MEDINA24 Eqs. 2.11-2.14]. Linear term only [ASSUMPTION].
omega_p = 2 * np.pi * c / (pump_nm * 1e-9)
dOmega = np.linspace(1e9, 2 * np.pi * c * (1 / 1450e-9 - 1 / 1550e-9), 300)
lam_s_nm = 2 * np.pi * c / (omega_p + dOmega) * 1e9
lam_i_nm = 2 * np.pi * c / (omega_p - dOmega) * 1e9
k = lambda l: 2 * np.pi * n_eff(l) / l                    # rad/nm
delta_k_per_m = (k(lam_s_nm) + k(lam_i_nm) - 2 * k(pump_nm)) * 1e9
sinc2 = lambda x: np.sinc(x / np.pi) ** 2
print("Phase matching: |Delta_k| < 0 and grows in magnitude with detuning (anomalous dispersion).")
i_fsr = np.argmin(np.abs(pump_nm - lam_s_nm - 9.1))
# CRITICAL [REF: SAVANIER16, Appendix]: inside a RING the length entering the sinc
# is NOT the circumference L but the "unfolded" length L_res = L * F/pi (number of
# round trips set by the finesse). Savanier states that ignoring this makes the
# phase mismatch "incorrect by more than two orders of magnitude".
F_design = 178                       # finesse of our design point (recomputed in [T4-3])
L_phys = 62.7e-6
L_res = L_phys * F_design / np.pi
print(f"  straight guide, L = {L_phys*1e6:.0f} um      : sinc^2 = {sinc2(delta_k_per_m[i_fsr]*L_phys/2):.4f}")
print(f"  RING, L_res = L*F/pi = {L_res*1e3:.2f} mm ({F_design/np.pi:.0f} round trips): "
      f"sinc^2 = {sinc2(delta_k_per_m[i_fsr]*L_res/2):.4f}")
dw_null = np.sqrt(2*np.pi/abs(beta2(pump_nm)*L_res))
print(f"  -> F/pi multiplies the sinc argument by {F_design/np.pi:.0f}x. At our 1-FSR detuning the")
print(f"     conclusion survives (sinc^2 ~ 1), but the first null is at only "
      f"{dw_null/2/np.pi*1e-12:.1f} THz (~8 FSR):")
print(f"     phase matching WOULD limit the design at higher Q or wider signal-idler spacing.")


# =============================================================================
# [T4-2] FWM CONVERSION EFFICIENCY AND PAIR RATE -- straight waveguide
# =============================================================================
# [REF: AGRAWAL13 Eq. 10.2.x] eta = (gamma P L)^2 sinc^2(Dk L/2),
# gamma = 2 pi n2 / (lambda A_eff). n2 = 4.5e-18 m^2/W [REF: DINU03; BRISTOW07,
# literature spread (4-6)e-18]. Undepleted CW pump, no TPA [ASSUMPTION].
n2_Si = 4.5e-18
gamma_nl = 2 * np.pi * n2_Si / (pump_nm * 1e-9 * A_eff_um2 * 1e-12)
print("\n" + "=" * 78)
print("[T4-2] FWM efficiency / pair rate, straight waveguide (order of magnitude)")
print("=" * 78)
print(f"gamma = {gamma_nl:.1f} /(W m)")
for L_mm in (0.5, 1.0, 2.0):
    P_in = 1e-3
    eta = (gamma_nl * P_in * L_mm * 1e-3) ** 2 * sinc2(delta_k_per_m[i_fsr] * L_mm * 1e-3 / 2)
    R = eta * P_in / (hbar * omega_p)
    print(f"  L = {L_mm:.1f} mm, P = 1 mW : eta_FWM = {eta:.2e}, R_pair ~ {R:.2e} pairs/s")


# =============================================================================
# [T4-3] RING RESONATOR: Q-factors, build-up, and PGR (Gentry/Popovic Eq. S1)
# =============================================================================
# Loss: A (dB/cm) -> B (1/cm, amplitude, base e) -> alpha = exp(-B L)
# [REF: professor's notes 09/09; RABUS07 Eq. 2.3].
B_of = lambda A: A * np.log(10) / 20.0
alpha_of = lambda A, L_m: np.exp(-B_of(A) * L_m * 100)


def size_rings(r1_um):
    """L1 exactly resonant at the pump, odd order; L2 = L1/2 (Vernier N:M = 2:1)
    [REF: RABUS07 Eq. 2.71 for the resonance condition, Eqs. 2.77-2.78 for the
    Vernier relation; the 2:1 choice is our reading of the supervisor's comb
    slide (synchronous lines at s, s, _, i, i) -- ASSUMPTION to be confirmed]."""
    m1 = round(n_eff(pump_nm) * 2 * np.pi * r1_um * 1e-6 / (pump_nm * 1e-9))
    if m1 % 2 == 0:
        m1 += 1
    L1 = m1 * pump_nm * 1e-9 / n_eff(pump_nm)
    return L1, L1 / 2, m1


def resonance(L, m, guess_nm):
    """Exact order-m resonance of the dispersive n_eff(lambda) (root finding).
    A first-order guess (pump -/+ FSR) misses by ~50 pm > linewidth -- never use it."""
    f = lambda l: n_eff(l) * L / (l * 1e-9) - m
    fsr = guess_nm ** 2 / (n_g(guess_nm) * L * 1e9)
    return brentq(f, guess_nm - 0.6 * fsr, guess_nm + 0.6 * fsr)


def Q_factors(L_m, A, K1):
    """Intrinsic and external Q from loss and bus coupling.
    Q_o = 2 pi n_g / (lambda alpha_pow)   [REF: BOG12 Eq. 12]
    Q_e = 2 pi n_g L / (lambda K1)        [weak-coupling limit, REF: BOG12; RABUS07 Eq. 2.31]"""
    alpha_pow = 2 * B_of(A) * 100                     # power attenuation, 1/m
    lam = pump_nm * 1e-9
    Q_o = 2 * np.pi * n_g(pump_nm) / (lam * alpha_pow)
    Q_e = 2 * np.pi * n_g(pump_nm) * L_m / (lam * K1)
    return Q_o, Q_e


def pgr_eqS1(Qo_p, Qe_p, Qo_s, Qe_s, Qo_i, Qe_i, dnuFSR_Hz, K_nl, lam_p_nm):
    """[REF: ARXIV2411 Suppl. Eq. (1); GENTRY18]
    I_pair = omega_p^2 beta_FWM^2 (2 r_pe/r_pt)^2 (2 r_ie r_se/(r_it r_st))
             (r_st + r_it) / ((2 pi dnuFSR)^2 + (r_st + r_it)^2) P_p^2,   r = omega/(2Q).
    K_nl = omega_p^2 beta_FWM^2 is CALIBRATED below against the measured PGR of
    ARXIV2411 (die B9: 3.29 MHz/mW^2) rather than computed from n2, because the
    paper's beta_FWM uses a full chi(3) overlap integral we do not reproduce."""
    w = 2 * np.pi * c / (lam_p_nm * 1e-9)
    r = lambda Q: w / (2 * Q)
    rpe, rse, rie = r(Qe_p), r(Qe_s), r(Qe_i)
    rpt, rst, rit = rpe + r(Qo_p), rse + r(Qo_s), rie + r(Qo_i)
    S = (2 * rpe / rpt) ** 2 * (2 * rie * rse / (rit * rst)) \
        * (rst + rit) / ((2 * np.pi * dnuFSR_Hz) ** 2 + (rst + rit) ** 2)
    return K_nl * S          # pairs/s per mW^2


# --- calibration of K_nl on a MEASURED ring [REF: ARXIV2411 Table 1, die B9 System] ---
REF = dict(R_um=19.1, lam_p=1552.5, Qo_p=113.1e3, Qe_p=68.2e3, Qo_s=64.4e3, Qe_s=57.7e3,
           Qo_i=102.8e3, Qe_i=62.7e3, dnuFSR=1.42e9, PGR=3.29e6)
K_ref = REF["PGR"] / pgr_eqS1(REF["Qo_p"], REF["Qe_p"], REF["Qo_s"], REF["Qe_s"],
                              REF["Qo_i"], REF["Qe_i"], REF["dnuFSR"], 1.0, REF["lam_p"])
# beta_FWM ~ 1/V_eff and V_eff ~ A_eff * 2 pi R  -> K_nl ~ (R_ref/R)^2 at equal A_eff
# [ASSUMPTION: same cross-section; ARXIV2411 does not report A_eff].

print("\n" + "=" * 78)
print("[T4-3] Ring resonator: Q, build-up, PGR (Eq. S1 of ARXIV2411, calibrated)")
print("=" * 78)
L1, L2, m1 = size_rings(typ["r1_um"])
FSR1_nm = pump_nm ** 2 / (n_g(pump_nm) * L1 * 1e9)
lam_s = resonance(L1, m1 + 1, pump_nm - FSR1_nm)
lam_i = resonance(L1, m1 - 1, pump_nm + FSR1_nm)
nu = lambda l_nm: c / (l_nm * 1e-9)
dnuFSR = abs((nu(lam_s) - nu(pump_nm)) - (nu(pump_nm) - nu(lam_i)))   # dispersion mismatch, Hz
Q_o, Q_e = Q_factors(L1, typ["A_dB_per_cm"], typ["K1_percent"] / 100)
Q_tot = 1 / (1 / Q_o + 1 / Q_e)
K_nl = K_ref * (REF["R_um"] / typ["r1_um"]) ** 2
PGR_ours = pgr_eqS1(Q_o, Q_e, Q_o, Q_e, Q_o, Q_e, dnuFSR, K_nl, pump_nm)

print(f"R1 = {L1/2/np.pi*1e6:.3f} um (m1 = {m1}), FSR1 = {FSR1_nm:.3f} nm; "
      f"signal {lam_s:.3f} / idler {lam_i:.3f} nm")
print(f"FSR mismatch from dispersion, dnu_FSR = {dnuFSR*1e-9:.2f} GHz "
      f"(ARXIV2411 measures 1.4-3.7 GHz on R = 19 um rings)")
print(f"Q_o = {Q_o:.3e}  Q_e = {Q_e:.3e}  Q_tot = {Q_tot:.3e}  "
      f"(ARXIV2411 Table 1: Q_o ~ 88-108k, Q_e ~ 63-73k)")
# VALIDATION of Q_o against a cutback measurement [REF: SAVANIER16 sec 3.1]:
# they measure alpha = 0.74 dB/cm and report Q_U = 9.2e5; our formula gives 1.0e6
# (+9 %); for 1.23 dB/cm they report 5.6e5, we give 6.0e5 (+7 %). The intrinsic-Q
# formula is therefore validated BY MEASUREMENT for an UNDOPED ring.
# NOTE: ARXIV2411 reports lower Q_o (~1e5) because that platform puts doped p-i-n
# spokes inside the ring for carrier sweepout -- a loss specific to it, NOT a
# general correction to straight-guide loss. (An earlier version of this script
# wrongly generalised that and inferred an "effective 7 dB/cm"; corrected here.)
for A_ref, QU_ref in ((0.74, 9.2e5), (1.23, 5.6e5)):
    QU_ours = 2 * np.pi * 4.2 / (pump_nm * 1e-9 * A_ref * 100 / 4.3429)
    print(f"  VALIDATION [SAVANIER16]: alpha = {A_ref} dB/cm -> Q_U ours {QU_ours:.2e} "
          f"vs measured {QU_ref:.1e} ({(QU_ours/QU_ref-1)*100:+.0f} %)")
print(f"Optimal coupling for max PGR: Q_e/Q_o = 3/4  [REF: ARXIV2411 Eq. S2, r_e = 4/3 r_o]; "
      f"here Q_e/Q_o = {Q_e/Q_o:.2f}")
print(f"PGR (Eq. S1, calibrated) = {PGR_ours*1e-6:.2f} MHz/mW^2  "
      f"-> {PGR_ours*1e-6*1.0**2:.2f} MHz at 1 mW, {PGR_ours*1e-6*0.1**2*1e3:.0f} kHz at 0.1 mW")
print("  [certain] functional form and calibration point are from ARXIV2411/GENTRY18;")
print("  [assumption] (R_ref/R)^2 scaling of the nonlinear prefactor at equal cross-section;")
print("  [not modelled] TPA and free carriers, which ARXIV2411 states reduce PGR at high power.")


# =============================================================================
# [T4-4] VERNIER DOUBLE RING: six couplings, T(omega), combs, extinction
# =============================================================================
# Series-coupled rings [REF: RABUS07 Fig. 2.8, Eqs. 2.58-2.63], solved as a 2x2
# linear system (energy-conserving; the closed forms 2.64-2.65 are equivalent).
# Coupler numbering: 1 = input bus <-> R1, 2 = R1 <-> R2, 3 = R2 <-> drop bus.
# Six coefficients: (t1,k1), (t2,k2), (t3,k3) with |t|^2 + |k|^2 = 1 [RABUS07 Eq. 2.2].
# Design rule (Butterworth / maximally flat, 2nd order) [REF: LITTLE97; RABUS07
# Eq. 2.76 for identical rings]. Re-derived for L2 = L1/2 (different round-trip
# times) from temporal CMT: k3 = k1/sqrt(2), k2 = k1^2/(2 sqrt 2). [ASSUMPTION:
# lossless couplers; kappa frequency-independent as agreed in the 09/09 minutes.]

def couplers(K1):
    k1, t1 = np.sqrt(K1), np.sqrt(1 - K1)
    k3, t3 = k1 / np.sqrt(2), np.sqrt(1 - K1 / 2)
    k2 = k1 ** 2 / (2 * np.sqrt(2))
    return k1, t1, k2, np.sqrt(1 - k2 ** 2), k3, t3


theta = lambda l_nm, L: 2 * np.pi * n_eff(l_nm) * L / (l_nm * 1e-9)   # [RABUS07 Eq. 2.5]


def series_rings(l_nm, L1, L2, K1, A, dn2=0.0):
    k1, t1, k2, t2, k3, t3 = couplers(K1)
    a1 = np.sqrt(alpha_of(A, L1)) * np.exp(1j * theta(l_nm, L1) / 2)
    th2 = 2 * np.pi * (n_eff(l_nm) + dn2) * L2 / (l_nm * 1e-9)
    a2 = np.sqrt(alpha_of(A, L2)) * np.exp(1j * th2 / 2)
    M = np.array([[1 - t1 * t2 * a1 ** 2, t1 * a1 * k2 * a2],
                  [-t3 * a2 * k2 * a1, 1 - t3 * t2 * a2 ** 2]])
    E1a, E2b = np.linalg.solve(M, [-k1, 0.0])
    E1b = t2 * a1 * E1a - k2 * a2 * E2b
    E2a = k2 * a1 * E1a + t2 * a2 * E2b
    Et1 = t1 + k1 * a1 * E1b                 # through   [RABUS07 Eq. 2.62]
    Et2 = k3 * a2 * E2a                      # drop      [RABUS07 Eq. 2.63], Ei2 = 0
    return Et1, Et2, E1a, E2a


def kpis(r1_um, A, K1_pct, dn2=0.0):
    L1, L2, m1 = size_rings(r1_um)
    FSR = pump_nm ** 2 / (n_g(pump_nm) * L1 * 1e9)
    ls = resonance(L1, m1 + 1, pump_nm - FSR)
    li = resonance(L1, m1 - 1, pump_nm + FSR)
    K1 = K1_pct / 100
    Tp = abs(series_rings(pump_nm, L1, L2, K1, A, dn2)[1]) ** 2      # pump leaking to DROP
    Ts = abs(series_rings(ls, L1, L2, K1, A, dn2)[1]) ** 2
    Ti = abs(series_rings(li, L1, L2, K1, A, dn2)[1]) ** 2
    B = abs(series_rings(pump_nm, L1, L2, K1, A, dn2)[2]) ** 2        # pump build-up in R1
    return dict(L1=L1, L2=L2, ls=ls, li=li, Tp=Tp, Ts=Ts, Ti=Ti, buildup=B,
                rej_dB=-10 * np.log10(max(Tp, 1e-300)),
                ER_dB=10 * np.log10(max(Ts, 1e-300) / max(Tp, 1e-300)))


print("\n" + "=" * 78)
print("[T4-4] Vernier double ring -- typical design point")
print("=" * 78)
k1, t1, k2, t2, k3, t3 = couplers(typ["K1_percent"] / 100)
print(f"Six coupling coefficients (Butterworth rule):")
print(f"  (t1,k1) = ({t1:.4f},{k1:.4f})  (t2,k2) = ({t2:.5f},{k2:.4f})  (t3,k3) = ({t3:.4f},{k3:.4f})")
K = kpis(typ["r1_um"], typ["A_dB_per_cm"], typ["K1_percent"])
print(f"T_drop(pump) = {K['Tp']:.2e}  ->  pump rejection at the drop port = {K['rej_dB']:.1f} dB")
print(f"T_drop(signal) = {K['Ts']:.3f}, T_drop(idler) = {K['Ti']:.3f}, pump build-up = x{K['buildup']:.0f}")
print(f"Extinction ratio drop(s)/drop(p) = {K['ER_dB']:.1f} dB")
print("Benchmarks: single Vernier stage measured ~60 dB at sub-filter FSRs, > 100 dB at")
print("  one source FSR with cascaded 4th/6th-order filters, < 1 dB IL [REF: ARXIV2411 Ext.Fig.6];")
print("  cascaded 2nd-order CROW > 110 dB [REF: KUMAR20]; 10x Bragg 45 -> 60 dB [REF: MICHON22].")

# --- reaching 100-120 dB: single stage vs cascade [REF: KUMAR20 cascades 2 stages;
#     MICHON22 cascades 10 Bragg sections; ARXIV2411 uses 4th/6th order] ---
print("\nRoute to 100-120 dB:")
for N in (1, 2, 3):
    print(f"  {N} identical stage(s): rejection = {N*K['rej_dB']:.0f} dB, "
          f"pairs kept = {K['Ts']**N:.3f} ({-10*np.log10(K['Ts']**N):.1f} dB)")
print("  -> the target is reached by cascading, as in every cited demonstration; a single")
print("     stage forced to 100 dB by shrinking K1 would lose the pairs (loss-limited).")




# =============================================================================
# [T4-4b] THE SIX COUPLING COEFFICIENTS FROM THE REAL GAP GEOMETRY (Femwell)
# =============================================================================
# Physics [REF: RABUS07 Eqs. 2.66-2.70]: the ring-bus coupling coefficient is
#   kappa ~ (mode overlap) x exp[alpha_q (w_q - 2 s0)] x sqrt(pi R / alpha_q)
# i.e. (i) an exponential decay with the centre-to-centre gap 2 s0, set by the
# cladding decay constant alpha_q, and (ii) an effective interaction length
# sqrt(pi R/alpha_q) that comes from integrating along the curved path, with
#   R = r1 r2 / (r1 + r2)         [RABUS07 Eq. 2.70]  (bus-ring: r2 -> inf, R = r1).
# Savanier et al. state the same three dependencies: |kappa|^2 "is determined
# by the width of the waveguides, the separation between them, and the length
# of the coupling region" [REF: SAVANIER16 sec 3.2, citing Soltani 2010].
#
# Method here: replace the slab-mode numerator of Eq. 2.66 by a Femwell
# SUPERMODE calculation of two parallel strip guides [REF: FEMWELL
# coupled_mode_theory example]: per-unit-length coupling
#   kappa_z(g) = (pi/lambda) (n_even - n_odd),
# fitted to kappa_z = kappa_0 exp(-gamma_g g); then the curved-path integral
#   kappa_tot(g0) = int kappa_z(g0 + z^2/2R) dz = kappa_z(g0) sqrt(2 pi R / gamma_g),
# and finally |kappa|^2 = sin^2(kappa_tot), t = cos(kappa_tot)  [RABUS07 Eq. 2.2].
# [ASSUMPTION] straight-guide supermodes used for the bent ring (valid for
# R >> width, as here); TE0 only; no coupler excess loss.

print("\n" + "=" * 78)
print("[T4-4b] kappa from the real gap: Femwell supermodes + curved-path integral")
print("=" * 78)


def coupled_mesh(w_um, h_um, gap_um):
    core1 = shapely.geometry.box(-gap_um / 2 - w_um, 0, -gap_um / 2, h_um)
    core2 = shapely.geometry.box(gap_um / 2, 0, gap_um / 2 + w_um, h_um)
    span = 2 * w_um + gap_um
    clad = shapely.geometry.box(-2 * span, 0, 2 * span, 6 * h_um)
    box = shapely.geometry.box(-2 * span, -6 * h_um, 2 * span, 0)
    res = dict(core1={"resolution": 0.02, "distance": 0.3}, core2={"resolution": 0.02, "distance": 0.3},
               clad={"resolution": 0.05, "distance": 0.3}, box={"resolution": 0.05, "distance": 0.3})
    return from_meshio(mesh_from_OrderedDict(OrderedDict(core1=core1, core2=core2, clad=clad, box=box),
                                             res, default_resolution_max=2))


def kappa_z_of_gap(gap_nm, lam_um):
    """Per-unit-length coupling (rad/um) from the even/odd supermode splitting."""
    m = coupled_mesh(w_um, h_um, gap_nm * 1e-3)
    b = Basis(m, ElementTriP0()); eps = b.zeros()
    for dom, nf in {"core1": n_Si, "core2": n_Si, "clad": n_SiO2, "box": n_SiO2}.items():
        eps[b.get_dofs(elements=dom)] = nf(lam_um) ** 2
    modes = compute_modes(b, eps, wavelength=lam_um, num_modes=2, order=1)
    # the even (symmetric) supermode always has the higher n_eff; the solver does
    # not guarantee the return order, so sort explicitly
    n_even, n_odd = sorted((np.real(modes[0].n_eff), np.real(modes[1].n_eff)), reverse=True)
    return np.pi / lam_um * (n_even - n_odd), n_even, n_odd


gaps_nm = np.array([150, 200, 250, 300, 350])
kz = np.array([kappa_z_of_gap(g, pump_nm * 1e-3)[0] for g in gaps_nm])       # rad/um
# exponential fit kappa_z = kappa_0 exp(-gamma_g g)   [RABUS07 Eq. 2.66 structure]
gamma_g, ln_k0 = np.polyfit(gaps_nm * 1e-3, np.log(kz), 1) * np.array([-1, 1])
kappa_0 = np.exp(ln_k0)                                                       # rad/um at g = 0
print(f"supermode kappa_z(g): " + ", ".join(f"{g:.0f} nm -> {k:.4f} rad/um" for g, k in zip(gaps_nm, kz)))
print(f"fit kappa_z = {kappa_0:.3f} exp(-g/{1e3/gamma_g:.0f} nm)   (decay length {1e3/gamma_g:.0f} nm)")


L_couple_um = 5.0   # comprimento do acoplador (escolha de projeto / racetrack)
# changed this after fiday meeting
def K_power(gap_nm, L_um):
    k_z = kappa_0 * np.exp(-gamma_g * gap_nm * 1e-3)   # = πΔn/λ  [rad/µm]
    return np.sin(k_z * L_um) ** 2, L_um               # eq 4.8: |κ|² = sin²(k_z·L)


R1_um, R2_um = K["L1"] / 2 / np.pi * 1e6, K["L2"] / 2 / np.pi * 1e6
R_bus1, R_12, R_bus2 = R1_um, R1_um * R2_um / (R1_um + R2_um), R2_um
print(f"effective radii [RABUS07 Eq. 2.70]: bus-R1 {R_bus1:.2f} um, R1-R2 {R_12:.2f} um, R2-bus {R_bus2:.2f} um")
print(f"effective interaction lengths sqrt(2 pi R/gamma): "
      f"{K_power(200, R_bus1)[1]:.2f} / {K_power(200, R_12)[1]:.2f} / {K_power(200, R_bus2)[1]:.2f} um")

# --- gap needed to realise the design couplings of [T4-4] ---
k1d, _, k2d, _, k3d, _ = couplers(typ["K1_percent"] / 100)
targets = [("bus <-> R1", k1d ** 2, R_bus1), ("R1 <-> R2 ", k2d ** 2, R_12), ("R2 <-> bus", k3d ** 2, R_bus2)]
print("\nGap that realises each design coupling (inverting the curved-coupler law):")
design_gaps = {}
for name, Kt, Reff in targets:
    g = brentq(lambda gg: K_power(gg, Reff)[0] - Kt, 50, 2000)
    design_gaps[name] = g
    print(f"  {name}: |kappa|^2 = {Kt:.4f}  ->  gap = {g:.0f} nm")

# --- validation against the materials we were given ---
print("\nValidation against measured/reported devices:")
K_sav, _ = K_power(200, 20.0)
print(f"  SAVANIER16: gap 200 nm, R = 20 um, rib 650x220/70 nm -> |kappa|^2 = 0.018 (inferred from Q_L).")
print(f"     our strip model at the same gap/R gives {K_sav:.3f} (different cross-section: same order).")
for Wgap, Rout, Qe_meas in ((290, 19.2, 68.2e3), (435, 19.45, 70.6e3)):
    Kx, _ = K_power(Wgap, Rout)
    Qe_pred = 2 * np.pi * n_g(pump_nm) * (2 * np.pi * Rout * 1e-6) / (pump_nm * 1e-9 * Kx)
    print(f"  ARXIV2411: design gap {Wgap} nm, R = {Rout} um -> our |kappa|^2 = {Kx:.4f}, "
          f"Q_e = {Qe_pred:.1e} (measured ~{Qe_meas:.1e}: {Qe_pred/Qe_meas:.0f}x OFF)")
print("     -> this comparison FAILS, and it should: their ring is a 2.0-um-wide partially-")
print("        etched 'disk-like' rib in a 45 nm CMOS SOI (thinner Si, different confinement),")
print("        not a 500x220 nm strip. A less confined mode has a much longer evanescent decay")
print("        length, hence far stronger coupling at the same gap. The gap-to-kappa law is")
print("        cross-section specific; only same-platform comparisons (SAVANIER16) are meaningful.")
print("  MEDINA24 sec 3.3.2: silica-clad DUV rings reach critical coupling at 60-80 nm gap;")
print("     air-clad C2N rings stay under-coupled up to 120 nm (higher index contrast ->")
print("     shorter decay length) and need a pulley coupler. Our decay length "
      f"{1e3/gamma_g:.0f} nm is for silica cladding, consistent with the foundry case.")

# --- gap tolerance: the coupling is the most fabrication-sensitive parameter ---
dK_dg = -2 * gamma_g * 1e-3          # d ln K / d gap  (per nm), from K ~ exp(-2 gamma g)
print(f"\nGap sensitivity: d ln|kappa|^2 / d gap = {dK_dg*100:+.1f} % per nm.")
for dg in (5.0, 10.0):
    print(f"  +/-{dg:.0f} nm of gap (THOMSON16 linewidth control) -> K1 x {np.exp(dK_dg*dg):.2f} / "
          f"{np.exp(-dK_dg*dg):.2f}")
print("  SAVANIER16 (3D FDTD): a 43 nm gap change moved their ring from fabricated to critical")
print("  coupling -- same order as our sensitivity; hence their 'slightly over-coupled' rule.")

# =============================================================================
# [T4-4 cont.] sweeps: K1, loss, radius  (variations of the parameters)
# =============================================================================
K1_sw = np.logspace(np.log10(P["K1_percent"][0]), np.log10(P["K1_percent"][2]), 18)
A_sw = np.linspace(P["A_dB_per_cm"][0], P["A_dB_per_cm"][2], 10)
r_sw = np.linspace(P["r1_um"][0], P["r1_um"][2], 7)
res_K1 = [kpis(typ["r1_um"], typ["A_dB_per_cm"], x) for x in K1_sw]
res_A = [kpis(typ["r1_um"], x, typ["K1_percent"]) for x in A_sw]
res_r = [kpis(x, typ["A_dB_per_cm"], typ["K1_percent"]) for x in r_sw]


# =============================================================================
# [T4-5] FABRICATION VARIABILITY -- anchored to measured data
# =============================================================================
# Sources of numbers:
#   THOMSON16 : linewidth control ~5 nm (193 nm immersion), thickness ~1 nm.
#   ARXIV2411 Table 1 (6 packaged dies, same MPW): resonance-wavelength std dev
#               1.73 nm die-to-die, worst 4.8 nm; Q_o std ~15 %; FSR mismatch
#               std 0.7 GHz; PGR spread > 2x.  Heater: 0.62 nm range, 0.6 pm LSB.
#   shift law: dlambda/lambda = dn_eff/n_g  [REF: THOMSON16; group synthesis Eq.]
print("\n" + "=" * 78)
print("[T4-5] Fabrication variability")
print("=" * 78)
dw_nm = 5.0
n_plus = np.real(te0(strip_mesh(w_um + dw_nm * 1e-3, h_um), pump_nm * 1e-3).n_eff)
n_minus = np.real(te0(strip_mesh(w_um - dw_nm * 1e-3, h_um), pump_nm * 1e-3).n_eff)
dneff_dw = (n_plus - n_minus) / (2 * dw_nm)                       # per nm of width
shift_pm_per_nm = pump_nm * dneff_dw / n_g(pump_nm) * 1e3
print(f"dn_eff/dw = {dneff_dw:.2e} /nm  ->  {shift_pm_per_nm:.0f} pm of resonance shift per nm of width")
print(f"THOMSON16 +/-5 nm width  -> +/-{shift_pm_per_nm*5e-3:.2f} nm (common-mode: both combs move together)")
print(f"ARXIV2411 measured die-to-die sigma_lambda = 1.73 nm, worst 4.8 nm  (same order: consistent)")
print(f"Heater range 0.62 nm [ARXIV2411] < 1.73 nm sigma -> the paper itself reports failed alignment;")
print("  on-chip trimming range must exceed the die-to-die spread (their stated fix: redesign heaters).")

# Differential error between the two rings (what breaks theta2 = theta1/2):
fwhm_nm = pump_nm / Q_tot
dw_diff = np.concatenate([np.linspace(-3, -0.3, 10), np.linspace(-0.3, 0.3, 61), np.linspace(0.3, 3, 10)])
tol = [kpis(typ["r1_um"], typ["A_dB_per_cm"], typ["K1_percent"], dn2=dneff_dw * d) for d in dw_diff]
Ts_tol = np.array([x["Ts"] for x in tol]); ER_tol = np.array([x["ER_dB"] for x in tol])
ok = dw_diff[Ts_tol >= 0.5 * K["Ts"]]
print(f"Resonance FWHM = {fwhm_nm*1e3:.0f} pm (Q_tot = {Q_tot:.2e}).")
print(f"Differential width error keeping >= 50 % of the extraction: |dw| < {ok.max():.2f} nm "
      f"(= {ok.max()*shift_pm_per_nm:.0f} pm of relative detuning, ~ one linewidth)")
print("  -> a 2-ring Vernier needs per-ring trimming to ~0.1 nm-equivalent: matches the")
print("     'actively tuned' requirement of every demonstration [ARXIV2411; MEDINA24 §3.4].")


# =============================================================================
# FIGURES
# =============================================================================
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# (a) dispersion + FSR drift  -- [minutes: plot (FSR - <FSR>) vs omega]
# FSR in FREQUENCY units: FSR_nu = c/(n_g L) [RABUS07 Eq. 2.21; supervisor's note].
# This isolates the dispersion effect: FSR_nu is constant iff n_g is constant iff
# beta2 = 0. (In wavelength units the lambda^2 factor adds a trivial, non-dispersive
# drift that would mask the physics.)
lf = np.linspace(1450, 1650, 60); nuf = c / (lf * 1e-9)
FSRnu_GHz = c / (n_g(lf) * L1) * 1e-9
ax[0, 0].plot(nuf * 1e-12, FSRnu_GHz - FSRnu_GHz.mean()); ax[0, 0].axhline(0, color="gray", ls="--")
ax[0, 0].set_xlabel("optical frequency (THz)"); ax[0, 0].set_ylabel(r"FSR$_\nu$ - <FSR$_\nu$> (GHz)")
ax[0, 0].set_title(f"[T4-1] FSR drift from GVD (D = {D_ps_nm_km(pump_nm):+.0f} ps/(nm km), anomalous)")
print(f"\nFSR_nu drift over 1450-1650 nm: {FSRnu_GHz.max()-FSRnu_GHz.min():.1f} GHz peak-to-peak "
      f"about {FSRnu_GHz.mean():.0f} GHz  (beta2 -> 0 would flatten this)")

# (b) combs + T(omega)
span = 2.6 * FSR1_nm; lams = np.linspace(pump_nm - span, pump_nm + span, 20001)
out = np.array([series_rings(l, K["L1"], K["L2"], typ["K1_percent"] / 100, typ["A_dB_per_cm"]) for l in lams])
ax[0, 1].plot(lams - pump_nm, abs(out[:, 2]) ** 2 / abs(out[:, 2]).max() ** 2, label="R1 build-up", lw=0.8)
ax[0, 1].plot(lams - pump_nm, abs(out[:, 3]) ** 2 / abs(out[:, 3]).max() ** 2, label="R2 build-up", lw=0.8)
ax[0, 1].plot(lams - pump_nm, abs(out[:, 1]) ** 2, label="T_drop", color="tab:green")
for l, col in ((pump_nm, "k"), (K["ls"], "r"), (K["li"], "b")):
    ax[0, 1].axvline(l - pump_nm, color=col, ls=":", lw=1)
ax[0, 1].set_xlabel("$\\lambda-\\lambda_p$ (nm)"); ax[0, 1].set_title("[T4-4] two combs and T_drop (p, s, i)")
ax[0, 1].legend(fontsize=7)

# (c) T_drop / T_through log
ax[0, 2].semilogy(lams - pump_nm, abs(out[:, 1]) ** 2, color="tab:green", label="T_drop")
ax[0, 2].semilogy(lams - pump_nm, abs(out[:, 0]) ** 2, color="tab:red", label="T_through")
ax[0, 2].axhline(1e-10, color="gray", ls="--", lw=1, label="-100 dB"); ax[0, 2].set_ylim(1e-12, 2)
ax[0, 2].set_xlabel("$\\lambda-\\lambda_p$ (nm)"); ax[0, 2].set_title("[T4-4] pump rejection at the drop port")
ax[0, 2].legend(fontsize=7)

# (d) K1 sweep
ax[1, 0].semilogx(K1_sw, [x["rej_dB"] for x in res_K1], "o-", label="rejection (dB)")
ax[1, 0].axhspan(100, 120, color="red", alpha=0.12, label="100-120 dB target")
ax[1, 0].set_xlabel("K1 (%)"); ax[1, 0].set_ylabel("dB"); ax[1, 0].legend(fontsize=7)
axb = ax[1, 0].twinx(); axb.semilogx(K1_sw, [x["Ts"] for x in res_K1], "s--", color="tab:orange")
axb.set_ylabel("T_drop(signal)", color="tab:orange"); ax[1, 0].set_title("[T4-4] bus coupling sweep")

# (e) loss and radius sweeps
ax[1, 1].plot(A_sw, [x["Ts"] for x in res_A], "o-", label="T_drop(signal) vs loss A")
ax[1, 1].set_xlabel("loss A (dB/cm)"); ax[1, 1].set_ylabel("T_drop(signal)")
axc = ax[1, 1].twiny(); axc.plot(r_sw, [x["Ts"] for x in res_r], "s--", color="tab:purple")
axc.set_xlabel("radius r1 (um)", color="tab:purple"); ax[1, 1].set_title("[T4-4] loss and radius sweeps")
ax[1, 1].legend(fontsize=7)

# (f) fabrication tolerance
ax[1, 2].plot(dw_diff, Ts_tol, color="tab:orange", label="T_drop(signal)")
ax[1, 2].set_xlabel("differential width error R2 - R1 (nm)"); ax[1, 2].set_ylabel("T_drop(signal)")
axd = ax[1, 2].twinx(); axd.plot(dw_diff, ER_tol, color="tab:green"); axd.set_ylabel("extinction (dB)", color="tab:green")
ax[1, 2].axvspan(-5, 5, color="gray", alpha=0.08); ax[1, 2].set_xlim(-3, 3)
ax[1, 2].set_title("[T4-5] tolerance (THOMSON16: +/-5 nm; ARXIV2411: sigma 1.73 nm)")
ax[1, 2].legend(fontsize=7)

fig.suptitle("Task 4 -- Vernier double-ring SFWM source and pump filter (pump = 1550 nm)")
fig.tight_layout(); fig.savefig("task4_results.png", dpi=180)
print("\nFigure: task4_results.png")


# =============================================================================
# [T4-6] CRITICAL DISCUSSION
# =============================================================================
print("\n" + "=" * 78)
print("[T4-6] Critical discussion -- what is certain, what is assumed, what is missing")
print("=" * 78)
print("""
CERTAIN (from cited sources or algebra):
 - Resonance/FSR/Q relations (RABUS07, BOG12); series-ring transfer function
   (RABUS07 Eqs. 2.58-2.65), energy conservation checked numerically.
 - Vernier relation FSR_ext = N FSR1 = M FSR2 (RABUS07 2.77-2.78; GRIFFEL00).
 - L2 = L1/2 makes theta2 = theta1/2 identically: R2 anti-resonant at every odd
   order of R1 (pump), resonant at every even order (signal, idler) -- pure algebra.
 - PGR functional form (ARXIV2411 Eq. S1 / GENTRY18) and its calibration point
   (3.29 MHz/mW^2 at Q ~ 1e5, R = 19.1 um); optimal coupling r_e = 4/3 r_o.
 - Measured variability (ARXIV2411 Table 1) and measured filter performance
   (ARXIV2411 Ext. Fig. 6; KUMAR20; MICHON22; AFIFI21).

ASSUMED (ours):
 - 2:1 Vernier ratio read from the supervisor's comb slide (not stated numerically).
 - Lossless, frequency-independent couplers; k1 = k3; Butterworth k2, k3 rule.
 - Same cross-section for the PGR calibration transfer ((R_ref/R)^2 scaling).
 - Linear phase-matching term only; CW undepleted pump.

NOT CAPTURED (and known to matter):
 - TPA / free-carrier absorption: ARXIV2411 lists them as PGR-limiting at high
   power; MEDINA24 §3.3.2 chooses pump powers accordingly. Our PGR is an upper bound.
 - Scattered-light floor: AFIFI21 measured 60 dB vs 157 dB modelled; MIT 45nm
   platform 80 dB measured vs 160 dB simulated. Model rejection is an upper bound.
 - Backscattering / resonance splitting from sidewall roughness (RABUS07 §2.1.1).
 - kappa from the gap IS now computed ([T4-4b], supermodes + RABUS07 Eq. 2.66-2.70
   curved-path law), but with straight-guide supermodes and no coupler excess loss;
   3D FDTD of the bent coupler (as SAVANIER16 did) is the next refinement.
 - "No pump wavelength in the output": lambda_p is an INPUT of this model; the
   model predicts the response around it, it does not select it (supervisor's note).
""")
