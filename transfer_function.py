import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.interpolate import UnivariateSpline
from skfem import Basis, ElementTriP0
from skfem.io import from_meshio
from tqdm import tqdm

from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

c = 299792458.0  # m/s


# =============================================================================
# 1. Waveguide and Femwell mode solver
# =============================================================================
width = 0.50   # um
height = 0.22  # um
pump_nm = 1550.0
wavelength_range = [1400, 1700]  # nm
wavelength_step = 30


def build_mesh(width, height):
    core = shapely.geometry.box(-width / 2, 0, +width / 2, height)
    cladding = shapely.geometry.box(-width * 4, 0, width * 4, height * 6)
    buried_oxide = shapely.geometry.box(-width * 4, -height * 6, width * 4, 0)
    polygon = OrderedDict(core=core, cladding=cladding, buried_oxide=buried_oxide)
    resolutions = dict(
        core={"resolution": 0.02, "distance": 0.3},
        cladding={"resolution": 0.05, "distance": 0.3},
        buried_oxide={"resolution": 0.05, "distance": 0.3},
    )
    return from_meshio(mesh_from_OrderedDict(polygon, resolutions, default_resolution_max=2))


# Use sellmeier equation to determine the refractive index of material
def n_Si(wavelength):
    # https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg
    if (1.357 <= wavelength <= 11.04):
        l2 = wavelength ** 2
        return math.sqrt(
            10.6684293 * l2 / (l2 - 0.301516485 ** 2)
            + 0.0030434748 * l2 / (l2 - 1.13475115 ** 2)
            + 1.54133408 * l2 / (l2 - 1104 ** 2) 
            + 1)
    else:
        raise ValueError(f"wavelength provided is {wavelength}um, is out of the range for Si")


def n_SiO2(wavelength):
    if wavelength < 0.21 or wavelength > 6.7:
        raise ValueError(f"wavelength {wavelength} um out of range for SiO2")
    l2 = wavelength ** 2
    return np.sqrt(0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                   + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                   + 0.8974794 * l2 / (l2 - 9.896161 ** 2) + 1)


n_dict = {"core": n_Si, "cladding": n_SiO2, "buried_oxide": n_SiO2}

print("=" * 70)
print("1. Femwell sweep: n_eff(lambda) for the TE0 mode")
print("=" * 70)
mesh = build_mesh(width, height)
wavelength_list = np.linspace(wavelength_range[0], wavelength_range[1], wavelength_step)
neff_list = []
basis0 = Basis(mesh, ElementTriP0())
epsilon = basis0.zeros()
for wl in tqdm(wavelength_list):
    wl_um = wl * 1e-3
    for subdomain, n in n_dict.items():
        epsilon[basis0.get_dofs(elements=subdomain)] = n(wl_um) ** 2
    modes = compute_modes(basis0, epsilon, wavelength=wl_um, num_modes=3, order=1)
    mode = modes.sorted(key=lambda m: -np.real(m.te_fraction))[0]
    neff_list.append(np.real(mode.n_eff))
neff_list = np.array(neff_list)

print(f"\nWaveguide: SOI strip {width*1e3:.0f} x {height*1e3:.0f} nm")
print(f"n_eff at {pump_nm:.0f} nm (TE0, from Femwell) = "
      f"{np.interp(pump_nm, wavelength_list, neff_list):.4f}")

# =============================================================================
# 2. Dispersion: n_eff(lambda) spline -> n_g, FSR   (Bogaerts 2012, Eqs. 2-4)
# =============================================================================
neff_spl = UnivariateSpline(wavelength_list, neff_list, s=0, k=3)
dneff_spl = neff_spl.derivative(1)


def n_eff(lam_nm):
    return neff_spl(lam_nm)


def n_g(lam_nm):
    return neff_spl(lam_nm) - lam_nm * dneff_spl(lam_nm)


print("\n" + "=" * 70)
print("2. Dispersion")
print("=" * 70)
print(f"n_eff({pump_nm:.0f} nm) = {n_eff(pump_nm):.5f}")
print(f"n_g({pump_nm:.0f} nm)   = {n_g(pump_nm):.4f}")

# =============================================================================
# 3. Ring 1 (generator) and Ring 2 (Vernier filter, L2 = L1/2)
# =============================================================================
# Ring 1: sized to be exactly resonant at the pump.
r1_target = 10.0e-6  # m -- a round design value, not yet optimised
m1 = round(n_eff(pump_nm) * 2 * np.pi * r1_target / (pump_nm * 1e-9))
if m1 % 2 == 0:
    m1 += 1  # force an odd order at the pump (see point 4 below)
L1 = m1 * pump_nm * 1e-9 / n_eff(pump_nm)
FSR1_nm = pump_nm ** 2 / (n_g(pump_nm) * L1 * 1e9)

# Ring 2: EXACTLY half the perimeter of ring 1 -- this is the Vernier
# idea read off the professor's slide.
L2 = L1 / 2

# resonance condition: theta(lam)
def theta(lam_nm, L):
    return 2 * np.pi * n_eff(lam_nm) * L / (lam_nm * 1e-9)


print("\n" + "=" * 70)
print("3. Ring sizing and the Vernier check (L2 = L1/2)")
print("=" * 70)
print(f"Ring 1 (generator): L1 = {L1*1e6:.3f} um (r1 = {L1/2/np.pi*1e6:.3f} um), "
      f"pump order m1 = {m1}, FSR1 = {FSR1_nm:.3f} nm")
print(f"Ring 2 (filter)   : L2 = L1/2 = {L2*1e6:.3f} um (r2 = {L2/2/np.pi*1e6:.3f} um)")

lam_s = pump_nm - FSR1_nm   # signal: neighbouring resonance of ring 1 (even order)
lam_i = pump_nm + FSR1_nm   # idler : neighbouring resonance of ring 1 (even order)

print("\nQuantitative check -- fractional part of theta2/2pi (0 or 1 = resonant, "
      "0.5 = anti-resonant):")
for name, lam in (("pump", pump_nm), ("signal", lam_s), ("idler", lam_i)):
    frac = (theta(lam, L2) / (2 * np.pi)) % 1.0
    state = "RESONANT" if min(frac, 1 - frac) < 0.05 else "ANTI-RESONANT" if abs(frac - 0.5) < 0.05 else "detuned"
    print(f"  {name:6s} (lambda = {lam:8.3f} nm): theta2/2pi mod 1 = {frac:.3f}  -> {state}")


# =============================================================================
# 4. Transfer function T(omega): two rings coupled in series
# =============================================================================
# a_i = sqrt(alpha_Ri) * exp(j*theta_i/2)  (half round trip, coupler to coupler)

K1 = 0.05                                   # bus power coupling (placeholder)
kappa1, t1 = np.sqrt(K1), np.sqrt(1 - K1)
kappa3, t3 = kappa1, t1                     # symmetric design, placeholder
kappa2, t2 = 0.01, np.sqrt(1 - 0.01 ** 2)   # inter-ring coupling (placeholder, weak
                                             # enough to avoid visible mode splitting

loss_dB_per_cm = 2.0                        # order of magnitude (meeting notes point 4)
alpha_R1 = 10 ** (-loss_dB_per_cm * (L1 * 100) / 20)
alpha_R2 = 10 ** (-loss_dB_per_cm * (L2 * 100) / 20)

print("\n" + "=" * 70)
print("4. Transfer function T(omega) -- Eqs. (2.58)-(2.63)")
print("=" * 70)
print(f"Coupling coefficients used (PLACEHOLDER, not yet from the gap geometry):")
print(f"  t1 = {t1:.3f}, k1 = {kappa1:.3f} | t2 = {t2:.3f}, k2 = {kappa2:.3f} | "
      f"t3 = {t3:.3f}, k3 = {kappa3:.3f}")
print(f"Loss: {loss_dB_per_cm} dB/cm (order of magnitude) -> "
      f"alpha_R1 = {alpha_R1:.5f}, alpha_R2 = {alpha_R2:.5f}")


def series_rings(lam_nm):
    """Solve Eqs. (2.58)-(2.63) as a 2x2 linear system for the through
    (Et1) and drop (Et2) amplitudes, with the pump entering at Ei1."""
    a1 = np.sqrt(alpha_R1) * np.exp(1j * theta(lam_nm, L1) / 2)
    a2 = np.sqrt(alpha_R2) * np.exp(1j * theta(lam_nm, L2) / 2)
    A = np.array([[1 - t1 * t2 * a1 ** 2, t1 * a1 * kappa2 * a2],
                  [-t3 * a2 * kappa2 * a1, 1 - t3 * t2 * a2 ** 2]])
    E1a, E2b = np.linalg.solve(A, [-kappa1, 0.0])
    E1b = t2 * a1 * E1a - kappa2 * a2 * E2b
    E2a = kappa2 * a1 * E1a + t2 * a2 * E2b
    Et1 = t1 * 1.0 + kappa1 * a1 * E1b          # through port
    Et2 = t3 * 0.0 + kappa3 * a2 * E2a          # drop port
    return Et1, Et2


span = 2.5 * FSR1_nm
lams = np.linspace(pump_nm - span, pump_nm + span, 40000)
T_through, T_drop = [], []
for lam in lams:
    Et1, Et2 = series_rings(lam)
    T_through.append(abs(Et1) ** 2)
    T_drop.append(abs(Et2) ** 2)
T_through, T_drop = np.array(T_through), np.array(T_drop)

Tp_through = np.interp(pump_nm, lams, T_through)
Ts_drop = np.interp(lam_s, lams, T_drop)
print(f"\nAt the pump  : T_through = {Tp_through:.3f}  (energy conservation check: "
      f"T_through + T_drop = {Tp_through + np.interp(pump_nm, lams, T_drop):.3f} <= 1)")
print(f"At the signal: T_drop    = {Ts_drop:.3f}")

# =============================================================================
# 5. Figure: the ring-1 comb, the ring-2 comb, and T(omega)
# =============================================================================
def single_ring_comb(lams_nm, L, t, alpha):
    return (1 - t ** 2) / np.abs(1 - t * alpha * np.exp(1j * theta(lams_nm, L))) ** 2


comb1 = single_ring_comb(lams, L1, t1, alpha_R1)
comb2 = single_ring_comb(lams, L2, t3, alpha_R2)

fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
x = lams - pump_nm

axs[0].plot(x, comb1 / comb1.max(), color="tab:blue")
axs[0].set_ylabel("ring 1\n(generator)")
axs[0].set_title("The two frequency combs and the transfer function T(omega)")

axs[1].plot(x, comb2 / comb2.max(), color="tab:orange")
axs[1].set_ylabel("ring 2\n(filter, L1/2)")

axs[2].plot(x, T_through, color="tab:red", label="T_through")
axs[2].plot(x, T_drop, color="tab:green", label="T_drop")
axs[2].set_ylabel("transmission")
axs[2].set_xlabel(r"$\lambda - \lambda_{pump}$ (nm)")
axs[2].legend()

for ax in axs:
    ax.axvline(0, color="green", ls="--", lw=1)
    ax.axvline(lam_s - pump_nm, color="red", ls=":", lw=1)
    ax.axvline(lam_i - pump_nm, color="blue", ls=":", lw=1)
    ax.grid(alpha=0.3)

axs[0].text(0, 1.05, "p", color="green", ha="center", transform=axs[0].get_xaxis_transform())
axs[0].text(lam_s - pump_nm, 1.05, "s", color="red", ha="center", transform=axs[0].get_xaxis_transform())
axs[0].text(lam_i - pump_nm, 1.05, "i", color="blue", ha="center", transform=axs[0].get_xaxis_transform())

plt.tight_layout()
plt.savefig("combs_and_transfer_function.png", dpi=200)
