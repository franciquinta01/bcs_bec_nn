import numpy as np
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from tqdm import tqdm


# ============================================================
# Settings
# ============================================================

n_jobs = 15

Ne = 200
E_grid = np.linspace(-1.5, 1.5, 1000)

a_values = np.linspace(0.2, 1.0, 5)
b_values = np.linspace(0.2, 1.0, 5)
de_values = np.array([0.002])

n_values = np.linspace(0.05, 1.0, 50)
V_values = np.linspace(0.1, 5.0, 100)
omega_values = np.linspace(0.1, 2.0, 50)

d = 0.002


# ============================================================
# BCS-BEC equations
# ============================================================

def sigmoid(epsilon, mu, d):
    return 1.0 / (1.0 + np.exp((epsilon - mu) / d))


def F(x, epsilon, DOS, omega0, V0, n, d):
    delta = x[0]
    mu = x[1]

    dE = epsilon[1] - epsilon[0]

    t = sigmoid(np.abs(epsilon - mu), omega0, d)

    Ek = np.sqrt((epsilon - mu)**2 + delta**2 * t**2)

    fac = 1.0 / (2.0 * Ek)

    v2 = 0.5 * (1.0 - (epsilon - mu) / Ek)

    eq_gap = 1.0 - dE * V0 * np.sum(DOS * t**2 * fac)
    eq_num = n - dE * np.sum(DOS * v2)

    return np.array([eq_gap, eq_num])


def return_sol(epsilon, DOS, omega0, V0, n, d, x0):
    sol, info, ier, msg = fsolve(
        F,
        x0,
        args=(epsilon, DOS, omega0, V0, n, d),
        full_output=True,
        xtol=1e-10,
        maxfev=1000
    )

    delta_sol = sol[0]
    mu_sol = sol[1]

    converged = ier == 1

    return delta_sol, mu_sol, converged, msg


# ============================================================
# Dispersion and DOS
# ============================================================

def epsilon_3D(kx, ky, kz, a, b):
    return -0.5 * (np.cos(kx) + a * np.cos(ky) + b * np.cos(kz))


def dos_single_E(Ei, epsilon, de):
    diff = Ei - epsilon
    return np.sum(de / (diff**2 + de**2))


def dos_parallel(E_grid, epsilon, de, n_jobs):
    rho = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(dos_single_E)(Ei, epsilon, de)
        for Ei in E_grid
    )

    rho = np.array(rho)

    rho = rho / len(epsilon)
    rho = rho / np.trapezoid(rho, E_grid)

    return rho


# ============================================================
# k-grid
# ============================================================

k = np.linspace(-np.pi, np.pi, Ne)

kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")

kx = kx.ravel()
ky = ky.ravel()
kz = kz.ravel()


# ============================================================
# Dataset generation
# ============================================================

X_list = []
Y_list = []
failed = []

total = (
    len(a_values)
    * len(b_values)
    * len(de_values)
    * len(n_values)
    * len(V_values)
    * len(omega_values)
)

pbar = tqdm(total=total)

for a in a_values:
    for b in b_values:

        eps_k = epsilon_3D(kx, ky, kz, a, b)

        for de in de_values:

            DOS = dos_parallel(E_grid, eps_k, de, n_jobs=n_jobs)

            for n in n_values:

                x0 = np.array([0.8, 0.0])

                for V0 in V_values:
                    for omega0 in omega_values:

                        delta, mu, converged, msg = return_sol(
                            E_grid,
                            DOS,
                            omega0,
                            V0,
                            n,
                            d,
                            x0
                        )

                        if (
                            converged
                            and np.isfinite(delta)
                            and np.isfinite(mu)
                            and delta > 0
                        ):
                            X_list.append([a, b, de, n, V0, omega0])
                            Y_list.append([delta, mu])

                            x0 = np.array([delta, mu])

                        else:
                            failed.append([a, b, de, n, V0, omega0, msg])

                        pbar.update(1)

pbar.close()


# ============================================================
# Save
# ============================================================

X = np.array(X_list, dtype=np.float64)
Y = np.array(Y_list, dtype=np.float64)
failed = np.array(failed, dtype=object)

np.savez(
    "gap_mu_dataset.npz",
    X=X,
    Y=Y,
    failed=failed,
    a_values=a_values,
    b_values=b_values,
    de_values=de_values,
    n_values=n_values,
    V_values=V_values,
    omega_values=omega_values,
    E_grid=E_grid,
    d=d
)

print("Saved gap_mu_dataset.npz")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("failed:", len(failed))
