import numpy as np
from joblib import Parallel, delayed

def epsilon_3D(kx,ky,kz):
    en = -0.5*(np.cos(kx)+np.cos(ky)+np.cos(kz))

    return en

def dos(E, epsilon, de):
    dos = np.zeros(len(E))

    for i, Ei in enumerate(E):
        diff = Ei - epsilon
        dos[i] = np.sum(de / (diff**2 + de**2))

    dos = dos / len(epsilon)
    dos = dos / np.trapz(dos, E)

    return dos

def dos_single_E(Ei, epsilon, de):
    diff = Ei - epsilon
    return np.sum(de / (diff**2 + de**2))


def dos_parallel(E_grid, epsilon, de, n_jobs=8):
    rho = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(dos_single_E)(Ei, epsilon, de)
        for Ei in E_grid
    )

    rho = np.array(rho)

    rho = rho / len(epsilon)
    rho = rho / np.trapezoid(rho, E_grid)

    return rho


Ne = 100
de = 1e-2
kx = np.linspace(-np.pi,np.pi,Ne)
ky = kx
kz = kx
X,Y,Z = np.meshgrid(kx,ky,kz)

e = epsilon_3D(X,Y,Z).ravel()
e_min = np.min(e)
e_max = np.max(e)

E_grid = np.linspace(e_min, e_max, 1000)
de_values = np.linspace(0.005, 0.1, 50)

A = []
B = []

for de in de_values:
    dos_values = dos(E_grid, e, de)

    for E, rho in zip(E_grid, dos_values):
        A.append([E, de])
        B.append([rho])

A = np.array(A)
B = np.array(B)


np.savez_compressed(
    "dos_dataset.npz",
    A=A,
    B=B,
    e=e,
    E_grid=E_grid,
    de_values=de_values
)

