import numpy as np
from joblib import Parallel, delayed

n_jobs = 10

def epsilon_3D(kx,ky,kz,a,b):
    en = -0.5*(np.cos(kx)+a*np.cos(ky)+b*np.cos(kz))

    return en

def epsilon_3D_parallel(kx,ky,kz,a,b,n_jobs=n_jobs):
    en = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(epsilon_3D)(kx,ky,ai,b)
        for ai in a
    )

    en = np.array(en)

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


def dos_parallel(E_grid, epsilon, de, n_jobs=n_jobs):
    rho = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(dos_single_E)(Ei, epsilon, de)
        for Ei in E_grid
    )

    rho = np.array(rho)

    rho = rho / len(epsilon)
    rho = rho / np.trapezoid(rho, E_grid)

    return rho

if __name__ == "__main__":

    Ne = 200
    de = 1e-2
    kx = np.linspace(-np.pi,np.pi,Ne)
    ky = kx
    kz = kx
    X,Y,Z = np.meshgrid(kx,ky,kz)

    #e = epsilon_3D(X,Y,Z).ravel()
    #e_min = np.min(e)
    #e_max = np.max(e)

    #E_grid = np.linspace(e_min, e_max, 1000)
    de_values = np.linspace(0.005, 0.1, 20)
    a_values = np.linspace(0,2,10)
    b_values = np.linspace(0,2,10)

    A = []
    B = []

    for a in a_values:
        for b in b_values:
            e = epsilon_3D(X,Y,Z,a,b).ravel()
            e_min = np.min(e)
            e_max = np.max(e)
            E_grid = np.linspace(e_min, e_max, 200)

            for de in de_values:
                dos_values = dos_parallel(E_grid, e, de)

                for E, rho in zip(E_grid, dos_values):
                    A.append([E, de, a, b])
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

