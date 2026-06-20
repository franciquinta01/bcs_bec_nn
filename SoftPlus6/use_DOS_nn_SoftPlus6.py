import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_DOS import epsilon_3D, dos_parallel

class DOSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_dos_model(path="dos_net_SoftPlus6.pt"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model = DOSNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_mean = checkpoint["X_mean"]
    X_std = checkpoint["X_std"]

    return model, X_mean, X_std


def dos_from_nn(E_grid, de, a, b, model, X_mean, X_std):
    X = np.column_stack([
        E_grid,
        np.full_like(E_grid, de),
        np.full_like(E_grid, a),
        np.full_like(E_grid, b)
    ])

    X_norm = (X - X_mean) / X_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        y_log = model(X_tensor).numpy().flatten()

    rho = np.exp(y_log)

    e_min = -0.5 * (1 + abs(a) + abs(b))
    e_max =  0.5 * (1 + abs(a) + abs(b))

    rho[(E_grid < e_min) | (E_grid > e_max)] = 0.0

    return rho


model, X_mean, X_std = load_dos_model("dos_net_SoftPlus6.pt")

data = np.load("/home/fquintavalle/bcs_bec_nn/dos_dataset.npz")
E_grid = data["E_grid"]
epsilon = data["e"]

a = 1
b = 1
de = 0.0137

rho_nn = dos_from_nn(E_grid, de, a, b, model, X_mean, X_std)

Ne = 200
kx = np.linspace(-np.pi,np.pi,Ne)
ky = kx
kz = kx
X,Y,Z = np.meshgrid(kx,ky,kz)

e = epsilon_3D(X,Y,Z,a,b).ravel()

rho_num = dos_parallel(E_grid, e, de)

plt.figure(figsize=(7, 5))
plt.plot(E_grid, rho_num, label="Numeric DOS")
plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.xlabel("E")
plt.ylabel("DOS(E)")
plt.title(fr"Comparison between numerical and NN for $N_k = 200, de={de}$")
plt.legend()
plt.text(
    0.98, 0.95,
    f"a = {a:.3f}\nb = {b:.3f}",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)
plt.grid(True)
plt.tight_layout()
plt.savefig("images/DOS_nn.png",dpi=300)

#==============================================================
# Comparison with finer BZ
#==============================================================

Ne = 300
kx = np.linspace(-np.pi,np.pi,Ne)
ky = kx
kz = kx
X,Y,Z = np.meshgrid(kx,ky,kz)

e = epsilon_3D(X,Y,Z,a,b).ravel()

rho_num = dos_parallel(E_grid, e, de)

plt.figure(figsize=(7, 5))
plt.plot(E_grid, rho_num, label="Numeric DOS")
plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.xlabel("E")
plt.ylabel("DOS(E)")
plt.title(fr"Comparison between numerical and NN for $N_k = {Ne}, de={de}$")
plt.legend()
plt.text(
    0.98, 0.95,
    f"a = {a:.3f}\nb = {b:.3f}",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)
plt.grid(True)
plt.tight_layout()
plt.savefig("images/DOS_nn_finer.png",dpi=300)
