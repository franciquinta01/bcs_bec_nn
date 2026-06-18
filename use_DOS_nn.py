import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset_DOS import epsilon_3D, dos_parallel

class DOSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_dos_model(path="dos_net.pt"):
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

    return rho


model, X_mean, X_std = load_dos_model("dos_net.pt")

data = np.load("dos_dataset.npz")
E_grid = data["E_grid"]
epsilon = data["e"]

de = 2e-2
a = 0.5
b = 0.2

rho_nn = dos_from_nn(E_grid, de, a, b, model, X_mean, X_std)

Ne = 200
kx = np.linspace(-np.pi,np.pi,Ne)
ky = kx
kz = kx
X,Y,Z = np.meshgrid(kx,ky,kz)

e = epsilon_3D(X,Y,Z,a,b).ravel()

rho_num = dos_parallel(E_grid, e, de)

plt.figure(figsize=(7, 5))
#plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.plot(E_grid, rho_num, label="Numeric DOS")
plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.xlabel("E")
plt.ylabel("DOS(E)")
plt.title(fr"Comparison between numerical and NN for $N_k = 200, de={de}$")
plt.legend()
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
#plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.plot(E_grid, rho_num, label="Numeric DOS")
plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.xlabel("E")
plt.ylabel("DOS(E)")
plt.title(fr"Comparison between numerical and NN for $N_k = {Ne}, de={de}$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/DOS_nn_finer.png",dpi=300)
