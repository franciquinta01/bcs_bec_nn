import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DOSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 128),
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
    #checkpoint = torch.load(path, map_location="cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model = DOSNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_mean = checkpoint["X_mean"]
    X_std = checkpoint["X_std"]

    return model, X_mean, X_std


def dos_from_nn(E_grid, de, model, X_mean, X_std):
    X = np.column_stack([
        E_grid,
        np.full_like(E_grid, de)
    ])

    X_norm = (X - X_mean) / X_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        y_log = model(X_tensor).numpy().flatten()

    rho = y_log

    return rho


model, X_mean, X_std = load_dos_model("dos_net.pt")

data = np.load("dos_dataset.npz")
E_grid = data["E_grid"]

de = 0.02

rho_nn = dos_from_nn(E_grid, de, model, X_mean, X_std)

plt.figure(figsize=(7, 5))
plt.plot(E_grid, rho_nn, label="DOS from NN")
plt.xlabel("E")
plt.ylabel("DOS(E)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/DOS_nn.png",dpi=300)
