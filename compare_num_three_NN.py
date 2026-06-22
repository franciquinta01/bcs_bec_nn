import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset_DOS import epsilon_3D, dos_parallel


class DOSNet(nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def predict_dos(model, E_grid, de, a, b, X_mean, X_std, device):
    X_plot = np.column_stack([
        E_grid,
        np.full_like(E_grid, de),
        np.full_like(E_grid, a),
        np.full_like(E_grid, b)
    ])

    X_plot_norm = (X_plot - X_mean) / X_std
    x_tensor = torch.tensor(X_plot_norm, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        y_log = model(x_tensor).cpu().numpy().flatten()

    return np.exp(y_log)


def load_model(model_path, activation, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = DOSNet(activation).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint["X_mean"], checkpoint["X_std"]


# ============================================================
# Parameters to test
# ============================================================

Ne = 800
de = 0.002
a = 1.0
b = 0
n_jobs = 10

E_points = 1000

# ============================================================
# Numerical DOS
# ============================================================

kx = np.linspace(-np.pi, np.pi, Ne)
ky = kx
kz = kx

X, Y, Z = np.meshgrid(kx, ky, kz, indexing="ij")
epsilon = epsilon_3D(X, Y, Z, a, b).ravel()

E_grid = np.linspace(epsilon.min(), epsilon.max(), E_points)
rho_num = dos_parallel(E_grid, epsilon, de, n_jobs=n_jobs)

# ============================================================
# Neural networks
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

models = {
    "Tanh6": {
        "path": "Tanh6/dos_net_Tanh6.pt",
        "activation": nn.Tanh,
    },
    "SoftPlus6": {
        "path": "SoftPlus6/dos_net_SoftPlus6.pt",
        "activation": nn.Softplus,
    },
    "SiLU6": {
        "path": "SiLU6/dos_net_SiLU6.pt",
        "activation": nn.SiLU,
    },
}

rho_predictions = {}

for name, cfg in models.items():
    if not os.path.exists(cfg["path"]):
        raise FileNotFoundError(f"Missing model file: {cfg['path']}")

    model, X_mean, X_std = load_model(
        cfg["path"],
        cfg["activation"],
        device
    )

    rho_predictions[name] = predict_dos(
        model,
        E_grid,
        de,
        a,
        b,
        X_mean,
        X_std,
        device
    )

# ============================================================
# Plot
# ============================================================

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(9, 6))

plt.plot(E_grid, rho_num, color="black", linewidth=2.5, label="Numerical DOS")

for name, rho in rho_predictions.items():
    plt.plot(E_grid, rho, linewidth=1.6, label=name)

plt.xlabel("E")
plt.ylabel(r"$\rho(E)$")
plt.title(fr"DOS comparison: $a={a}$, $b={b}$, $de={de}$, $N_k={Ne}$")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("images/DOS_three_NN_comparison.png", dpi=300)
