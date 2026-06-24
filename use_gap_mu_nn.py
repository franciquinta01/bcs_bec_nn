import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 1. Network architecture
# ============================================================

class GapMuNet(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. Utilities
# ============================================================

def nearest_value(values, target):
    values = np.unique(values)
    return values[np.argmin(np.abs(values - target))]


def predict_gap_mu(model, X_raw, X_mean, X_std, Y_mean, Y_std, eps_delta, device):
    X_raw = np.asarray(X_raw, dtype=np.float64)

    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)

    X_norm = (X_raw - X_mean) / X_std

    x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        y_norm = model(x_tensor).cpu().numpy()

    y_trans = y_norm * Y_std + Y_mean

    delta = np.exp(y_trans[:, 0]) - eps_delta
    mu = y_trans[:, 1]

    return delta, mu


def select_slice(X, a0, b0, de0, n0, omega0=None, V0=None):
    mask = (
        np.isclose(X[:, 0], a0)
        & np.isclose(X[:, 1], b0)
        & np.isclose(X[:, 2], de0)
        & np.isclose(X[:, 3], n0)
    )

    if omega0 is not None:
        mask &= np.isclose(X[:, 5], omega0)

    if V0 is not None:
        mask &= np.isclose(X[:, 4], V0)

    return mask


# ============================================================
# 3. Load model
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint = torch.load(
    "gap_mu_net.pt",
    map_location=device,
    weights_only=False
)

model = GapMuNet(input_dim=6).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

X_mean = checkpoint["X_mean"]
X_std = checkpoint["X_std"]
Y_mean = checkpoint["Y_mean"]
Y_std = checkpoint["Y_std"]
eps_delta = checkpoint["eps_delta"]


# ============================================================
# 4. Load numerical dataset
# ============================================================

data = np.load("gap_mu_dataset.npz", allow_pickle=True)

X = data["X"]
Y = data["Y"]

Delta = Y[:, 0]
mu = Y[:, 1]

mask_clean = (
    np.isfinite(X).all(axis=1)
    & np.isfinite(Y).all(axis=1)
    & (Delta > 1e-8)
    & (np.abs(mu) < 10)
)

X = X[mask_clean]
Y = Y[mask_clean]

print("Clean dataset:", X.shape, Y.shape)


# ============================================================
# 5. Choose existing parameter values
# ============================================================

# requested values
a_req = 0.2
b_req = 0.2
de_req = 0.002
n_req = 0.05
omega_req = 1.0
V_req = 2.0

# nearest values actually present
a0 = nearest_value(X[:, 0], a_req)
b0 = nearest_value(X[:, 1], b_req)
de0 = nearest_value(X[:, 2], de_req)
n0 = nearest_value(X[:, 3], n_req)

base_mask = (
    np.isclose(X[:, 0], a0)
    & np.isclose(X[:, 1], b0)
    & np.isclose(X[:, 2], de0)
    & np.isclose(X[:, 3], n0)
)

X_base = X[base_mask]

if len(X_base) == 0:
    raise RuntimeError("Base slice is empty. Choose different a,b,de,n.")

omega_fixed = nearest_value(X_base[:, 5], omega_req)
V_fixed = nearest_value(X_base[:, 4], V_req)

print("\nRequested values:")
print("a,b,de,n,V0,omega0 =", a_req, b_req, de_req, n_req, V_req, omega_req)

print("\nUsing nearest available values:")
print("a,b,de,n,V0,omega0 =", a0, b0, de0, n0, V_fixed, omega_fixed)

print("\nAvailable points in base slice:", len(X_base))


# ============================================================
# 6. Delta and mu vs V0 at fixed omega0
# ============================================================

mask_V = select_slice(
    X,
    a0=a0,
    b0=b0,
    de0=de0,
    n0=n0,
    omega0=omega_fixed
)

X_V = X[mask_V]
Y_V = Y[mask_V]

print("Points for V0 scan:", len(X_V))

if len(X_V) > 0:
    order = np.argsort(X_V[:, 4])
    X_V = X_V[order]
    Y_V = Y_V[order]

    Delta_pred_V, mu_pred_V = predict_gap_mu(
        model, X_V, X_mean, X_std, Y_mean, Y_std, eps_delta, device
    )

    V_axis = X_V[:, 4]

    plt.figure(figsize=(7, 5))
    plt.plot(V_axis, Y_V[:, 0], "o", markersize=3, label="Numerical")
    plt.plot(V_axis, Delta_pred_V, "-", linewidth=2, label="NN")
    plt.xlabel(r"$V_0$")
    plt.ylabel(r"$\Delta$")
    plt.title(fr"$\Delta$ vs $V_0$ | a={a0:.3f}, b={b0:.3f}, n={n0:.3f}, $\omega_0$={omega_fixed:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/Delta_vs_V0_NN_vs_num.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(V_axis, Y_V[:, 1], "o", markersize=3, label="Numerical")
    plt.plot(V_axis, mu_pred_V, "-", linewidth=2, label="NN")
    plt.xlabel(r"$V_0$")
    plt.ylabel(r"$\mu$")
    plt.title(fr"$\mu$ vs $V_0$ | a={a0:.3f}, b={b0:.3f}, n={n0:.3f}, $\omega_0$={omega_fixed:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/mu_vs_V0_NN_vs_num.png", dpi=300)
    plt.show()


# ============================================================
# 7. Delta and mu vs omega0 at fixed V0
# ============================================================

mask_w = select_slice(
    X,
    a0=a0,
    b0=b0,
    de0=de0,
    n0=n0,
    V0=V_fixed
)

X_w = X[mask_w]
Y_w = Y[mask_w]

print("Points for omega0 scan:", len(X_w))

if len(X_w) > 0:
    order = np.argsort(X_w[:, 5])
    X_w = X_w[order]
    Y_w = Y_w[order]

    Delta_pred_w, mu_pred_w = predict_gap_mu(
        model, X_w, X_mean, X_std, Y_mean, Y_std, eps_delta, device
    )

    omega_axis = X_w[:, 5]

    plt.figure(figsize=(7, 5))
    plt.plot(omega_axis, Y_w[:, 0], "o", markersize=3, label="Numerical")
    plt.plot(omega_axis, Delta_pred_w, "-", linewidth=2, label="NN")
    plt.xlabel(r"$\omega_0$")
    plt.ylabel(r"$\Delta$")
    plt.title(fr"$\Delta$ vs $\omega_0$ | a={a0:.3f}, b={b0:.3f}, n={n0:.3f}, $V_0$={V_fixed:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/Delta_vs_omega0_NN_vs_num.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(omega_axis, Y_w[:, 1], "o", markersize=3, label="Numerical")
    plt.plot(omega_axis, mu_pred_w, "-", linewidth=2, label="NN")
    plt.xlabel(r"$\omega_0$")
    plt.ylabel(r"$\mu$")
    plt.title(fr"$\mu$ vs $\omega_0$ | a={a0:.3f}, b={b0:.3f}, n={n0:.3f}, $V_0$={V_fixed:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/mu_vs_omega0_NN_vs_num.png", dpi=300)
    plt.show()


# ============================================================
# 8. Global scatter NN vs numerical
# ============================================================

rng = np.random.default_rng(123)
N_plot = min(100000, len(X))
idx = rng.choice(len(X), size=N_plot, replace=False)

X_sub = X[idx]
Y_sub = Y[idx]

Delta_pred, mu_pred = predict_gap_mu(
    model, X_sub, X_mean, X_std, Y_mean, Y_std, eps_delta, device
)

Delta_true = Y_sub[:, 0]
mu_true = Y_sub[:, 1]

plt.figure(figsize=(6, 6))
plt.scatter(Delta_true, Delta_pred, s=2, alpha=0.25)
lims = [
    min(Delta_true.min(), Delta_pred.min()),
    max(Delta_true.max(), Delta_pred.max())
]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel(r"$\Delta_{\mathrm{num}}$")
plt.ylabel(r"$\Delta_{\mathrm{NN}}$")
plt.title(r"$\Delta$: NN vs numerical")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/Delta_scatter_NN_vs_num.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(mu_true, mu_pred, s=2, alpha=0.25)
lims = [
    min(mu_true.min(), mu_pred.min()),
    max(mu_true.max(), mu_pred.max())
]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel(r"$\mu_{\mathrm{num}}$")
plt.ylabel(r"$\mu_{\mathrm{NN}}$")
plt.title(r"$\mu$: NN vs numerical")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/mu_scatter_NN_vs_num.png", dpi=300)
plt.show()


# ============================================================
# 9. Error metrics
# ============================================================

eps = 1e-12

rel_err_Delta = np.abs(Delta_pred - Delta_true) / (np.abs(Delta_true) + eps)
abs_err_mu = np.abs(mu_pred - mu_true)

print("\nError metrics on random subset")
print("--------------------------------")
print("Delta relative error:")
print("mean   =", np.mean(rel_err_Delta))
print("median =", np.median(rel_err_Delta))
print("p95    =", np.percentile(rel_err_Delta, 95))
print("p99    =", np.percentile(rel_err_Delta, 99))

print("\nmu absolute error:")
print("mean   =", np.mean(abs_err_mu))
print("median =", np.median(abs_err_mu))
print("p95    =", np.percentile(abs_err_mu, 95))
print("p99    =", np.percentile(abs_err_mu, 99))
