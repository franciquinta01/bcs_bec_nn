import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


# ============================================================
# 1. Load dataset
# ============================================================

data = np.load("gap_mu_dataset.npz", allow_pickle=True)

X = data["X"]   # [a, b, de, n, V0, omega0]
Y = data["Y"]   # [Delta, mu]

print("Original X:", X.shape)
print("Original Y:", Y.shape)


# ============================================================
# 2. Clean dataset
# ============================================================

Delta = Y[:, 0]
mu = Y[:, 1]

mask = (
    np.isfinite(X).all(axis=1)
    & np.isfinite(Y).all(axis=1)
    & (Delta > 1e-8)
    & (np.abs(mu) < 10)
)

X = X[mask]
Y = Y[mask]

print("Clean X:", X.shape)
print("Clean Y:", Y.shape)


# ============================================================
# 3. Transform target
#    Learn log(Delta), mu
# ============================================================

eps_delta = 1e-8

Y_trans = np.zeros_like(Y)
Y_trans[:, 0] = np.log(Y[:, 0] + eps_delta)
Y_trans[:, 1] = Y[:, 1]


# ============================================================
# 4. Normalize input and output
# ============================================================

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

Y_mean = Y_trans.mean(axis=0)
Y_std = Y_trans.std(axis=0)

X_norm = (X - X_mean) / X_std
Y_norm = (Y_trans - Y_mean) / Y_std


# ============================================================
# 5. Torch tensors
# ============================================================

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)


# ============================================================
# 6. Train / validation split
# ============================================================

N = len(dataset)
N_train = int(0.8 * N)
N_val = N - N_train

train_set, val_set = random_split(
    dataset,
    [N_train, N_val],
    generator=torch.Generator().manual_seed(123)
)

batch_size = 4096

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# 7. Neural network
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
# 8. Training setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = GapMuNet(input_dim=6).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-6
)

loss_fn = nn.MSELoss()

epochs = 300

train_losses = []
val_losses = []


# ============================================================
# 9. Training loop
# ============================================================

for epoch in range(epochs):

    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train loss = {train_loss:.6e} | "
            f"Val loss = {val_loss:.6e}"
        )


# ============================================================
# 10. Save model
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "eps_delta": eps_delta,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "input_features": ["a", "b", "de", "n", "V0", "omega0"],
        "output_targets": ["log_delta", "mu"],
    },
    "gap_mu_net.pt"
)

print("Saved model: gap_mu_net.pt")


# ============================================================
# 11. Plot loss
# ============================================================

plt.figure(figsize=(7, 5))
plt.semilogy(train_losses, label="Train")
plt.semilogy(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/gap_mu_training_loss.png", dpi=300)
plt.show()


# ============================================================
# 12. Prediction helper
# ============================================================

def predict_gap_mu(model, X_raw, X_mean, X_std, Y_mean, Y_std, eps_delta, device):
    X_raw = np.array(X_raw, dtype=np.float64)

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
