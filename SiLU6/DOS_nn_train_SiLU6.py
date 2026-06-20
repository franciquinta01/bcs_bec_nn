import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


# ============================================================
# 1. Load dataset
# ============================================================

data = np.load("/home/fquintavalle/bcs_bec_nn/dos_dataset.npz")

X = data["A"]      # shape: (N_samples, 4) -> [E, de, a, b]
Y = data["B"]      # shape: (N_samples, 1) -> [DOS(E)]

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# 2. Prepare target: learn log(DOS), not DOS directly
# ============================================================

eps_log = 1e-12
Y_log = np.log(Y + eps_log)


# ============================================================
# 3. Normalize input
# ============================================================

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X_norm = (X - X_mean) / X_std


# ============================================================
# 4. Convert to PyTorch tensors
# ============================================================

X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_log, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)


# ============================================================
# 5. Train / validation split
# ============================================================

N = len(dataset)
N_train = int(0.8 * N)
N_val = N - N_train

train_set, val_set = random_split(dataset, [N_train, N_val])

train_loader = DataLoader(
    train_set,
    batch_size=2048,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=2048,
    shuffle=False
)


# ============================================================
# 6. Neural network
# ============================================================

class DOSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 7. Training setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = DOSNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)
loss_fn = nn.MSELoss()

epochs = 600

train_losses = []
val_losses = []


# ============================================================
# 8. Training loop
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

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train loss = {train_loss:.6e} | "
            f"Val loss = {val_loss:.6e}"
        )


# ============================================================
# 9. Save model
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "eps_log": eps_log,
        "train_losses": train_losses,
        "val_losses": val_losses,
    },
    "dos_net.pt"
)


# ============================================================
# 10. Plot loss
# ============================================================

plt.figure(figsize=(7, 5))
plt.semilogy(train_losses, label="Train")
plt.semilogy(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss on log(DOS)")
plt.legend()
plt.grid(True)
plt.savefig("images/loss_plot.png",dpi=300)


# ============================================================
# 11. Function to predict a full DOS curve
# ============================================================

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

    rho = np.exp(y_log)

    return rho
