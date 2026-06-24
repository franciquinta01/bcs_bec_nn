import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Load dataset
# ============================================================

data = np.load("gap_mu_dataset.npz", allow_pickle=True)

X = data["X"]
failed = data["failed"]

columns = ["a", "b", "de", "n", "V0", "omega0"]

valid_df = pd.DataFrame(X, columns=columns)

failed_df = pd.DataFrame(
    failed[:, :6].astype(float),
    columns=columns
)

valid_df["status"] = "valid"
failed_df["status"] = "failed"

all_df = pd.concat([valid_df, failed_df], ignore_index=True)


# ============================================================
# Choose fixed parameters
# ============================================================

a0 = all_df["a"].unique()[0]
b0 = all_df["b"].unique()[0]
n0 = all_df["n"].unique()[0]
de0 = all_df["de"].unique()[0]

print("Using:")
print("a =", a0)
print("b =", b0)
print("n =", n0)
print("de =", de0)


subset = all_df[
    (all_df["a"] == a0) &
    (all_df["b"] == b0) &
    (all_df["n"] == n0) &
    (all_df["de"] == de0)
]


# ============================================================
# Failure rate in the plane (V0, omega0)
# ============================================================

total_counts = subset.groupby(["V0", "omega0"]).size()
failed_counts = (
    subset[subset["status"] == "failed"]
    .groupby(["V0", "omega0"])
    .size()
)

failure_rate = failed_counts / total_counts
failure_rate = failure_rate.fillna(0.0)

heatmap = failure_rate.unstack("omega0")


# ============================================================
# Plot heatmap
# ============================================================

plt.figure(figsize=(9, 6))

plt.imshow(
    heatmap.values,
    origin="lower",
    aspect="auto",
    extent=[
        heatmap.columns.min(),
        heatmap.columns.max(),
        heatmap.index.min(),
        heatmap.index.max()
    ]
)

plt.colorbar(label="Failure rate")
plt.xlabel(r"$\omega_0$")
plt.ylabel(r"$V_0$")
plt.title(
    fr"Failure rate heatmap | a={a0:.3f}, b={b0:.3f}, n={n0:.3f}, de={de0:.4f}"
)

plt.tight_layout()
plt.savefig("images/failure_heatmap_V0_omega0.png", dpi=300)
