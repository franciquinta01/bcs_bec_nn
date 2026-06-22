import numpy as np

data = np.load("dos_dataset.npz")

X = data["A"]   # [E, de, a, b]
Y = data["B"]   # DOS

E = X[:, 0]
de = X[:, 1]
a = X[:, 2]
b = X[:, 3]

percentile = 90
extra_window = 2   # allarga un po' la regione attorno ai picchi

peak_mask = np.zeros(len(X), dtype=bool)

# gruppi curva per curva: stessa de, a, b
curves = {}

for i, key in enumerate(zip(de, a, b)):
    curves.setdefault(key, []).append(i)

for key, idxs in curves.items():
    idxs = np.array(idxs)

    y_curve = Y[idxs, 0]

    threshold = np.percentile(y_curve, percentile)

    local_mask = y_curve >= threshold

    # allarga la regione selezionata per non prendere solo il massimo
    selected = np.where(local_mask)[0]

    for j in selected:
        jmin = max(0, j - extra_window)
        jmax = min(len(idxs), j + extra_window + 1)
        peak_mask[idxs[jmin:jmax]] = True

X_peak = X[peak_mask]
Y_peak = Y[peak_mask]

X_outer = X[~peak_mask]
Y_outer = Y[~peak_mask]

print("Original:", X.shape, Y.shape)
print("Peak:", X_peak.shape, Y_peak.shape)
print("Outer:", X_outer.shape, Y_outer.shape)

np.savez(
    "dos_dataset_peak.npz",
    A=X_peak,
    B=Y_peak
)

np.savez(
    "dos_dataset_outer.npz",
    A=X_outer,
    B=Y_outer
)

np.savez(
    "dos_dataset_with_masks.npz",
    A=X,
    B=Y,
    peak_mask=peak_mask
)
