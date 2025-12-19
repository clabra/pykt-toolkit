import numpy as np
from scipy.interpolate import CubicSpline

# Data points
lambdas = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
aucs = np.array([0.7249, 0.7217, 0.7100, 0.6911, 0.6787, 0.6700])
corrs = np.array([0.5090, 0.6609, 0.7802, 0.8610, 0.8890, 0.9081])

# Create continuous range for lambda
l_fine = np.linspace(0, 1, 1000)

# Spline interpolation
cs_auc = CubicSpline(lambdas, aucs)
cs_corr = CubicSpline(lambdas, corrs)

# Construct fine-grained curve C(l) = (x(l), y(l))
x = cs_corr(l_fine)
y = cs_auc(l_fine)

# First derivatives
dx = np.gradient(x, l_fine)
dy = np.gradient(y, l_fine)

# Second derivatives
dx2 = np.gradient(dx, l_fine)
dy2 = np.gradient(dy, l_fine)

# Curvature formula for parametric curve: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(1.5)
curvature = np.abs(dx * dy2 - dy * dx2) / (dx**2 + dy**2)**1.5

# Find index of max curvature
knee_idx = np.argmax(curvature)
knee_lambda = l_fine[knee_idx]
knee_auc = y[knee_idx]
knee_corr = x[knee_idx]

print(f"--- Continuous Knee Point Analysis ---")
print(f"Max Curvature Lambda: {knee_lambda:.4f}")
print(f"Corresponding AUC: {knee_auc:.4f}")
print(f"Corresponding Correlation: {knee_corr:.4f}")

# Distance-to-Chord method (Continuous version)
# Line between P1(corr[0], auc[0]) and Pn(corr[-1], auc[-1])
p1 = np.array([corrs[0], aucs[0]])
pn = np.array([corrs[-1], aucs[-1]])
vec = pn - p1
u_vec = vec / np.linalg.norm(vec)

distances = []
for i in range(len(x)):
    p = np.array([x[i], y[i]])
    dist = np.linalg.norm(np.cross(vec, p1 - p)) / np.linalg.norm(vec)
    distances.append(dist)

knee_dist_idx = np.argmax(distances)
print(f"\n--- Max Distance-to-Chord Analysis ---")
print(f"Max Distance Lambda: {l_fine[knee_dist_idx]:.4f}")
print(f"Max Distance: {distances[knee_dist_idx]:.6f}")
