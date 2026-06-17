"""
================================================================================
Kolmogorov-Arnold Network (KAN) for Probabilistic Renewable Energy Forecasting
================================================================================
This module implements a lightweight KAN with radial basis function (RBF)
splines for stable training. It predicts both mean (mu) and standard
deviation (sigma) of wind/solar power, enabling data-driven uncertainty
quantification for stochastic optimization.

Reference:
- Liu et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
- Aydin et al. (2025). Energy and AI. (KAN for energy systems)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. KAN Core Implementation (RBF-based for stability)
# ==============================================================================
class KANLayer(nn.Module):
    """
    Single KAN layer: y_j = sum_i [ w_base * SiLU(x_i) + sum_g w_spline * RBF(x_i, grid_g) ]
    """
    def __init__(self, in_dim, out_dim, grid_size=10, spline_order=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        # Base path (SiLU + linear weight)
        self.base_activation = nn.SiLU()
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

        # Spline path: RBF on uniform grid
        grid_min, grid_max = -2.0, 2.0
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.grid_width = (grid_max - grid_min) / max(grid_size - 1, 1)

        # Spline coefficients: (out_dim, in_dim, grid_size)
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        self.spline_scale = nn.Parameter(torch.ones(out_dim, in_dim) * 0.5)

    def forward(self, x):
        # x: (batch, in_dim)
        # Base component
        base = self.base_activation(x)  # (batch, in_dim)
        base_out = torch.einsum('bi,oi->bo', base, self.base_weight)  # (batch, out_dim)

        # Spline component: RBF basis
        # (batch, in_dim, 1) - (grid_size,) -> (batch, in_dim, grid_size)
        dist = x.unsqueeze(-1) - self.grid
        rbf = torch.exp(-0.5 * (dist / (self.grid_width + 1e-8)) ** 2)

        # Weighted sum over grid points, then over input dim
        spline = torch.einsum('big,oig->bo', rbf, self.spline_weight)
        # Scale: (out_dim,) -> (1, out_dim) for broadcasting with (batch, out_dim)
        scale = self.spline_scale.mean(dim=1).unsqueeze(0)
        spline = spline * scale

        return base_out + spline


class ProbabilisticKAN(nn.Module):
    """
    KAN for probabilistic time-series forecasting.
    Output: mu (mean), sigma (std) of the predictive distribution.
    Assumes Gaussian: y ~ N(mu, sigma^2)
    """
    def __init__(self, input_size, hidden_size, num_layers=2, grid_size=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.kan_layers = nn.ModuleList([
            KANLayer(hidden_size, hidden_size, grid_size=grid_size)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, features) or (batch, seq_len)
        # Flatten time dimension for KAN
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, feat = x.shape
        x = x.reshape(batch_size, seq_len * feat)
        x = self.input_proj(x)

        for layer in self.kan_layers:
            x = layer(x)
            x = self.dropout(x)

        out = self.output_layer(x)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-4  # Ensure positivity
        return mu, sigma

    def nll_loss(self, y_true, mu, sigma):
        """Negative log-likelihood for Gaussian distribution."""
        return torch.mean(0.5 * torch.log(2 * np.pi * sigma ** 2) +
                          0.5 * ((y_true - mu) ** 2) / (sigma ** 2))


# ==============================================================================
# 2. Dataset
# ==============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 3. Data Preprocessing
# ==============================================================================
def load_renewable_series(csv_path, col_actual="actual_pu", col_pred="predicted_pu"):
    """Load renewable power series from user's prediction CSV."""
    df = pd.read_csv(csv_path)
    actual = df[col_actual].bfill().ffill().values.astype(float)
    # If predicted exists, use residual analysis; otherwise use actual only
    if col_pred in df.columns:
        pred = df[col_pred].bfill().ffill().values.astype(float)
    else:
        pred = actual.copy()
    return actual, pred


def build_supervised_dataset(series, lookback=168, n_features=1):
    """
    Build supervised dataset for multi-step ahead probabilistic forecasting.
    We predict the mean and variance of the NEXT lookback hours.
    For simplicity: predict next 1 hour, but use multi-horizon target std.
    """
    T = len(series)
    X, y = [], []
    for i in range(T - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    X = np.array(X).reshape(-1, lookback, n_features)
    y = np.array(y)
    return X, y


def train_test_split(X, y, ratio=0.8):
    split = int(len(X) * ratio)
    return X[:split], X[split:], y[:split], y[split:]


# ==============================================================================
# 4. Training Pipeline
# ==============================================================================
def train_kan_forecaster(X_train, y_train, X_val, y_val, params):
    """
    Train ProbabilisticKAN and return best model + history.
    """
    device = torch.device(params.get("device", "cpu"))
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    lr = params["lr"]

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ProbabilisticKAN(
        input_size=X_train.shape[1] * X_train.shape[2],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        grid_size=params["grid_size"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sigma = model(xb)
            loss = model.nll_loss(yb, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, sigma = model(xb)
                loss = model.nll_loss(yb, mu, sigma)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [KAN Epoch {epoch+1:3d}/{epochs}] Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ==============================================================================
# 5. Inference & Residual Analysis for Scenario Generation
# ==============================================================================
def extract_residual_distribution(model, X, y, device="cpu"):
    """
    Analyze prediction residuals to build empirical error distribution.
    Returns: mu_pred, sigma_pred, residuals (y - mu)
    """
    model.eval()
    ds = TimeSeriesDataset(X, y)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    mus, sigmas, residuals = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            mu, sigma = model(xb)
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            yb = yb.numpy()
            mus.extend(mu)
            sigmas.extend(sigma)
            residuals.extend(yb - mu)

    return np.array(mus), np.array(sigmas), np.array(residuals)


def generate_probabilistic_forecast(model, series, lookback, device="cpu"):
    """
    Generate full-length mu and sigma for the series.
    """
    model.eval()
    T = len(series)
    mu_full = np.full(T, np.nan)
    sigma_full = np.full(T, np.nan)

    with torch.no_grad():
        for i in range(T - lookback):
            x = series[i:i + lookback].reshape(1, lookback, 1)
            xt = torch.tensor(x, dtype=torch.float32).to(device)
            mu, sigma = model(xt)
            mu_full[i + lookback] = mu.item()
            sigma_full[i + lookback] = sigma.item()

    # Forward fill NaNs at the beginning
    mu_full = pd.Series(mu_full).bfill().values
    sigma_full = pd.Series(sigma_full).bfill().values
    return mu_full, sigma_full


# ==============================================================================
# 6. Main Entry
# ==============================================================================
def run_kan_pipeline(csv_path, params, asset_name="Wind"):
    """
    End-to-end KAN training and probabilistic forecasting.
    Returns: dict with model, mu, sigma, residuals, history
    """
    print(f"\n{'='*60}")
    print(f"KAN Probabilistic Forecasting: {asset_name}")
    print(f"{'='*60}")

    actual, _ = load_renewable_series(csv_path)
    # Normalize to [-1, 1] for stable KAN training
    mean_val = np.mean(actual)
    std_val = np.std(actual) + 1e-8
    series_norm = (actual - mean_val) / std_val

    X, y = build_supervised_dataset(series_norm, lookback=params["lookback"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, ratio=params["train_ratio"])

    print(f"  Samples: Train={len(X_train)}, Val={len(X_val)}")
    print(f"  Input shape: {X_train.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["device"] = str(device)
    print(f"  Device: {device}")

    model, history = train_kan_forecaster(X_train, y_train, X_val, y_val, params)

    # Generate full probabilistic forecast
    mu_norm, sigma_norm = generate_probabilistic_forecast(model, series_norm, params["lookback"], device)

    # Denormalize
    mu = mu_norm * std_val + mean_val
    sigma = sigma_norm * std_val  # Standard deviation scales linearly

    # Clip to physical bounds [0, 1]
    mu = np.clip(mu, 0.0, 1.0)
    sigma = np.clip(sigma, 0.001, 0.5)

    # Residuals
    residuals = actual - mu

    print(f"  Forecast stats: mu=[{mu.min():.3f}, {mu.max():.3f}], sigma=[{sigma.min():.3f}, {sigma.max():.3f}]")
    print(f"  Residual MAE: {np.mean(np.abs(residuals)):.4f}, RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")

    return {
        "model": model,
        "mu": mu,
        "sigma": sigma,
        "residuals": residuals,
        "history": history,
        "mean_val": mean_val,
        "std_val": std_val,
        "actual": actual,
    }


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from config import KANParams, DataPaths

    result_wind = run_kan_pipeline(DataPaths["wind_pred"], KANParams, asset_name="Gansu Wind")
    result_solar = run_kan_pipeline(DataPaths["solar_pred"], KANParams, asset_name="Gansu Solar")
