"""
================================================================================
Baseline Predictors for Comparative Evaluation against KAN
================================================================================
Implements LSTM, GRU, and MLP probabilistic forecasters with identical
input/output interface to KAN for fair comparison.

Usage:
    cd src
    python baseline_predictors.py

Or from project root:
    PYTHONPATH=src python src/baseline_predictors.py
================================================================================"""

import os
import sys

# Fix Python path (works when run from src/ or project root)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(os.path.dirname(__file__)) == "src":
    sys.path.insert(0, _project_root)
    sys.path.insert(0, os.path.join(_project_root, "src"))
else:
    sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. Probabilistic Baseline Models
# ==============================================================================

class ProbabilisticLSTM(nn.Module):
    """LSTM-based probabilistic forecaster. Output: mu, sigma."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-4
        return mu, sigma

    def nll_loss(self, y_true, mu, sigma):
        return torch.mean(0.5 * torch.log(2 * np.pi * sigma ** 2) +
                          0.5 * ((y_true - mu) ** 2) / (sigma ** 2))


class ProbabilisticGRU(nn.Module):
    """GRU-based probabilistic forecaster. Output: mu, sigma."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-4
        return mu, sigma

    def nll_loss(self, y_true, mu, sigma):
        return torch.mean(0.5 * torch.log(2 * np.pi * sigma ** 2) +
                          0.5 * ((y_true - mu) ** 2) / (sigma ** 2))


class ProbabilisticMLP(nn.Module):
    """MLP-based probabilistic forecaster. Flattens time dimension."""
    def __init__(self, input_size=168, hidden_size=128, num_layers=3):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        out = self.net(x)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-4
        return mu, sigma

    def nll_loss(self, y_true, mu, sigma):
        return torch.mean(0.5 * torch.log(2 * np.pi * sigma ** 2) +
                          0.5 * ((y_true - mu) ** 2) / (sigma ** 2))


# ==============================================================================
# 2. Unified Training & Evaluation Utilities
# ==============================================================================

class _TSDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def train_baseline_model(model, X_train, y_train, X_val, y_val, epochs=60,
                         batch_size=512, lr=0.001, device="cpu"):
    """Generic training loop for any baseline model."""
    train_ds = _TSDataset(X_train, y_train)
    val_ds = _TSDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val = float("inf")
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
                val_losses.append(model.nll_loss(yb, mu, sigma).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [Epoch {epoch+1:3d}/{epochs}] Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def generate_forecast(model, series, lookback, device="cpu"):
    """Generate mu, sigma for full series."""
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
    mu_full = pd.Series(mu_full).bfill().values
    sigma_full = pd.Series(sigma_full).bfill().values
    return mu_full, sigma_full


# ==============================================================================
# 3. Evaluation Metrics
# ==============================================================================

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def crps_gaussian(y_true, mu, sigma):
    """CRPS for Gaussian distribution (closed-form)."""
    from scipy.stats import norm
    z = (y_true - mu) / (sigma + 1e-8)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    return np.mean(crps)


def evaluate_forecaster(y_true, mu_pred, sigma_pred):
    """Compute all metrics."""
    return {
        "RMSE": rmse(y_true, mu_pred),
        "MAE": mae(y_true, mu_pred),
        "MAPE_%": mape(y_true, mu_pred),
        "CRPS": crps_gaussian(y_true, mu_pred, sigma_pred),
    }


# ==============================================================================
# 4. Main Comparison Pipeline
# ==============================================================================

def run_comparison(csv_path, params, asset_name="Wind", device="cpu"):
    """
    Run KAN + LSTM + GRU + MLP on the same dataset and return comparison table.
    """
    from kan_predictor import load_renewable_series, build_supervised_dataset, train_test_split

    actual, _ = load_renewable_series(csv_path)
    mean_val = np.mean(actual)
    std_val = np.std(actual) + 1e-8
    series_norm = (actual - mean_val) / std_val

    X, y = build_supervised_dataset(series_norm, lookback=params["lookback"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, ratio=params["train_ratio"])

    print(f"\n{'='*70}")
    print(f"Baseline Comparison: {asset_name}")
    print(f"{'='*70}")
    print(f"  Train={len(X_train)}, Val={len(X_val)}, Shape={X_train.shape}")

    results = {}
    val_results = {}

    # ---------- KAN ----------
    from kan_predictor import ProbabilisticKAN, train_kan_forecaster
    print(f"\n[1/4] Training KAN...")
    kan_model, _ = train_kan_forecaster(X_train, y_train, X_val, y_val, {
        **params, "device": device
    })
    mu_k, sig_k = generate_forecast(kan_model, series_norm, params["lookback"], device)
    mu_k = mu_k * std_val + mean_val
    sig_k = sig_k * std_val
    results["KAN"] = {"mu": mu_k, "sigma": sig_k}
    val_results["KAN"] = evaluate_forecaster(actual, mu_k, sig_k)

    # ---------- LSTM ----------
    print(f"\n[2/4] Training LSTM...")
    lstm = ProbabilisticLSTM(input_size=1, hidden_size=64, num_layers=2)
    lstm, _ = train_baseline_model(lstm, X_train, y_train, X_val, y_val,
                                   epochs=params["epochs"], batch_size=params["batch_size"],
                                   lr=params["lr"], device=device)
    mu_l, sig_l = generate_forecast(lstm, series_norm, params["lookback"], device)
    mu_l = mu_l * std_val + mean_val
    sig_l = sig_l * std_val
    results["LSTM"] = {"mu": mu_l, "sigma": sig_l}
    val_results["LSTM"] = evaluate_forecaster(actual, mu_l, sig_l)

    # ---------- GRU ----------
    print(f"\n[3/4] Training GRU...")
    gru = ProbabilisticGRU(input_size=1, hidden_size=64, num_layers=2)
    gru, _ = train_baseline_model(gru, X_train, y_train, X_val, y_val,
                                  epochs=params["epochs"], batch_size=params["batch_size"],
                                  lr=params["lr"], device=device)
    mu_g, sig_g = generate_forecast(gru, series_norm, params["lookback"], device)
    mu_g = mu_g * std_val + mean_val
    sig_g = sig_g * std_val
    results["GRU"] = {"mu": mu_g, "sigma": sig_g}
    val_results["GRU"] = evaluate_forecaster(actual, mu_g, sig_g)

    # ---------- MLP ----------
    print(f"\n[4/4] Training MLP...")
    mlp = ProbabilisticMLP(input_size=params["lookback"], hidden_size=128, num_layers=3)
    mlp, _ = train_baseline_model(mlp, X_train, y_train, X_val, y_val,
                                  epochs=params["epochs"], batch_size=params["batch_size"],
                                  lr=params["lr"], device=device)
    mu_m, sig_m = generate_forecast(mlp, series_norm, params["lookback"], device)
    mu_m = mu_m * std_val + mean_val
    sig_m = sig_m * std_val
    results["MLP"] = {"mu": mu_m, "sigma": sig_m}
    val_results["MLP"] = evaluate_forecaster(actual, mu_m, sig_m)

    # ---------- Summary Table ----------
    print(f"\n{'='*70}")
    print("Comparison Results")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'RMSE':<10} {'MAE':<10} {'MAPE_%':<10} {'CRPS':<10}")
    print("-" * 70)
    for model_name, metrics in val_results.items():
        print(f"{model_name:<10} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} "
              f"{metrics['MAPE_%']:<10.2f} {metrics['CRPS']:<10.4f}")

    return results, val_results


if __name__ == "__main__":
    from config import KANParams, DataPaths

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run comparison for wind
    _, wind_metrics = run_comparison(DataPaths["wind_pred"], KANParams,
                                     asset_name="Gansu Wind", device=device)
    # Run comparison for solar
    _, solar_metrics = run_comparison(DataPaths["solar_pred"], KANParams,
                                      asset_name="Gansu Solar", device=device)

    # Save results
    os.makedirs("results/tables", exist_ok=True)
    import pandas as pd
    summary = []
    for name, metrics in [("Wind", wind_metrics), ("Solar", solar_metrics)]:
        for model, vals in metrics.items():
            summary.append({"Asset": name, "Model": model, **vals})
    pd.DataFrame(summary).to_csv("results/tables/baseline_comparison_full.csv", index=False)
    print("\nSaved: results/tables/baseline_comparison_full.csv")
