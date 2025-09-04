# Insurance Claims Simulation - Starter Notebook
# This cell sets up a full, runnable simulation for your resume project.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log

# --- Reproducibility ---
rng = np.random.default_rng(42)

# --- Core simulation helpers ---
def simulate_annual_losses(n_years: int, lam: float, severity_dist: str, severity_params: dict, rng=rng):
    """
    Simulate aggregate annual losses for n_years.
    Frequency: Poisson(lam)
    Severity: one of {'exponential','normal','lognormal'}
      - exponential: {'mean': m}
      - normal: {'mean': m, 'sd': s} (truncated at 0)
      - lognormal: {'mu': mu, 'sigma': sigma} (parameters of the underlying normal)
    Returns: np.array of length n_years (aggregate annual loss)
    """
    counts = rng.poisson(lam=lam, size=n_years)
    totals = np.zeros(n_years, dtype=float)

    if severity_dist == 'exponential':
        scale = severity_params['mean']
        for i, c in enumerate(counts):
            if c > 0:
                totals[i] = rng.exponential(scale, size=c).sum()

    elif severity_dist == 'normal':
        mean = severity_params['mean']
        sd = severity_params['sd']
        for i, c in enumerate(counts):
            if c > 0:
                sev = rng.normal(mean, sd, size=c)
                sev = np.clip(sev, 0, None)  # truncate at 0 to avoid negative severities
                totals[i] = sev.sum()

    elif severity_dist == 'lognormal':
        mu = severity_params['mu']
        sigma = severity_params['sigma']
        for i, c in enumerate(counts):
            if c > 0:
                totals[i] = rng.lognormal(mu, sigma, size=c).sum()
    else:
        raise ValueError("Unsupported severity_dist")

    return totals

def stop_loss(total_losses: np.ndarray, attachment: float, limit: float = np.inf):
    """
    Aggregate stop-loss reinsurance:
    - ceded = min(max(total - attachment, 0), limit)
    - net   = total - ceded
    Returns ceded, net arrays.
    """
    excess = np.maximum(total_losses - attachment, 0.0)
    ceded = np.minimum(excess, limit)
    net = total_losses - ceded
    return ceded, net

def metrics(x: np.ndarray, label: str):
    return {
        "Scenario": label,
        "Mean": float(np.mean(x)),
        "Variance": float(np.var(x)),
        "VaR 95%": float(np.percentile(x, 95)),
    }

# --- Simulation configuration ---
N_YEARS = 50000      # number of simulated years
LAM = 10             # expected claims per year (frequency parameter)
# Severity parameter choices
exp_params = {"mean": 5000.0}                       # mean $5,000
norm_params = {"mean": 5000.0, "sd": 2000.0}        # truncated at 0
# Lognormal with target mean around 7000: mean = exp(mu + 0.5*sigma^2)
# choose sigma=0.9 -> mu = ln(7000) - 0.5*(0.9^2)
sigma = 0.9
mu = log(7000) - 0.5*(sigma**2)
lognorm_params = {"mu": mu, "sigma": sigma}

# --- Run simulations for three severities ---
results = {}
for sev_name, sev_params in [
    ("Exponential", exp_params),
    ("Normal (truncated ≥0)", norm_params),
    ("Lognormal", lognorm_params),
]:
    gross = simulate_annual_losses(N_YEARS, LAM, sev_name.split()[0].lower(), sev_params, rng=rng)
    ceded, net = stop_loss(gross, attachment=80000.0, limit=100000.0)  # sample stop-loss: attach 80k, limit 100k
    results[sev_name] = {"gross": gross, "ceded": ceded, "net": net}

# --- Build metrics table ---
rows = []
for sev_name, data in results.items():
    g = data["gross"]
    c = data["ceded"]
    n = data["net"]
    rows.append({"Severity": sev_name, "Type": "Gross", **metrics(g, sev_name + " Gross")})
    rows.append({"Severity": sev_name, "Type": "Net (Stop-Loss 80k xs 100k)", **metrics(n, sev_name + " Net")})
    rows.append({"Severity": sev_name, "Type": "Ceded", **metrics(c, sev_name + " Ceded")})
metrics_df = pd.DataFrame(rows)[["Severity","Type","Mean","Variance","VaR 95%"]]

# Display metrics in a clean table format
print("Insurance Claims Simulation - Metrics")
print("=" * 50)
print(metrics_df.to_string(index=False))
print("\n")

# --- Visualizations ---
# 1) Severity *shape* examples (draw 100k severities from each to visualize distribution)
sev_draws = {
    "Exponential": rng.exponential(exp_params["mean"], size=100000),
    "Normal (truncated ≥0)": np.clip(rng.normal(norm_params["mean"], norm_params["sd"], size=100000), 0, None),
    "Lognormal": rng.lognormal(lognorm_params["mu"], lognorm_params["sigma"], size=100000),
}

for name, arr in sev_draws.items():
    plt.figure(figsize=(10, 6))
    plt.hist(arr, bins=100, alpha=0.7, edgecolor='black')
    plt.title(f"Severity Distribution: {name}")
    plt.xlabel("Claim Size ($)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 2) Annual aggregate losses (gross) histograms
for sev_name, data in results.items():
    plt.figure(figsize=(10, 6))
    plt.hist(data["gross"], bins=100, alpha=0.7, edgecolor='black')
    plt.title(f"Annual Aggregate Losses (Gross) — {sev_name}")
    plt.xlabel("Annual Total Loss ($)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 3) Risk transfer effect for Lognormal: Gross vs Net
log_gross = results["Lognormal"]["gross"]
log_net = results["Lognormal"]["net"]

plt.figure(figsize=(10, 6))
plt.hist(log_gross, bins=100, alpha=0.6, label="Gross", edgecolor='black')
plt.hist(log_net, bins=100, alpha=0.6, label="Net (Stop-Loss)", edgecolor='black')
plt.title("Risk Transfer: Gross vs Net — Lognormal Severity")
plt.xlabel("Annual Total Loss ($)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Save artifacts for your portfolio ---
# Save metrics to CSV
try:
    metrics_path = "insurance_claims_simulation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved to: {metrics_path}")
except Exception as e:
    print(f"Could not save CSV: {e}")

# Save a simple, well-documented Python script version
try:
    script_path = "insurance_claims_simulation.py"
    with open(script_path, "w") as f:
        f.write('''
import numpy as np

def simulate_annual_losses(n_years, lam, severity_dist, params, seed=42):
    """
    Simulate annual insurance losses with Poisson frequency and specified severity distribution.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=lam, size=n_years)
    totals = np.zeros(n_years, dtype=float)

    if severity_dist == 'exponential':
        scale = params['mean']
        for i, c in enumerate(counts):
            if c > 0:
                totals[i] = rng.exponential(scale, size=c).sum()

    elif severity_dist == 'normal':
        mean = params['mean']; sd = params['sd']
        for i, c in enumerate(counts):
            if c > 0:
                sev = rng.normal(mean, sd, size=c)
                sev = np.clip(sev, 0, None)
                totals[i] = sev.sum()

    elif severity_dist == 'lognormal':
        mu = params['mu']; sigma = params['sigma']
        for i, c in enumerate(counts):
            if c > 0:
                totals[i] = rng.lognormal(mu, sigma, size=c).sum()
    else:
        raise ValueError("Unsupported severity_dist")

    return totals

def stop_loss(total_losses, attachment, limit=np.inf):
    """
    Apply stop-loss reinsurance to aggregate losses.
    """
    excess = np.maximum(total_losses - attachment, 0.0)
    ceded = np.minimum(excess, limit)
    net = total_losses - ceded
    return ceded, net

def calculate_metrics(losses, label):
    """
    Calculate key risk metrics for a loss distribution.
    """
    return {
        "Mean": np.mean(losses),
        "Variance": np.var(losses),
        "VaR 95%": np.percentile(losses, 95)
    }

if __name__ == '__main__':
    # Example usage
    N_YEARS = 50000
    LAM = 10

    # Severity parameters
    exp_params = {"mean": 5000.0}
    norm_params = {"mean": 5000.0, "sd": 2000.0}
    sigma = 0.9
    mu = np.log(7000) - 0.5*(sigma**2)
    lognorm_params = {"mu": mu, "sigma": sigma}

    print("Insurance Claims Simulation Results")
    print("=" * 40)
    
    for name, p in [('exponential', exp_params), ('normal', norm_params), ('lognormal', lognorm_params)]:
        gross = simulate_annual_losses(N_YEARS, LAM, name, p)
        ceded, net = stop_loss(gross, attachment=80000.0, limit=100000.0)
        
        gross_metrics = calculate_metrics(gross, "Gross")
        net_metrics = calculate_metrics(net, "Net")
        
        print(f"\\n{name.title()} Severity:")
        print(f"Gross - Mean: ${gross_metrics['Mean']:,.2f}, VaR 95%: ${gross_metrics['VaR 95%']:,.2f}")
        print(f"Net   - Mean: ${net_metrics['Mean']:,.2f}, VaR 95%: ${net_metrics['VaR 95%']:,.2f}")
''')
    print(f"✓ Python script saved to: {script_path}")
except Exception as e:
    print(f"Could not save Python script: {e}")

print("\nSimulation completed successfully!")