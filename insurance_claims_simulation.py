
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
        
        print(f"\n{name.title()} Severity:")
        print(f"Gross - Mean: ${gross_metrics['Mean']:,.2f}, VaR 95%: ${gross_metrics['VaR 95%']:,.2f}")
        print(f"Net   - Mean: ${net_metrics['Mean']:,.2f}, VaR 95%: ${net_metrics['VaR 95%']:,.2f}")
