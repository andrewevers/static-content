import numpy as np
from scipy.stats import qmc, norm
from scipy.interpolate import CubicSpline
 
# ── Constants ──────────────────────────────────────────────────────
TENOR_MONTHS = np.array([1, 3, 6, 12, 24, 36, 60, 84, 120], dtype=float)
FACTOR_NAMES = ['CPR', 'CDR', 'ABS_Spread', 'SOFR']
N_FACTORS    = 4
 
BASE_PARAMS = {
    'cpr_mean':    0.08,   'cpr_std':     0.04,
    'cpr_min':     0.01,   'cpr_max':     0.35,
    'cdr_mean':    0.015,  'cdr_std':     0.010,
    'cdr_min':     0.001,  'cdr_max':     0.12,
    'spread_mean': 0.0175, 'spread_std':  0.0050,
    'spread_min':  0.0050, 'spread_max':  0.0500,
    'sofr_mean':   0.000,  'sofr_std':    0.0075,
    'sofr_min':   -0.025,  'sofr_max':    0.025,
}
 
# ── 1. SOFR Forward Curve ──────────────────────────────────────────
def sofr_curve(rates, n_months=120, tenor_months=None):
    rates         = np.asarray(rates, dtype=float).flatten()
    t_months      = np.asarray(tenor_months or TENOR_MONTHS, dtype=float)
    cs            = CubicSpline(t_months, rates, bc_type='not-a-knot')
    months        = np.arange(1, n_months + 1, dtype=float)
    forward_rates = np.clip(cs(months), 0.001, 0.25)
    return {
        'months':        months,
        'forward_rates': forward_rates,
        'tenor_months':  t_months,
        'tenor_rates':   rates,
    }
 
# ── 2. Correlation Matrix + Cholesky ──────────────────────────────
def build_corr_matrix(corr_cpr_sofr,    corr_cdr_spread,
                      corr_cpr_cdr,     corr_sofr_spread,
                      corr_cpr_spread=0.15, corr_cdr_sofr=0.20):
    C = np.eye(N_FACTORS)
    for (i, j, v) in [
        (0, 1, corr_cpr_cdr),
        (0, 2, corr_cpr_spread),
        (0, 3, corr_cpr_sofr),
        (1, 2, corr_cdr_spread),
        (1, 3, corr_cdr_sofr),
        (2, 3, corr_sofr_spread),
    ]:
        C[i, j] = v
        C[j, i] = v
    eigenvalues = np.linalg.eigvalsh(C)
    min_eig     = float(eigenvalues.min())
    is_psd      = bool(min_eig > -1e-10)
    if not is_psd:
        C += (-min_eig + 1e-8) * np.eye(N_FACTORS)
        d  = np.sqrt(np.diag(C))
        C  = C / np.outer(d, d)
    L = np.linalg.cholesky(C)
    return {
        'corr_matrix':  C,
        'cholesky':     L,
        'is_psd':       is_psd,
        'min_eigenval': min_eig,
        'factor_names': FACTOR_NAMES,
    }
 
# ── 3. Sobol QMC Draws ────────────────────────────────────────────
def sobol_engine(n_draws=16384, n_factors=N_FACTORS, scramble=True, seed=42):
    is_p2 = (n_draws & (n_draws - 1)) == 0
    if not is_p2:
        n_draws = int(2 ** np.ceil(np.log2(n_draws)))
    sampler         = qmc.Sobol(d=n_factors, scramble=scramble, seed=seed)
    uniform         = sampler.random(n_draws)
    uniform_clipped = np.clip(uniform, 1e-10, 1 - 1e-10)
    normal          = norm.ppf(uniform_clipped)
    return {
        'uniform':   uniform,
        'normal':    normal,
        'n_draws':   n_draws,
        'n_factors': n_factors,
        'is_power2': is_p2,
    }
 
# ── 4. Correlated Path Simulation ────────────────────────────────
def simulate_paths(corr_result, sobol_result, params=None):
    p      = {**BASE_PARAMS, **(params or {})}
    L      = corr_result['cholesky']
    Z      = sobol_result['normal']
    Z_corr = (L @ Z.T).T
 
    def tn(z, mean, std, lo, hi):
        return np.clip(mean + std * z, lo, hi)
 
    return {
        'cpr':               tn(Z_corr[:,0], p['cpr_mean'],    p['cpr_std'],    p['cpr_min'],    p['cpr_max']),
        'cdr':               tn(Z_corr[:,1], p['cdr_mean'],    p['cdr_std'],    p['cdr_min'],    p['cdr_max']),
        'spread':            tn(Z_corr[:,2], p['spread_mean'], p['spread_std'], p['spread_min'], p['spread_max']),
        'sofr_shock':        tn(Z_corr[:,3], p['sofr_mean'],   p['sofr_std'],   p['sofr_min'],   p['sofr_max']),
        'correlated_normal': Z_corr,
        'n_draws':           sobol_result['n_draws'],
        'factor_names':      FACTOR_NAMES,
        'params_used':       p,
    }
 
# ── 5. Master Run Function ────────────────────────────────────────
def run_simulation(sofr_rates,
                   corr_cpr_sofr=-0.40,  corr_cdr_spread=0.60,
                   corr_cpr_cdr=0.10,    corr_sofr_spread=0.30,
                   n_draws=16384,        params=None):
    rates = np.asarray(sofr_rates, dtype=float).flatten()
    curve = sofr_curve(rates)
    corr  = build_corr_matrix(corr_cpr_sofr, corr_cdr_spread,
                               corr_cpr_cdr,  corr_sofr_spread)
    sobol = sobol_engine(int(n_draws))
    paths = simulate_paths(corr, sobol, params)
    return {
        'curve':       curve,
        'corr_matrix': corr['corr_matrix'],
        'cholesky':    corr['cholesky'],
        'is_psd':      corr['is_psd'],
        'paths':       paths,
        'n_draws':     sobol['n_draws'],
    }
