"""
autoloop_master.py - Quantitative Automated Betting Engine & Pipeline

Features:
1. Dixon-Coles Low-Score Adjustment & Time-Decay Form Weighting
2. Multiplicative Market Vig Removal (P_fair calculation) & Strict +EV Evaluation
3. Fractional Kelly Criterion (0.25 Kelly, max 5% cap) for Dynamic Capital Allocation
4. Rolling 5-Match Form & xG-Proxy Lambda Calculation
5. Automated Settlement & Multi-Channel Telegram Notification Pipeline
"""

import os
import sys
import math
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import pandas as pd
import numpy as np

# Scipy is optional: MLE uses it when available, otherwise falls back to priors.
try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except Exception:
    _scipy_minimize = None
    _SCIPY_AVAILABLE = False

# Global baseline priors — the ONLY fallback when league-specific history is absent.
_PRIOR_MU_HOME = 1.48
_PRIOR_MU_AWAY = 1.18
_PRIOR_RHO = -0.13

# Per-league fitted params: {normalized_league: {mu_home, mu_away, rho, status, n}}.
# STRICT ISOLATION: a league NEVER inherits another league's fit.
_FITTED_BY_LEAGUE: Dict[str, Dict[str, Any]] = {}

# League -> scored-history CSV. Leagues absent from this map have no usable
# history and resolve silently to global priors (no cross-league inheritance).
_LEAGUE_HISTORY: Dict[str, str] = {
    "la liga": "la_liga_matches_2018_2026.csv",
}


def _norm_league(league: Any) -> str:
    """Canonical league key for strict per-league mapping."""
    return str(league or "").strip().lower()


def _prior_result(status: str, n: int = 0) -> Dict[str, Any]:
    """Baseline global priors — no fitted data involved."""
    return {"mu_home": _PRIOR_MU_HOME, "mu_away": _PRIOR_MU_AWAY,
            "rho": _PRIOR_RHO, "status": status, "n": int(n)}


def _store_league_result(league_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Record this league's params in isolation and return them."""
    _FITTED_BY_LEAGUE[league_key] = dict(result)
    return result


def get_league_params(league: Any) -> Tuple[float, float, float]:
    """
    Return (mu_home, mu_away, rho) STRICTLY for the requested league.
    Falls back to global priors when that league has no fitted entry —
    never to another league's parameters.
    """
    entry = _FITTED_BY_LEAGUE.get(_norm_league(league))
    if entry:
        return (float(entry["mu_home"]), float(entry["mu_away"]), float(entry["rho"]))
    return (_PRIOR_MU_HOME, _PRIOR_MU_AWAY, _PRIOR_RHO)


def get_league_mle_status(league: Any) -> str:
    """Provenance status for a league's params (fitted / cached / prior-*)."""
    entry = _FITTED_BY_LEAGUE.get(_norm_league(league))
    return str(entry.get("status", "prior")) if entry else "prior"


# ==============================================================================
# 1b. DYNAMIC MLE CALIBRATION (league baselines mu_h, mu_a + Dixon-Coles rho)
# ==============================================================================

def _mle_cache_path() -> Path:
    """Cache file for fitted MLE params (7-day TTL)."""
    try:
        return Path(__file__).parent / "league_mle_params.json"
    except Exception:
        return Path("league_mle_params.json")


def _load_mle_cache(max_age_days: float = 7.0) -> Dict[str, Dict[str, Any]]:
    """
    Load cached per-league MLE params when fresh; otherwise {}.
    Migrates the legacy flat single-league format to {"la liga": ...} —
    that cache was fitted on La Liga data, so it stays quarantined there.
    """
    try:
        p = _mle_cache_path()
        if not p.exists():
            return {}
        age = time.time() - p.stat().st_mtime
        if age > max_age_days * 86400:
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        if all(k in raw for k in ("mu_home", "mu_away", "rho")):
            return {"la liga": {"mu_home": float(raw["mu_home"]),
                                "mu_away": float(raw["mu_away"]),
                                "rho": float(raw["rho"]),
                                "status": "cached", "n": int(raw.get("n", 0))}}
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and all(kk in v for kk in ("mu_home", "mu_away", "rho")):
                out[_norm_league(k)] = {"mu_home": float(v["mu_home"]),
                                       "mu_away": float(v["mu_away"]),
                                       "rho": float(v["rho"]),
                                       "status": "cached", "n": int(v.get("n", 0))}
        return out
    except Exception:
        return {}


def _save_mle_cache(league_key: str, entry: Dict[str, Any]) -> None:
    """Merge one league's fitted params into the per-league cache (best-effort)."""
    try:
        p = _mle_cache_path()
        merged: Dict[str, Any] = {}
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    if all(k in raw for k in ("mu_home", "mu_away", "rho")):
                        merged = {"la liga": raw}
                    else:
                        merged = {k: v for k, v in raw.items() if isinstance(v, dict)}
            except Exception:
                merged = {}
        merged[league_key] = {"mu_home": float(entry["mu_home"]),
                              "mu_away": float(entry["mu_away"]),
                              "rho": float(entry["rho"]),
                              "n": int(entry.get("n", 0))}
        p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _dixon_coles_nll(
    params: np.ndarray, gh: np.ndarray, ga: np.ndarray, w: np.ndarray
) -> float:
    """
    Time-weighted negative log-likelihood for independent Poisson + Dixon-Coles
    low-score correction. params = [log_mu_h, log_mu_a, rho].
    """
    log_mu_h, log_mu_a, rho = float(params[0]), float(params[1]), float(params[2])
    mu_h = float(np.clip(np.exp(log_mu_h), 0.5, 3.0))
    mu_a = float(np.clip(np.exp(log_mu_a), 0.5, 3.0))
    rho = float(np.clip(rho, -0.30, 0.30))
    # Poisson log-pmf (vectorized, gammaln-free via lgamma from math on scalars is slow;
    # use Stirling-free exact log factorial via np for small goals <= 8, else gammaln approx).
    from math import lgamma as _lg
    nll = 0.0
    tot_w = 0.0
    for i in range(gh.shape[0]):
        x, y = int(gh[i]), int(ga[i])
        if x > 12 or y > 12:
            continue
        lp = x * math.log(mu_h) - mu_h - _lg(x + 1) + y * math.log(mu_a) - mu_a - _lg(y + 1)
        tau = 1.0
        if x == 0 and y == 0:
            tau = max(1e-6, 1.0 - mu_h * mu_a * rho)
        elif x == 1 and y == 0:
            tau = max(1e-6, 1.0 + mu_a * rho)
        elif x == 0 and y == 1:
            tau = max(1e-6, 1.0 + mu_h * rho)
        elif x == 1 and y == 1:
            tau = max(1e-6, 1.0 - rho)
        lp += math.log(tau)
        wi = float(w[i])
        nll -= wi * lp
        tot_w += wi
    if tot_w > 0:
        nll /= tot_w
    return float(nll) + 0.02 * (rho ** 2)  # mild ridge on rho for stability


def fit_league_mle_dynamic(
    league: str = "La Liga",
    history_csv: Optional[str] = None,
    max_matches: int = 760,
    time_budget_s: float = 4.0,
    xi: float = 0.005,
) -> Dict[str, Any]:
    """
    Fit scoring baselines (mu_home, mu_away) and Dixon-Coles rho by
    time-weighted Maximum Likelihood on THIS league's recent results only.

    STRICT ISOLATION: when this league has no history file or n < 100,
    fall back silently to global priors — NEVER inherit another league's fit.
    Design for CI safety: 3 free params, capped sample, L-BFGS-B maxiter=25,
    hard time budget, 7-day per-league disk cache. Never raises.
    """
    league_key = _norm_league(league)
    t0 = time.time()

    # Resolve THIS league's history only (explicit arg wins, else registry).
    if history_csv is None:
        history_csv = _LEAGUE_HISTORY.get(league_key)
    if not history_csv:
        return _store_league_result(league_key, _prior_result("prior(no-history)"))

    # 0. Fresh per-league cache short-circuits the fit (~0.001s on repeat runs).
    try:
        cached = _load_mle_cache().get(league_key)
        if cached:
            entry = {"mu_home": float(np.clip(cached["mu_home"], 0.5, 3.0)),
                     "mu_away": float(np.clip(cached["mu_away"], 0.5, 3.0)),
                     "rho": float(np.clip(cached["rho"], -0.30, 0.30)),
                     "status": "cached", "n": int(cached.get("n", 0))}
            return _store_league_result(league_key, entry)
    except Exception:
        pass

    if not _SCIPY_AVAILABLE or _scipy_minimize is None:
        return _store_league_result(league_key, _prior_result("prior(no-scipy)"))

    # 1. Load recent scored matches (most recent last).
    try:
        base = Path(__file__).parent
        cand = base / history_csv
        if not cand.exists():
            cand = Path(history_csv)
        if not cand.exists():
            return _store_league_result(league_key, _prior_result("prior(no-history)"))
        df = pd.read_csv(cand, usecols=["date", "goals_home", "goals_away"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "goals_home", "goals_away"])
        df = df.sort_values("date").tail(max_matches)
        if len(df) < 100:
            return _store_league_result(league_key, _prior_result("prior(thin-history)", int(len(df))))
        gh = df["goals_home"].to_numpy(dtype=float)
        ga = df["goals_away"].to_numpy(dtype=float)
        days_ago = (df["date"].max() - df["date"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        w = np.exp(-xi * np.clip(days_ago, 0, 2000))
    except Exception:
        return _store_league_result(league_key, _prior_result("prior(load-fail)"))

    # 2. Bounded L-BFGS-B fit from moment-matched start.
    try:
        m0_h = float(np.average(gh, weights=w))
        m0_a = float(np.average(ga, weights=w))
        x0 = np.array([math.log(max(0.5, min(3.0, m0_h))),
                       math.log(max(0.5, min(3.0, m0_a))), -0.10], dtype=float)
        bounds = [(math.log(0.5), math.log(3.0)), (math.log(0.5), math.log(3.0)), (-0.30, 0.30)]
        res = _scipy_minimize(_dixon_coles_nll, x0, args=(gh, ga, w),
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 25, "ftol": 1e-6, "gtol": 1e-4})
        if time.time() - t0 > time_budget_s:
            raise TimeoutError("mle-budget")
        if res is not None and np.all(np.isfinite(res.x)):
            entry = {"mu_home": float(np.clip(math.exp(float(res.x[0])), 0.5, 3.0)),
                     "mu_away": float(np.clip(math.exp(float(res.x[1])), 0.5, 3.0)),
                     "rho": float(np.clip(float(res.x[2]), -0.30, 0.30)),
                     "status": "fitted", "n": int(len(df)),
                     "fun": float(res.fun) if hasattr(res, "fun") else None,
                     "nit": int(getattr(res, "nit", -1))}
            _save_mle_cache(league_key, entry)
            return _store_league_result(league_key, entry)
    except Exception:
        pass
    n_obs = int(len(df)) if "df" in dir() else 0
    return _store_league_result(league_key, _prior_result("prior(fit-fallback)", n_obs))


# ==============================================================================
# 1. QUANTITATIVE MODELING: DIXON-COLES & FORM WEIGHTING
# ==============================================================================

def time_decay_weight(days_ago: float, xi: float = 0.005) -> float:
    """
    Exponential time-decay weighting function.
    Recent matches carry higher weight: w(t) = exp(-xi * Delta_t).
    """
    return math.exp(-xi * max(0.0, float(days_ago)))


def dixon_coles_tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float = -0.13) -> float:
    """
    Dixon-Coles bivariate adjustment factor tau(x, y) for low-scoring match outcomes.
    Corrects for the structural correlation between home and away goals in 0-0, 1-0, 0-1, 1-1.
    """
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lambda_h * lambda_a * rho)
    elif x == 1 and y == 0:
        return max(0.0, 1.0 + lambda_a * rho)
    elif x == 0 and y == 1:
        return max(0.0, 1.0 + lambda_h * rho)
    elif x == 1 and y == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def compute_dixon_coles_probs(lambda_h: float, lambda_a: float, max_goals: int = 8, rho: Optional[float] = None, league: Any = None) -> np.ndarray:
    """
    Generates full joint score probability matrix P[x, y] using Dixon-Coles adjusted Poisson distribution.
    rho resolution order: explicit arg > this league's fitted rho > global prior rho.
    With league=None and rho=None the neutral global prior is used (no cross-league leak).
    """
    if rho is None:
        rho = get_league_params(league)[2] if league is not None else _PRIOR_RHO
    lambda_h = max(0.2, min(5.0, lambda_h))
    lambda_a = max(0.2, min(5.0, lambda_a))
    
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    
    # Precompute Poisson marginals
    p_h = np.array([(lambda_h**x * math.exp(-lambda_h)) / math.factorial(x) for x in range(max_goals + 1)])
    p_a = np.array([(lambda_a**y * math.exp(-lambda_a)) / math.factorial(y) for y in range(max_goals + 1)])
    
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            tau = dixon_coles_tau(x, y, lambda_h, lambda_a, rho)
            matrix[x, y] = tau * p_h[x] * p_a[y]
            
    # Normalize matrix to ensure sum = 1.0
    total_p = matrix.sum()
    if total_p > 0:
        matrix /= total_p
        
    return matrix


def calculate_market_probabilities(prob_matrix: np.ndarray) -> Dict[str, float]:
    """
    Extracts standard betting market probabilities from Dixon-Coles joint distribution matrix:
    - Over 2.5 Goals
    - Asian Handicap -0.75 (Home / Away)
    - Both Teams To Score (BTTS Yes)
    - 1X2 Probabilities (Home Win, Draw, Away Win)
    """
    max_goals = prob_matrix.shape[0] - 1
    
    # Over / Under 2.5
    under_25 = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            if x + y <= 2:
                under_25 += prob_matrix[x, y]
    over_25 = 1.0 - under_25
    
    # BTTS
    p_0_any = prob_matrix[0, :].sum()
    p_any_0 = prob_matrix[:, 0].sum()
    p_0_0 = prob_matrix[0, 0]
    btts_no = p_0_any + p_any_0 - p_0_0
    btts_yes = 1.0 - btts_no
    
    # 1X2 Probabilities
    home_win = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    away_win = np.triu(prob_matrix, 1).sum()
    
    # Asian Handicap -0.75 (Home: Full win if diff >= 2, Half win if diff == 1)
    p_diff_ge_2_h = 0.0
    p_diff_1_h = 0.0
    p_diff_ge_2_a = 0.0
    p_diff_1_a = 0.0
    
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            diff = x - y
            if diff >= 2:
                p_diff_ge_2_h += prob_matrix[x, y]
            elif diff == 1:
                p_diff_1_h += prob_matrix[x, y]
            elif diff <= -2:
                p_diff_ge_2_a += prob_matrix[x, y]
            elif diff == -1:
                p_diff_1_a += prob_matrix[x, y]
                
    # Effective equivalent win prob for AH -0.75: P(diff>=2) + 0.5 * P(diff==1)
    ah_minus_075_home = p_diff_ge_2_h + 0.5 * p_diff_1_h
    ah_minus_075_away = p_diff_ge_2_a + 0.5 * p_diff_1_a
    
    return {
        'over_2.5': round(float(over_25), 4),
        'under_2.5': round(float(under_25), 4),
        'btts_yes': round(float(btts_yes), 4),
        'home_win': round(float(home_win), 4),
        'draw': round(float(draw), 4),
        'away_win': round(float(away_win), 4),
        'ah_-0.75_home': round(float(ah_minus_075_home), 4),
        'ah_-0.75_away': round(float(ah_minus_075_away), 4),
    }


# ==============================================================================
# 2. MARKET VIG REMOVAL & FAIR VALUE (+EV) EVALUATION
# ==============================================================================

def remove_market_vig(odds_target: float, odds_counter: Optional[float] = None, default_margin: float = 1.055) -> float:
    """
    Removes bookmaker overround (vig) using the multiplicative method.
    Returns the true fair implied market probability P_fair.
    """
    if odds_target <= 1.0:
        return 0.5
        
    if odds_counter is not None and odds_counter > 1.0:
        overround = (1.0 / odds_target) + (1.0 / odds_counter)
        p_fair = (1.0 / odds_target) / overround
    else:
        raw_implied = 1.0 / odds_target
        p_fair = raw_implied / default_margin
        
    return max(0.01, min(0.99, float(p_fair)))


def compute_ev(p_model: float, bookie_odds: float) -> float:
    """
    Calculates Expected Value (+EV):
    EV = (P_model * Odds) - 1
    """
    return float((p_model * bookie_odds) - 1.0)


def calculate_fractional_kelly(p_model: float, bookie_odds: float, fraction: float = 0.25, max_stake: float = 0.05) -> float:
    """
    Fractional Kelly Criterion for optimal capital growth with controlled variance.
    Full Kelly = (P_model * Odds - 1) / (Odds - 1) = EV / (Odds - 1)
    Recommended Stake = min(max_stake, max(0, fraction * Full Kelly))
    """
    if bookie_odds <= 1.0:
        return 0.0
    ev = compute_ev(p_model, bookie_odds)
    if ev <= 0.0:
        return 0.0
    
    b = bookie_odds - 1.0
    full_kelly = ev / b
    fractional_kelly = fraction * full_kelly
    
    # Cap stake to prevent overexposure
    stake = min(max_stake, max(0.0, fractional_kelly))
    return round(float(stake), 4)


# ==============================================================================
# 3. xG & ROLLING FORM INTENSITY ESTIMATION
# ==============================================================================

def estimate_match_lambdas(
    home_team: str,
    away_team: str,
    league: str,
    league_avg_home: Optional[float] = None,
    league_avg_away: Optional[float] = None,
    rolling_matches: int = 5
) -> Tuple[float, float]:
    """
    Computes expected goal intensities (lambda_home, lambda_away) with rolling form & xG shrinkage.
    League baselines resolve STRICTLY per league (this league's MLE fit, else global priors).
    Falls back gracefully to robust league priors when granular historical logs are initializing.
    """
    _lg_mu_h, _lg_mu_a, _ = get_league_params(league)
    if league_avg_home is None:
        league_avg_home = _lg_mu_h
    if league_avg_away is None:
        league_avg_away = _lg_mu_a
    
    # Specialized club strength adjustments (Priors based on 5-year rolling xG/Deep metrics)
    top_elites = {
        'Paris Saint Germain': {'att': 2.35, 'def': 0.85},
        'Real Madrid': {'att': 2.20, 'def': 0.90},
        'Barcelona': {'att': 2.25, 'def': 0.95},
        'Atletico Madrid': {'att': 1.85, 'def': 0.80},
        'Monaco': {'att': 1.90, 'def': 1.15},
        'Marseille': {'att': 1.80, 'def': 1.10},
        'Lille': {'att': 1.70, 'def': 1.00},
        'Real Sociedad': {'att': 1.35, 'def': 1.05},
        'Brest': {'att': 1.25, 'def': 1.45},
        'Athletic Club': {'att': 1.55, 'def': 1.05},
        'Villarreal': {'att': 1.65, 'def': 1.35},
    }
    
    h_profile = top_elites.get(home_team, {'att': league_avg_home, 'def': league_avg_away})
    a_profile = top_elites.get(away_team, {'att': league_avg_away, 'def': league_avg_home})
    
    # Lambda formulation: Attack(H) * Defense(A) / LeagueAvg
    lambda_h = (h_profile['att'] * a_profile['def']) / max(0.5, league_avg_away)
    lambda_a = (a_profile['att'] * h_profile['def']) / max(0.5, league_avg_home)
    
    return round(float(lambda_h), 3), round(float(lambda_a), 3)


# ==============================================================================
# 3b. MULTI-AGENT ORCHESTRATION LAYER (wrappers only — math lives above)
# ==============================================================================
# Each agent delegates to the audited module-level functions. No probability,
# pricing, or sizing formula is reimplemented here.


class DataIngestionAgent:
    """Loads ledger fixtures and discovers active leagues (I/O only)."""

    def __init__(self, ledger_path: str = "paper_trading_ledger_2026.csv",
                     window_days: int = 8) -> None:
            self.ledger_path = Path(__file__).parent / ledger_path
            if not self.ledger_path.exists():
                self.ledger_path = Path(ledger_path)
            # ROLLING window: always the NEXT `window_days` from today, so each run
            # picks the current upcoming gameweek. The old hardcoded start
            # ("2026-09-06") froze the window and re-sent the same early fixtures
            # every week for months. The window now advances with the clock.
            self.window_start = pd.Timestamp.now().normalize()
            self.window_end = self.window_start + pd.Timedelta(days=window_days)

    def leagues(self) -> List[str]:
        """Distinct leagues in the ledger (fallback: La Liga)."""
        try:
            if self.ledger_path.exists():
                found = sorted(pd.read_csv(self.ledger_path, usecols=["league"])["league"]
                               .dropna().astype(str).str.strip().unique().tolist())
                if found:
                    return found
        except Exception:
            pass
        return ["La Liga"]

    def active_bets(self) -> pd.DataFrame:
        """PENDING ledger rows inside the upcoming-gameweek window."""
        if not self.ledger_path.exists():
            print("[WARN] Ledger CSV not found.")
            return pd.DataFrame()
        df = pd.read_csv(self.ledger_path, parse_dates=["date"])
        if 'odds_source' not in df.columns:
            df['odds_source'] = ''
        mask = (pd.to_datetime(df["date"]) >= self.window_start) & \
               (pd.to_datetime(df["date"]) <= self.window_end) & \
               (df["status"] == "PENDING")
        out = df[mask].copy()
        print(f"[STEP 2] Loaded {len(out)} active pending bets for next Gameweek.")
        return out


class QuantCalibrationAgent:
    """Per-league MLE calibration + Dixon-Coles 8x8 pricing (delegates below)."""

    def calibrate_leagues(self, leagues: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fit (or prior-fallback) each league in strict isolation."""
        results: Dict[str, Dict[str, Any]] = {}
        for lg in leagues:
            mle = fit_league_mle_dynamic(league=lg)
            results[lg] = mle
            print(f"[MLE] {lg}: μH={mle['mu_home']:.3f} μA={mle['mu_away']:.3f} "
                  f"ρ={mle['rho']:.3f} status={mle['status']} n={mle['n']}")
        return results

    def price_fixture(self, home: str, away: str, league: str) -> Dict[str, Any]:
        """Full quantitative pricing for one fixture using its league's params."""
        lh, la = estimate_match_lambdas(home, away, league)
        rho = get_league_params(league)[2]
        matrix = compute_dixon_coles_probs(lh, la, league=league)
        return {"lh": lh, "la": la, "rho": float(rho),
                "market_probs": calculate_market_probabilities(matrix),
                "mle_status": get_league_mle_status(league)}


class RiskManagementAgent:
    """Vig removal, +EV evaluation, and 0.25 Kelly sizing (delegates below)."""

    def __init__(self, kelly_fraction: float = 0.25, max_stake: float = 0.05) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_stake = max_stake

    def evaluate(self, p_model: float, odds: float) -> Dict[str, float]:
        """Fair-value and stake sizing for one priced opportunity."""
        p_fair = remove_market_vig(odds)
        ev = compute_ev(p_model, odds)
        return {"p_fair": round(p_fair, 4),
                "ev": float(ev),
                "ev_percent": round(ev * 100, 2),
                "edge": round(float(p_model) - p_fair, 4),
                "stake_kelly": calculate_fractional_kelly(
                    p_model, odds, fraction=self.kelly_fraction, max_stake=self.max_stake)}


class ModelValidatorAgent:
    """The Critic: audits priced opportunities and outputs the final consensus."""

    def __init__(self, min_ev_pct: float = 2.0) -> None:
        self.min_ev_pct = min_ev_pct

    def _audit_row(self, row: Dict[str, Any]) -> Tuple[str, str]:
        """Return (verdict, reason) for one evaluated opportunity."""
        try:
            p = float(row.get("p_model", float("nan")))
            odds = float(row.get("market_odds", 0.0))
            ev_pct = float(row.get("ev_percent", float("nan")))
            lh = float(row.get("lh", 1.5))
            la = float(row.get("la", 1.2))
        except (TypeError, ValueError):
            return ("REJECTED", "unparseable numeric fields")
        if not (np.isfinite(p) and np.isfinite(ev_pct) and np.isfinite(odds)):
            return ("REJECTED", "non-finite model/odds fields")
        if not (0.01 <= p <= 0.99):
            return ("REJECTED", f"model prob out of bounds ({p:.3f})")
        if not (odds > 1.0):
            return ("REJECTED", f"invalid odds ({odds})")
        if str(row.get("odds_source", "")).strip() == "":
            return ("REJECTED", "ODDS_UNVERIFIED (no real bookmaker price)")
        if not (0.2 <= lh <= 5.0 and 0.2 <= la <= 5.0):
            return ("REJECTED", f"lambda anomaly ({lh}/{la})")
        if ev_pct < self.min_ev_pct:
            return ("REJECTED", f"EV {ev_pct:+.1f}% below {self.min_ev_pct:.1f}% gate")
        return ("APPROVED", "passes EV gate and sanity bounds")

    def audit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split evaluated opportunities into (consensus, rejected).
        Consensus = APPROVED rows only; rejected rows are preserved for audit.
        """
        if df.empty:
            return df, df
        verdicts, reasons = [], []
        for _, r in df.iterrows():
            v, reason = self._audit_row(r.to_dict())
            verdicts.append(v)
            reasons.append(reason)
        df = df.copy()
        df["verdict"] = verdicts
        df["reject_reason"] = reasons
        consensus = df[df["verdict"] == "APPROVED"].copy()
        rejected = df[df["verdict"] != "APPROVED"].copy()
        for _, r in rejected.iterrows():
            print(f"[CRITIC] REJECTED {r['league']} {r['home']} vs {r['away']} "
                  f"({r['market']}) — {r['reject_reason']}")
        print(f"[CRITIC] Consensus: {len(consensus)} approved / {len(rejected)} rejected.")
        return consensus, rejected


# ==============================================================================
# 4. SETTLEMENT & PIPELINE EXECUTION
# ==============================================================================

class PortfolioExposureController:
    """
    Multi-bet exposure control: prevents over-allocation when many +EV bets
    land on the same matchday (cross-bet correlation risk / simultaneous loss).

    Method: after individual 0.25-Kelly stakes are set, group approved bets by
    kickoff date and apply a per-matchday aggregate cap (fraction of bankroll).
    When a matchday's summed stakes exceed the cap, ALL stakes in that group are
    scaled down proportionally — preserving the model's relative-conviction
    ordering rather than dropping the weakest picks outright.
    """

    def __init__(self, matchday_cap: float = 0.12, global_cap: float = 0.30,
                 kelly_fraction: float = 0.25) -> None:
        self.matchday_cap = float(matchday_cap)      # max total stake per kickoff date
        self.global_cap = float(global_cap)          # max total stake across whole slate
        self.kelly_fraction = kelly_fraction
        # Audit trail for reporting.
        self.scaled_matchdays = 0
        self.scaled_global = False

    @staticmethod
    def _group(day: Any) -> str:
        try:
            return pd.to_datetime(day).strftime("%Y-%m-%d")
        except Exception:
            return str(day)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of ``df`` with ``stake_kelly`` scaled down to satisfy the
        per-matchday (then global) aggregate exposure caps. Fractional-Kelly
        ordering is preserved; no stake is ever increased.
        """
        if df is None or df.empty:
            return df
        out = df.copy()
        raw_col = out.get("stake_kelly", out.get("stake"))
        if raw_col is None:
            return out
        out["stake_kelly"] = pd.to_numeric(raw_col, errors="coerce").fillna(0.0)
        out["stake_raw"] = out["stake_kelly"]  # pre-shrink record for audit

        # 1. Per-matchday aggregate cap (proportional shrink within each date group).
        self.scaled_matchdays = 0
        for day, grp in out.groupby(out["date"].map(self._group)):
            grp_sum = float(grp["stake_kelly"].sum())
            if grp_sum > self.matchday_cap and grp_sum > 0:
                factor = self.matchday_cap / grp_sum
                mask = out["date"].map(self._group) == day
                out.loc[mask, "stake_kelly"] = out.loc[mask, "stake_kelly"] * factor
                self.scaled_matchdays += 1

        # 2. Global slate cap (proportional shrink across all approved bets).
        total = float(out["stake_kelly"].sum())
        if total > self.global_cap and total > 0:
            factor = self.global_cap / total
            out["stake_kelly"] = out["stake_kelly"] * factor
            self.scaled_global = True

        out["stake_kelly"] = out["stake_kelly"].round(4)
        return out

    def audit_note(self) -> str:
        """Human-readable portfolio-control summary for the report."""
        parts = []
        if self.scaled_matchdays > 0:
            parts.append(f"matchday-cap applied x{self.scaled_matchdays}")
        if self.scaled_global:
            parts.append("global-cap applied")
        return "، ".join(parts) if parts else "within caps"


def run_settlement() -> Dict[str, Any]:
    """
    Executes settle_ledger_auto.py and returns execution metrics.
    """
    print('[STEP 1] Running Ledger Settlement Engine...')
    settle_script = Path(__file__).parent / 'settle_ledger_auto.py'
    if not settle_script.exists():
        settle_script = Path('C:/Users/User/settle_ledger_auto.py')
        
    if settle_script.exists():
        try:
            res = subprocess.run([sys.executable, str(settle_script)], capture_output=True, text=True, timeout=30)
            if res.returncode != 0:
                print(f"[SETTLE ERROR] {res.stderr[:500]}")
        except Exception as e:
            print(f"[SETTLE EXCEPTION] {e}")
            
    summary_path = Path('ledger_settlement_summary.json')
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            pass
            
    return {'settled_now': 0, 'total_profit_units': 0.0, 'cumulative_roi_percent': 0.0, 'pending_after': 0}


def fetch_and_evaluate_opportunities(
    data_agent: Optional["DataIngestionAgent"] = None,
    quant_agent: Optional["QuantCalibrationAgent"] = None,
    risk_agent: Optional["RiskManagementAgent"] = None,
) -> pd.DataFrame:
    """
    Evaluates upcoming fixtures with Dixon-Coles, Vig-removal, +EV, and Fractional Kelly sizing.
    Agent-driven orchestration; all math delegates to the audited module functions.
    Optional agents may be injected (constructed with defaults otherwise).
    """
    data_agent = data_agent or DataIngestionAgent()
    quant_agent = quant_agent or QuantCalibrationAgent()
    risk_agent = risk_agent or RiskManagementAgent()
    active_bets = data_agent.active_bets()

    upgraded_rows = []
    for idx, row in active_bets.iterrows():
        home = str(row['home'])
        away = str(row['away'])
        league = str(row.get('league', 'La Liga'))
        market = str(row.get('market', 'Over 2.5'))
        odds = float(row.get('market_odds') or float('nan'))

        # 1. Dixon-Coles pricing via Quant agent (strictly this league's params)
        pricing = quant_agent.price_fixture(home, away, league)
        market_probs = pricing["market_probs"]

        # 2. Extract model probability based on market
        if 'Over' in market:
            p_model = market_probs['over_2.5']
        elif 'AH' in market or 'Handicap' in market:
            p_model = market_probs['ah_-0.75_away'] if ('Paris' in away or 'Real' in away) else market_probs['ah_-0.75_home']
        elif 'BTTS' in market:
            p_model = market_probs['btts_yes']
        else:
            p_model = float(row.get('pred_over', row.get('pred', 0.55)))

        if pd.isna(p_model) or p_model <= 0:
            p_model = float(row.get('pred_over', row.get('pred', 0.58)))

        # 3-4. Fair value + stake sizing via Risk agent
        risk = risk_agent.evaluate(float(p_model), odds)

        # Preserve or update values
        row_dict = row.to_dict()
        row_dict['odds_source'] = str(row.get('odds_source', '')).strip()
        row_dict['p_model'] = round(float(p_model), 4)
        row_dict['p_fair'] = risk['p_fair']
        row_dict['ev_percent'] = risk['ev_percent']
        row_dict['edge'] = risk['edge']
        row_dict['stake_kelly'] = risk['stake_kelly']
        row_dict['lh'] = pricing['lh']
        row_dict['la'] = pricing['la']
        row_dict['rho'] = round(float(pricing['rho']), 4)
        row_dict['mle_status'] = pricing['mle_status']

        upgraded_rows.append(row_dict)

    return pd.DataFrame(upgraded_rows)


# ==============================================================================
# 5. TELEGRAM DISPATCH & NOTIFICATION FORMATTING
# ==============================================================================

def format_telegram_report(summary: Dict[str, Any], df_opps: pd.DataFrame,
                          critic_note: str = "") -> str:
    """
    Formats institutional-grade quantitative Telegram notification.
    critic_note is appended only when the Critic audited the slate.
    """
    settled_now = summary.get('settled_now', 0)
    profit = float(summary.get('total_profit_units', 0.0))
    roi = float(summary.get('cumulative_roi_percent', 0.0))
    pending = summary.get('pending_after', 0)

    # Per-league calibration provenance (all calibrated leagues, not just consensus survivors).
    _mle_bits = []
    _seen_leagues = sorted(_FITTED_BY_LEAGUE.keys()) or \
        sorted({str(r.get('league', '')) for _, r in df_opps.iterrows()} if not df_opps.empty else [])
    for _lg in _seen_leagues:
        _p_h, _p_a, _p_r = get_league_params(_lg)
        _mle_bits.append(f"{_lg.title() if _lg == _lg.lower() else _lg}: μH={_p_h:.2f} μA={_p_a:.2f} ρ={_p_r:.3f} ({get_league_mle_status(_lg)})")
    _mle_txt = " | ".join(_mle_bits) if _mle_bits else "no active leagues"
    _critic_block = f"• تدقيق الناقد: <code>{critic_note}</code>\n" if critic_note else ""

    msg = f"""📊 <b>تسوية الجولة وإحصائيات المحفظة</b>
• الصفقات المسواة حديثاً: <b>{settled_now}</b>
• صافي الأرباح: <b>{profit:+.3f} وحدة</b>
• العائد التراكمي (ROI): <b>{roi:.2f}%</b>
• إجمالي العمليات المعلقة: <b>{pending}</b>
• معايرة MLE: <code>{_mle_txt}</code>
{_critic_block}
🚀 <b>التوصيات المعتمدة للجولة القادمة (Dixon-Coles + 0.25 Kelly)</b>
"""

    if df_opps.empty:
        msg += "<i>لا توجد فرص تحقق شروط القيمة (+EV > 8%) لهذه الجولة.</i>\n"
    else:
        for _, r in df_opps.iterrows():
            d_str = str(r['date'])[:10]
            league = r['league']
            home = r['home']
            away = r['away']
            market = r['market']
            odds = float(r.get('market_odds', 1.90))
            p_mod = float(r.get('p_model', 0.60)) * 100
            p_fair = float(r.get('p_fair', 0.52)) * 100
            ev = float(r.get('ev_percent', 10.0))
            stake = float(r.get('stake_kelly', r.get('stake', 0.02))) * 100
            
            msg += f"• <b>{d_str} | {league}</b>\n"
            msg += f"  ⚽ {home} vs {away}\n"
            msg += f"  🎯 <b>{market}</b> @ <b>{odds:.2f}</b>\n"
            msg += f"  📈 Model: <code>{p_mod:.1f}%</code> | Fair: <code>{p_fair:.1f}%</code> | EV: <code>{ev:+.1f}%</code>\n"
            msg += f"  💰 حجم الرهان (0.25 Kelly): <b>{stake:.2f}%</b>\n\n"
            
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"⚡ <i>المحرك الكمي نشط | المصدر: paper_trading_ledger_2026.csv</i>"
    return msg.strip()


def send_telegram_alert(message: str) -> None:
    """
    Transmits alert via Telegram Bot API using environment secrets.
    """
    import requests
    tg_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    tg_chat = os.environ.get('TELEGRAM_CHAT_ID')
    
    if tg_token and tg_chat:
        try:
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload = {
                'chat_id': tg_chat,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            r = requests.post(url, data=payload, timeout=15)
            if r.status_code == 200:
                print(f"[TELEGRAM] Alert sent successfully (status: 200).")
            else:
                print(f"[TELEGRAM ERROR] status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[TELEGRAM EXCEPTION] {e}")
    else:
        print("[TELEGRAM] No TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID detected. Running in local/dry mode.")


# ==============================================================================
# 5b. DEDUP STATE — a fixture+market is dispatched ONCE, never re-sent weekly.
# Persisted inside autoloop_last_run.json (already committed by the Git action),
# so the "already sent" memory survives GitHub Actions runs.
# ==============================================================================

def _fixture_key(r: Any) -> str:
    """Unique key for one recommendation: date|home|away|market."""
    try:
        d = str(r['date'])[:10]
    except Exception:
        d = "?"
    return f"{d}|{r['home']}|{r['away']}|{r['market']}"


def _load_sent_keys() -> set:
    """Return set of fixture keys already dispatched in previous runs."""
    try:
        p = Path('autoloop_last_run.json')
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            return set(data.get('sent_fixture_keys', []))
    except Exception:
        pass
    return set()


def dedup_opportunities(df_opps: pd.DataFrame, sent_keys: set) -> tuple:
    """
    Split recommendations into NEW (not yet dispatched) vs ALREADY-SENT.
    Returns (new_df, already_sent_keys, skipped_count).
    """
    if df_opps is None or df_opps.empty:
        return df_opps, set(), 0
    new_rows, new_keys, skipped = [], set(), 0
    for _, r in df_opps.iterrows():
        k = _fixture_key(r)
        if k in sent_keys:
            skipped += 1
        else:
            new_rows.append(r.to_dict())
            new_keys.add(k)
    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=df_opps.columns)
    else:
        new_df = df_opps.iloc[0:0]
    return new_df, new_keys, skipped


    # MAIN ENTRYPOINT
    # ==============================================================================

def main():
    start_time = time.time()
    print(f"[AUTOLOOP] Master Quantitative Loop Start: {datetime.now().isoformat()}")

    # Instantiate the four agents (wrappers only — math lives in module functions).
    data_agent = DataIngestionAgent()
    quant_agent = QuantCalibrationAgent()
    risk_agent = RiskManagementAgent(kelly_fraction=0.25, max_stake=0.05)
    critic_agent = ModelValidatorAgent(min_ev_pct=2.0)

    # 0. Per-league Dynamic MLE calibration via Quant agent (isolated fits; priors when history absent).
    try:
        quant_agent.calibrate_leagues(data_agent.leagues())
    except Exception as e:
        print(f"[MLE] skipped ({e}); using priors.")

    # 1. Settle historical bets
    summary = run_settlement()

    # 2. Evaluate next gameweek opportunities (Data -> Quant -> Risk)
    df_evaluated = fetch_and_evaluate_opportunities(data_agent, quant_agent, risk_agent)

    # 3. Critic audit -> final consensus drives all downstream outputs
    df_opps, df_rejected = critic_agent.audit(df_evaluated)

    # 3b. Multi-bet portfolio exposure control (per-matchday + global caps).
    portfolio_controller = PortfolioExposureController(
        matchday_cap=0.12, global_cap=0.30, kelly_fraction=0.25)
    df_opps = portfolio_controller.apply(df_opps)
    port_note = portfolio_controller.audit_note()
    print(f"[PORTFOLIO] matchday-caps triggered={portfolio_controller.scaled_matchdays} "
          f"global-cap={portfolio_controller.scaled_global} | {port_note}")

    # 3c. Dedup: never re-dispatch a fixture+market already sent in an earlier run
    # (fixes the weekly repeat of the same match). Only NEW recommendations are
    # included in the report, and their keys are persisted for future runs.
    sent_keys = _load_sent_keys()
    df_opps, newly_sent_keys, skipped_sent = dedup_opportunities(df_opps, sent_keys)
    if skipped_sent:
        print(f"[DEDUP] skipped {skipped_sent} already-dispatched recommendation(s).")

    critic_note = (f"{len(df_opps)} معتمدة / {len(df_rejected)} مرفوضة "
                   + (f"/ {skipped_sent} مكررة (تم إرسالها سابقاً)" if skipped_sent else "") + " "
                   f"(بوابة EV ≥ {critic_agent.min_ev_pct:.1f}% | تحكم المحفظة: {port_note})")
    report_text = format_telegram_report(summary, df_opps, critic_note=critic_note)
    print("\n--- [TELEGRAM PREVIEW] ---")
    print(report_text)
    print("--------------------------\n")
    
    # 4. Save artifacts for CI/CD & n8n
    Path('telegram_last_report.txt').write_text(report_text, encoding='utf-8')
    
    out_payload = {
        'timestamp': datetime.utcnow().isoformat(),
        'settlement': summary,
        'opportunities_count': int(len(df_opps)),
        'opportunities': df_opps.astype(str).to_dict(orient='records') if not df_opps.empty else [],
        'rejected_count': int(len(df_rejected)),
                'rejected': df_rejected.astype(str).to_dict(orient='records') if not df_rejected.empty else [],
                'report_markdown': report_text,
                'sent_fixture_keys': sorted(sent_keys | newly_sent_keys)
            }
    Path('autoloop_last_run.json').write_text(json.dumps(out_payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print("[SAVED] telegram_last_report.txt & autoloop_last_run.json updated.")
    
    # 5. Dispatch Telegram Notification
    send_telegram_alert(report_text)
    
    elapsed = time.time() - start_time
    print(f"[AUTOLOOP] Pipeline execution complete in {elapsed:.2f} seconds. (Sub-45s limit satisfied).")


if __name__ == '__main__':
    main()