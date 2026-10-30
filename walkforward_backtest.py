"""
walkforward_backtest.py — STRICT out-of-sample walk-forward backtest.

Foreach La Liga match 2018-2026, the model's inputs (league scoring baselines
mu_h/mu_a AND both teams' attack/defense) are estimated EXCLUSIVELY from
matches already played BEFORE that match's kickoff. The model then emits
P(Over 2.5); we compare against the real result.

This answers the question "does the model actually predict?" via:
  * calibration  (does the predicted probability match realised frequency?)
  * Brier score and log-loss vs a naive "always league-average" baseline
  * hit-rate at betting thresholds

REAL bookmaker odds are NOT available in-repo (see the earlier audit), so ROI
is reported in two clearly-labeled illustrative forms only: at a flat assumed
price, and at a price synthesized from the model's own fair odds + margin
(the latter is CIRCULAR and proves nothing about real edge — shown for
transparency, not as a tradable result).
"""

import bisect
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from team_strength import K_SHRINK
from autoloop_master import compute_dixon_coles_probs, calculate_market_probabilities

CSV = 'la_liga_matches_2018_2026.csv'
RHO = -0.13            # Dixon-Coles low-score correction (fixed prior)
FLAT_ODDS = 2.00       # illustrative even-money price
MARGIN = 1.05          # illustrative overround for synthetic prices


def load():
    df = pd.read_csv(CSV, parse_dates=['date'])
    df = df[df['isResult'].astype(str).str.lower() == 'true'] if 'isResult' in df.columns else df
    df = df.dropna(subset=['goals_home', 'goals_away'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def build_prior_structures(df):
    """Return (dates_sorted, muH_at, muA_at, team_events)."""
    dates = df['date'].to_numpy()
    gh = df['goals_home'].to_numpy(float)
    ga = df['goals_away'].to_numpy(float)
    # prefix sums for league baselines (before index i = first i matches)
    cum_hg = np.concatenate(([0], np.cumsum(gh)))
    cum_ag = np.concatenate(([0], np.cumsum(ga)))
    # team_events: team -> sorted list of (date, gf_for, ga_for)
    team_events = defaultdict(list)
    for i in range(len(df)):
        d = dates[i]
        team_events[df['home'].iloc[i]].append((d, float(gh[i]), float(ga[i])))
        team_events[df['away'].iloc[i]].append((d, float(ga[i]), float(gh[i])))
    for t in team_events:
        team_events[t].sort()
    return dates, cum_hg, cum_ag, team_events


def run():
    df = load()
    dates, cum_hg, cum_ag, team_events = build_prior_structures(df)
    # nth days_seconds to bisect dates for league baselines at index i
    n = len(df)
    total_home_games = n
    total_away_games = n

    preds = []
    actuals = []
    sums = {'hit_50': 0, 'bets_50': 0, 'flat_pnl': 0.0, 'syn_pnl': 0.0}
    print(f"walk-forward over {n} La Liga matches | rho={RHO}")

    for i in range(len(df)):
        d = dates[i]
        # league baselines from first i (before this match)
        i_prior_h = float(cum_hg[i]) / i if i > 0 else 1.48
        i_prior_a = float(cum_ag[i]) / i if i > 0 else 1.18
        mu_h, mu_a = i_prior_h, i_prior_a
        league_pg = (mu_h + mu_a) / 2.0

        h = df['home'].iloc[i]; a = df['away'].iloc[i]
        gh = float(df['goals_home'].iloc[i]); ga = float(df['goals_away'].iloc[i])
        lo_h = bisect.bisect_left(team_events[h], (d, -1, -1))
        lo_a = bisect.bisect_left(team_events[a], (d, -1, -1))
        n_h = float(lo_h); n_a = float(lo_a)
        att_h = def_h = att_a = def_a = 1.0
        if n_h >= 1:
            gf_h = sum(e[1] for e in team_events[h][:lo_h])
            ga_h = sum(e[2] for e in team_events[h][:lo_h])
            shh = n_h / (n_h + K_SHRINK)
            att_h = 1 + shh * ((gf_h / n_h) / league_pg - 1)
            def_h = 1 + shh * ((ga_h / n_h) / league_pg - 1)
        if n_a >= 1:
            gf_a = sum(e[1] for e in team_events[a][:lo_a])
            ga_a = sum(e[2] for e in team_events[a][:lo_a])
            sha = n_a / (n_a + K_SHRINK)
            att_a = 1 + sha * ((gf_a / n_a) / league_pg - 1)
            def_a = 1 + sha * ((ga_a / n_a) / league_pg - 1)

        lh = min(5.0, max(0.3, mu_h * att_h * def_a))
        la = min(5.0, max(0.3, mu_a * att_a * def_h))
        mat = compute_dixon_coles_probs(lh, la, max_goals=8, rho=RHO)
        p_over = calculate_market_probabilities(mat)['over_2.5']

        actual = 1 if (gh + ga) > 2.5 else 0
        preds.append(p_over); actuals.append(actual)

        # hit-rate / ROI bookkeeping at illustrative thresholds
        if p_over >= 0.5:
            sums['bets_50'] += 1
            sums['hit_50'] += actual
            sums['flat_pnl'] += (FLAT_ODDS - 1.0) if actual else -1.0
            # synthetic market price = model fair +/- margin (CIRCULAR, illustrative)
            syn_odds = 1.0 / p_over * MARGIN
            sums['syn_pnl'] += (syn_odds - 1.0) if actual else -1.0

    preds = np.array(preds, dtype=float)
    actuals = np.array(actuals, dtype=float)

    # ---------- calibration ----------
    edges = np.linspace(0.30, 0.75, 10)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (preds >= lo) & (preds < hi)
        if m.sum() >= 20:
            rows.append((lo, hi, float(preds[m].mean()), float(actuals[m].mean()), int(m.sum())))
    max_dev = max((abs(pr - ac) for _, _, pr, ac, _ in rows), default=0)
    corr = float(np.corrcoef(preds, actuals)[0, 1])

    # ---------- proper scoring ----------
    eps = 1e-9
    brier = float(np.mean((preds - actuals) ** 2))
    base_rate = float(actuals.mean())
    brier_base = float(np.mean((base_rate - actuals) ** 2))
    brier_skill = 1 - brier / brier_base if brier_base > 0 else float('nan')
    logloss = float(-np.mean(actuals * np.log(np.clip(preds, eps, 1)) +
                             (1 - actuals) * np.log(np.clip(1 - preds, eps, 1))))
    ll_base = float(-(base_rate * math.log(base_rate + eps) +
                      (1 - base_rate) * math.log(1 - base_rate + eps)))
    ll_skill = 1 - logloss / ll_base if ll_base > 0 else float('nan')

    hits50 = sums['hit_50']; bets50 = sums['bets_50']
    result = {
        'matches': int(n),
        'base_over_rate_percent': round(base_rate * 100, 2),
        'mean_model_over_percent': round(float(preds.mean()) * 100, 2),
        'brier': round(brier, 4),
        'brier_baseline': round(brier_base, 4),
        'brier_skill_score': round(brier_skill, 4),
        'logloss': round(logloss, 4),
        'logloss_baseline': round(ll_base, 4),
        'logloss_skill': round(ll_skill, 4),
        'calibration_max_abs_dev': round(max_dev, 4),
        'pred_vs_actual_corr': round(corr, 4),
        'threshold_0.50': {
            'bets': int(bets50),
            'hit_rate_percent': round((hits50 / bets50 * 100) if bets50 else 0.0, 2),
        },
        'illustrative_roi_flat_2.0_percent': round((sums['flat_pnl'] / bets50 * 100) if bets50 else 0.0, 2),
        'illustrative_roi_synthetic_margin_percent': round((sums['syn_pnl'] / bets50 * 100) if bets50 else 0.0, 2),
        'NOTE_real_odds_not_available': 'ROI values use assumed/synthetic prices and are ILLUSTRATIVE, not tradable.',
    }

    print("\n=== CALIBRATION BINS (predicted vs realised) ===")
    print(f"{'pred lo':>7}{'pred hi':>8}{'mean pred':>11}{'realised':>10}{'n':>6}")
    for lo, hi, pr, ac, cnt in rows:
        print(f"{lo:>7.2f}{hi:>8.2f}{pr*100:>10.1f}%{ac*100:>11.1f}%{cnt:>7}")
    print("\n=== SUMMARY ===")
    print("matches:", result['matches'],
          "| base Over rate:", result['base_over_rate_percent'], "%",
          "| mean model p:", result['mean_model_over_percent'], "%")
    print("Brier:", result['brier'], "(baseline", result['brier_baseline'],
          ")  skill:", result['brier_skill_score'])
    print("LogLoss:", result['logloss'], "(baseline", result['logloss_baseline'],
          ")  skill:", result['logloss_skill'])
    print("Calibration max |dev|:", result['calibration_max_abs_dev'],
          "| corr(pred,actual):", result['pred_vs_actual_corr'])
    print("At p>=0.5 -> bets:", bets50, "| hit-rate:", result['threshold_0.50']['hit_rate_percent'], "%")
    print("ILLUSTRATIVE ROI @ flat 2.00:", result['illustrative_roi_flat_2.0_percent'], "%  |",
          "ROI @ synthetic margin:", result['illustrative_roi_synthetic_margin_percent'], "%")
    print("NOTE:", result['NOTE_real_odds_not_available'])

    pd.DataFrame(result, index=[0]).to_csv('walkforward_backtest_result.csv', index=False)
    return result


if __name__ == '__main__':
    run()