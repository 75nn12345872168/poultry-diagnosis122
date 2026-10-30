import pandas as pd, numpy as np, json, sys, os, time
from pathlib import Path
from datetime import datetime, timedelta

LEDGER = 'paper_trading_ledger_2026.csv'
# CUTOFF is now DYNAMIC: any PENDING bet whose kickoff date is in the past is
# eligible for settlement. The old hardcoded 2026-09-07 froze settlement and is
# why nothing ever settled. (Cutoff = now.)
CUTOFF = pd.Timestamp.now().normalize()

# Robust set of clubs treated as the Asian-Handicap favourite (-0.75). Only when
# EXACTLY ONE side is a strong club do we know which side the AH bet is on;
# ambiguous fixtures are left UNKNOWN rather than guessed.
BIG_CLUBS = {
    'Paris Saint Germain', 'Marseille', 'Monaco', 'Lille', 'Lyon', 'Rennes',
    'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Athletic Club', 'Sevilla',
    'Villarreal', 'Real Sociedad', 'Valencia', 'Real Betis',
}

# Result sources: live Understat (cached fresh copy) -> local snapshots.
_RESULT_DATES_KEYS = ('La Liga', 'Ligue 1')
_LIVE_FILES = {
    'La Liga': '_understat_La_Liga_2026_live.json',
    'Ligue 1': '_understat_Ligue_1_2026_live.json',
}
_LOCAL_FILES = {  # fallback snapshots already in repo
    'La Liga': 'understat_raw_2026.json',
    'Ligue 1': 'raw_Ligue1_8seasons.json',
}


def _fetch_live(league: str):
    """Fetch current-season match data from Understat and cache it locally."""
    comp = {'La Liga': 'La_Liga', 'Ligue 1': 'Ligue_1'}[league]
    url = f'https://understat.com/getLeagueData/{comp}/2026'
    import requests
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0',
                      'X-Requested-With': 'XMLHttpRequest',
                      'Referer': 'https://understat.com/'})
    r = s.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    Path(_LIVE_FILES[league]).write_text(
        json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return data.get('dates', [])


def _load_dates(league: str):
    """Return list of {h,a,datetime,isResult,goals} dicts for a league:
    fresh live cache -> live fetch -> local repo snapshot."""
    # 1) Live cache, refreshed if older than 6h.
    path = Path(_LIVE_FILES[league])
    if path.exists() and (time.time() - path.stat().st_mtime) < 6 * 3600:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data.get('dates', [])
    # 2) Try a live fetch so CI can settle without committing result caches.
    try:
        return _fetch_live(league)
    except Exception:
        pass
    # 3) Local repo snapshot.
    for cand in _LOCAL_FILES[league]:
        p = Path(cand)
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding='utf-8'))
                if isinstance(raw, dict) and 'dates' in raw:
                    return raw['dates']
                out = []
                for s, v in raw.items():
                    if isinstance(v, dict):
                        out.extend(v.get('dates', []))
                return out
            except Exception:
                continue
    return []


def fetch_result(league, home, away, date):
    """Real score (gh, ga) for a fixture, or (None, None) if not available."""
    dk = str(pd.to_datetime(date))[:10]
    for d in _load_dates(league):
        if (d.get('h', {}).get('title') == home
                and d.get('a', {}).get('title') == away
                and str(d.get('datetime', ''))[:10] == dk):
            if d.get('isResult') and d.get('goals'):
                return int(d['goals']['h']), int(d['goals']['a'])
    return None, None


def _ah_side(home, away):
    """(-1 = hcov, 0 = ambiguous/unknown, +1 = acov). Which club is the -0.75 fav."""
    hb = home in BIG_CLUBS
    ab = away in BIG_CLUBS
    if hb and not ab:
        return -1
    if ab and not hb:
        return +1
    return 0


def settle_row(row):
    """Resolve a PENDING row against a REAL result. Returns updated row dict."""
    league = str(row.get('league', 'La Liga'))
    market = str(row.get('market', 'Over 2.5'))
    home, away, date = row['home'], row['away'], row['date']

    if pd.to_datetime(date) > CUTOFF:
        return row  # still upcoming -> pending

    gh, ga = fetch_result(league, home, away, date)
    if gh is None:
        # No real result known (postponed / name mismatch / not played).
        row['status'] = 'PENDING'   # untouched
        return row

    stake = float(row.get('stake', row.get('stake_kelly', 0.0)) or 0.0)
    odds = float(row.get('market_odds', 0.0) or 0.0)
    draw_odds = pd.to_numeric(row.get('draw_odds', np.nan), errors='coerce')

    if 'Over' in market or 'Under' in market:
        total = gh + ga
        if market.startswith('Over'):
            won = total > 2.5
        else:
            won = total <= 2.5
        status = 'WIN' if won else 'LOSS'
        profit = stake * (odds - 1) if won else -stake
        row['result'] = f'{gh}-{ga}'
    elif 'AH' in market:
        side = _ah_side(home, away)
        if side == 0:
            row['status'] = 'UNKNOWN_AH'   # which side the fav is, ambiguous
            row['result'] = f'{gh}-{ga}'
            row['profit'] = 0.0
            return row
        diff = (gh - ga) if side == -1 else (ga - gh)  # favorite margin
        if diff >= 2:
            status, profit = 'WIN', stake * (odds - 1)
        elif diff == 1:
            status, profit = 'HALF WIN', stake * 0.5 * (odds - 1)
        else:
            status, profit = 'LOSS', -stake
        row['result'] = f'{gh}-{ga}'
    elif 'BTTS' in market:
        both = (gh > 0) and (ga > 0)
        status = ('WIN' if both else 'LOSS') if (market.endswith('Yes') or 'Yes' in market) \
            else ('WIN' if not both else 'LOSS')
        profit = stake * (odds - 1) if status == 'WIN' else -stake
        row['result'] = f'{gh}-{ga}'
    else:
        row['status'] = 'UNKNOWN_MARKET'
        row['result'] = f'{gh}-{ga}'
        row['profit'] = 0.0
        return row

    row['status'] = status
    row['profit'] = round(float(profit), 4)
    return row


def main():
    ledger = Path(LEDGER)
    if not ledger.exists():
        print(json.dumps({'error': 'ledger not found'}))
        sys.exit(1)

    df = pd.read_csv(LEDGER, parse_dates=['date'])
    for col, default in [('league', 'La Liga'), ('market', 'Over 2.5'),
                         ('stake', 0.0), ('market_odds', 0.0), ('profit', 0.0),
                         ('result', '')]:
        if col not in df.columns:
            df[col] = default
    # 'result' is a text column; an all-empty CSV column is read as float64 and
    # rejects strings like '4-1' -> force object dtype.
    df['result'] = df['result'].astype(object)

    pending_mask = df['status'] == 'PENDING'
    pending = df[pending_mask].copy()
    played = pending[pd.to_datetime(pending['date']).dt.date <= CUTOFF.date()]

    settled_count = 0
    not_found = 0
    ambiguous = 0
    for idx, row in played.iterrows():
        new = settle_row(row.copy())
        df.loc[idx, 'status'] = new['status']
        if 'result' in new:
            df.loc[idx, 'result'] = new['result']
        if 'profit' in new:
            df.loc[idx, 'profit'] = new['profit']
        if new['status'] != 'PENDING':
            settled_count += 1
            if new['status'] in ('UNKNOWN_AH', 'UNKNOWN_MARKET'):
                ambiguous += 1
        else:
            not_found += 1

    # Real, result-driven numbers (hit-rate is REAL; ROI is in ledger units and
    # still depends on whatever price is stored — flag it).
    done = df[df['status'].isin(['WIN', 'LOSS', 'HALF WIN'])]
    wins = df[df['status'] == 'WIN']
    half = df[df['status'] == 'HALF WIN']
    losses = df[df['status'] == 'LOSS']
    total_profit = float(done['profit'].sum()) if not done.empty else 0.0
    total_staked = float(done['stake'].sum()) if not done.empty else 0.0

    # Hit-rate with half-win counted as 0.5.
    settled_units = len(wins) + len(losses) + len(half)
    half_credit = {s: 0.5 for s in half.index}
    weighted_wins = len(wins) + 0.5 * len(half)
    hit_rate = (weighted_wins / settled_units) if settled_units else 0.0
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0

    df.to_csv(LEDGER, index=False)
    summary = {
        'cutoff': str(CUTOFF),
        'pending_before': int(len(pending)),
        'eligible_played': int(len(played)),
        'settled_now': int(settled_count),
        'result_not_found': int(not_found),
        'ambiguous_average': int(ambiguous),
        'pending_after': int(len(df[df['status'] == 'PENDING'])),
        'total_settled_all_time': int(len(done)),
        'wins': int(len(wins)),
        'half_wins': int(len(half)),
        'losses': int(len(losses)),
        'settled_units': int(settled_units),
        'hit_rate_percent': round(float(hit_rate) * 100, 2),
        'total_profit_units': round(total_profit, 4),
        'total_staked': round(total_staked, 4),
        'cumulative_roi_percent': round(float(roi), 2),
        'leagues': df['league'].unique().tolist(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open('ledger_settlement_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == '__main__':
    main()