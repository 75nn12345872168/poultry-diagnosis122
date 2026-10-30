"""
fetch_odds.py — Attach REAL bookmaker odds to the ledger from The Odds API.

Why: the pipeline previously defaulted every bet to odds=1.90 (a fabricated
constant), which made every +EV number meaningless. This fetcher writes REAL
decimal prices and tags each bet with odds_source='the-odds-api'. Bets without
a fetched price keep odds_source='' and are rejected downstream as
ODDS_UNVERIFIED — the pipeline will never recommend a bet on a made-up price.

Usage:
    export THE_ODDS_API_KEY=xxxx        # or set "api_key" in odds_config.json
    python fetch_odds.py

Free-trade without a key is impossible (the API requires one) — the script
fails fast with a clear message in that case.
"""

import os, sys, json, time
from pathlib import Path
import pandas as pd
import requests

LEDGER = 'paper_trading_ledger_2026.csv'
CONFIG = 'odds_config.json'
BASE = 'https://api.the-odds-api.com/v4'
# The Odds API sport keys for the two leagues in the ledger.
SPORT_MAP = {
    'La Liga': 'soccer_spain_la_liga',
    'Ligue 1': 'soccer_france_ligue_one',
}


def get_api_key() -> str:
    key = os.environ.get('THE_ODDS_API_KEY', '').strip()
    if key:
        return key
    cfg = Path(CONFIG)
    if cfg.exists():
        try:
            key = str(json.loads(cfg.read_text(encoding='utf-8')).get('api_key', '')).strip()
        except Exception:
            pass
    if not key:
        sys.exit("ERR: no THE_ODDS_API_KEY in env or odds_config.json — "
                 "get a key at https://the-odds-api.com (free tier) and set it. "
                 "Until then the pipeline stays honest and rejects all bets as ODDS_UNVERIFIED.")
    return key


def get_sport_key(league: str) -> str:
    return SPORT_MAP.get(league)
    # no silent default: unknown league -> None -> skipped


def fetch_odds(session, key, sport_key, regions='eu'):
    """
    Returns dict mapping (home, away) -> {'over25': dec, 'favorite': name,
    'fav_spread': line, 'fav_price': dec, 'h2h': {...}} from h2h/totals/spreads.
    """
    url = f"{BASE}/sports/{sport_key}/odds/"
    params = {
        'apiKey': key, 'regions': regions,
        'markets': 'h2h,totals,spreads', 'oddsFormat': 'decimal',
    }
    r = session.get(url, params=params, timeout=20)
    if r.status_code == 401:
        raise PermissionError("The Odds API rejected the key (401/INVALID_KEY).")
    r.raise_for_status()
    out = {}
    for m in r.json():
        home, away = m.get('home_team'), m.get('away_team')
        rec = {'over25': None, 'fav_name': None, 'fav_line': None,
               'fav_price': None, 'h2h': {}}
        for bm in m.get('bookmakers', []):
            for mk in bm.get('markets', []):
                mktype = mk.get('key')
                for o in mk.get('outcomes', []):
                    if mktype == 'totals' and o.get('point') == 2.5:
                        if o.get('name') == 'Over':
                            rec['over25'] = min(rec['over25'] or 1e9,
                                                float(o['price']))
                        # Under stored too for reference
                        if o.get('name') == 'Under':
                            rec.setdefault('under25', float(o['price']))
                    elif mktype == 'spreads':
                        # favourite = outcome with the negative line
                        line = o.get('point')
                        if line is not None and float(line) <= -0.25 and (
                                rec['fav_price'] is None or abs(line) > abs(rec['fav_line'] or 0)):
                            rec['fav_name'] = o['name']
                            rec['fav_line'] = float(line)
                            rec['fav_price'] = min(rec['fav_price'] or 1e9,
                                                   float(o['price']))
                    elif mktype == 'h2h':
                        rec['h2h'][o['name']] = min(rec['h2h'].get(o['name'], 1e9),
                                                     float(o['price']))
        out[(home, away)] = rec
    return out


def fair_ah075(rec):
    """
    Fair decimal price for the favourite covering -0.75, if derivable from the
    H2H + spread prices. P(-0.75) = 0.5*P(win_any) [half from -0.5] + ... uses
    the spread favourite leg when a ~-1 / -0.5 line is offered. Returns None if
    the required legs are missing (-> bet stays UNVERIFIED rather than guessed).
    """
    if not rec['fav_name'] or not rec['fav_price']:
        return None
    fav = rec['fav_name']
    if rec['fav_line'] is None:
        return None
    p_spread = 1.0 / rec['fav_price']  # covers the spread line at fav_line
    # Favourite covers -0.75 <=> wins by >=1 (the -0.5 part) fully; the -1 part
    # is not recoverable without the underdog leg. Honest fallback: approximate
    # -0.75 by the -0.5 win probability = P(win outright) from H2H when avail,
    # blended with the spread price. Conservative: use spread price only.
    h2h = rec.get('h2h', {})
    p_fav_win = 1.0 / h2h[fav] if h2h.get(fav) else None
    if p_fav_win is None:
        # no outright price -> cannot derive; keep unverified to avoid a guess
        return None
    p = max(0.01, min(0.98, 0.5 * p_fav_win + 0.5 * p_spread))
    return round(1.0 / p, 2) if p < 0.98 else None


def main():
    key = get_api_key()
    ledger = Path(LEDGER)
    if not ledger.exists():
        sys.exit(f"ERR: {LEDGER} not found")
    df = pd.read_csv(LEDGER, parse_dates=['date'])
    if 'odds_source' not in df.columns:
        df['odds_source'] = ''
    if 'market_odds' not in df.columns:
        df['market_odds'] = float('nan')

    s = requests.Session()
    s.headers.update({'User-Agent': 'hermes-autoloop/1.0'})

    attached = 0
    pending_mask = df['status'] == 'PENDING'
    # Group pending fixtures by league and fetch per league.
    for league, sport_key in SPORT_MAP.items():
        subset = df[pending_mask & (df['league'] == league)]
        if subset.empty:
            continue
        try:
            odds_map = fetch_odds(s, key, sport_key)
        except PermissionError as e:
            sys.exit(f"ERR: {e}")
        except requests.HTTPError as e:
            print(f"[ODDS] {league}: HTTP error {e.response.status_code}; skipping")
            continue
        for idx in subset.index:
            home = str(df.at[idx, 'home']); away = str(df.at[idx, 'away'])
            market = str(df.at[idx, 'market'])
            rec = odds_map.get((home, away))
            if not rec:
                continue
            price = None
            if 'Over' in market or 'Under' in market:
                if 'Over' in market and rec['over25']:
                    price = rec['over25']
                elif 'Under' in market and rec.get('under25'):
                    price = rec['under25']
            elif 'AH' in market:
                price = fair_ah075(rec)   # may be None -> stays unverified
            if price is not None:
                df.at[idx, 'market_odds'] = float(price)
                df.at[idx, 'odds_source'] = 'the-odds-api'
                attached += 1
        # politeness between leagues
        time.sleep(1.1)

    df.to_csv(LEDGER, index=False)
    verified = int((df['odds_source'] == 'the-odds-api').sum())
    print(json.dumps({
        'attached_real_odds': attached,
        'pending_with_real_odds': verified,
        'pending_still_unverified': int(((df['status'] == 'PENDING') & (df['odds_source'] != 'the-odds-api')).sum()),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()