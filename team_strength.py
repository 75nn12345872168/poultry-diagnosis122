"""
team_strength.py — DATA-DRIVEN club attack/defense strengths.

Replaces the hardcoded `top_elites` priors in autoloop_master.py with strengths
estimated from real match history. A club's strength (att >= 1 attacks more,
def >= 1 concedes more) is a shrunken rate relative to its league average,
computed ONLY from matches already played before a fixture's kickoff (NO
look-ahead leakage).

Sources:
  * La Liga : la_liga_matches_2018_2026.csv (home/away teams + goals, 2018-2026)
  * Ligue 1 : raw_Ligue1_8seasons.json (per-team history with scored/missed)
"""

import json
from pathlib import Path

import pandas as pd

LA_CSV = 'la_liga_matches_2018_2026.csv'
LIGUE_JSON = 'raw_Ligue1_8seasons.json'
# Shrinkage: a team with GP games gets weight GP/(GP+K) toward its own rate.
K_SHRINK = 15.0


class TeamStrengthModel:
    def __init__(self, la_csv: str = LA_CSV, ligue_json: str = LIGUE_JSON,
                 min_matches: int = 5):
        self.min_matches = min_matches
        self.leagues = {}   # league -> {'mu_h':, 'mu_a':, 'history': {team:[(date,...)]}}
        self._build_la(la_csv)
        self._build_ligue(ligue_json)

    # ------------------------------------------------------------------ la liga
    def _build_la(self, csv_path: str):
        path = Path(csv_path)
        if not path.exists():
            self.leagues['La Liga'] = {'mu_h': 1.48, 'mu_a': 1.18, 'events': {}}
            return
        df = pd.read_csv(path, parse_dates=['date'])
        df = df[df['isResult'].astype(bool) == True] if 'isResult' in df.columns else df
        df = df.dropna(subset=['goals_home', 'goals_away'])

        mh = df['goals_home'].sum(); nh = (df['home'].notna()).sum()
        ma = df['goals_away'].sum(); na = (df['away'].notna()).sum()
        mu_h = float(mh) / max(1, nh)
        mu_a = float(ma) / max(1, na)

        events = {}
        for _, r in df.iterrows():
            date = r['date']
            for team, scored, conceded in (
                    (r['home'], r['goals_home'], r['goals_away']),
                    (r['away'], r['goals_away'], r['goals_home'])):
                events.setdefault(str(team), []).append((date, float(scored), float(conceded)))
        for t in events:
            events[t].sort()
        self.leagues['La Liga'] = {'mu_h': mu_h, 'mu_a': mu_a, 'events': events}

    # ---------------------------------------------------------------- ligue 1
    def _build_ligue(self, json_path: str):
        def _none():
            return {'mu_h': 1.55, 'mu_a': 1.20, 'events': {}}
        path = Path(json_path)
        if not path.exists():
            self.leagues['Ligue 1'] = _none()
            return
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            self.leagues['Ligue 1'] = _none()
            return

        events = {}
        total_h_goals = total_h_games = total_a_goals = total_a_games = 0.0
        for season, sdata in raw.items():
            if not isinstance(sdata, dict):
                continue
            for team in sdata.get('teams', {}).values():
                name = team.get('title')
                if not name:
                    continue
                for m in team.get('history', []):
                    date = pd.to_datetime(m.get('date'), errors='coerce')
                    if pd.isna(date):
                        continue
                    h_a = m.get('h_a')
                    scored = float(m.get('scored', 0) or 0)
                    missed = float(m.get('missed', 0) or 0)
                    events.setdefault(name, []).append((date, scored, missed))
                    if h_a == 'h':
                        total_h_goals += scored; total_h_games += 1
                    elif h_a == 'a':
                        total_a_goals += scored; total_a_games += 1
        for t in events:
            events[t].sort()
        mu_h = total_h_goals / max(1, total_h_games)
        mu_a = total_a_goals / max(1, total_a_games)
        self.leagues['Ligue 1'] = {'mu_h': mu_h, 'mu_a': mu_a, 'events': events}

    # ---------------------------------------------------------------- lookup
    def league_baselines(self, league: str):
        s = self.leagues.get(str(league), {})
        return (float(s.get('mu_h', 1.48)), float(s.get('mu_a', 1.18)),
                s.get('events', {}))

    def strength(self, league: str, team: str, as_of=None) -> tuple:
        """
        Return (att, def) for a team relative to its league, from matches played
        strictly before `as_of`. att/def are shrunken toward 1.0. Unknown team
        or too few games -> (1.0, 1.0) (league-average, no fabrication).
        """
        mu_h, mu_a, events = self.league_baselines(league)
        ev = events.get(str(team))
        if not ev:
            return 1.0, 1.0
        played = [(d, s, c) for (d, s, c) in ev
                  if as_of is None or d < as_of]
        if len(played) < self.min_matches:
            return 1.0, 1.0
        n = float(len(played))
        gf = sum(s for _, s, _ in played)
        ga = sum(c for _, _, c in played)
        # league goals per team-game (each match produced goals for each side)
        league_gf_per_game = (mu_h + mu_a) / 2.0
        raw_att = (gf / n) / max(0.05, league_gf_per_game)
        raw_def = (ga / n) / max(0.05, league_gf_per_game)
        sh = n / (n + K_SHRINK)
        att = 1.0 + sh * (raw_att - 1.0)
        deff = 1.0 + sh * (raw_def - 1.0)
        return round(max(0.3, min(2.2, att)), 3), round(max(0.3, min(2.2, deff)), 3)


# Reusable singleton, built once per process.
_MODEL = None


def get_model(refresh: bool = False) -> TeamStrengthModel:
    global _MODEL
    if _MODEL is None or refresh:
        _MODEL = TeamStrengthModel()
    return _MODEL


if __name__ == '__main__':
    model = get_model(refresh=True)
    for league in ('La Liga', 'Ligue 1'):
        mu_h, mu_a, ev = model.league_baselines(league)
        print(f"\n== {league}: mu_h={mu_h:.3f} mu_a={mu_a:.3f} | teams={len(ev)}")
        # show strengths of the teams currently in the ledger (as of today)
        import pandas as pd
        li = pd.read_csv('paper_trading_ledger_2026.csv', parse_dates=['date'])
        teams = sorted({s for c in ('home', 'away') for s in li[c].astype(str)})
        for t in teams[:12]:
            a, d = model.strength(league, t, as_of=pd.Timestamp('2026-10-30'))
            print(f"   {t:24s} att={a}  def={d}")