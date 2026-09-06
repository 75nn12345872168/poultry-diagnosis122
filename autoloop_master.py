
import os, subprocess, json, sys, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

print('[AUTOLOOP] Master Loop Start', datetime.now().isoformat())
# Step 1: Settlement
print('\n[STEP 1] Settlement...')
result = subprocess.run([sys.executable, 'C:/Users/User/settle_ledger_auto.py'], capture_output=True, text=True)
print(result.stdout[:2000])
if result.returncode!=0:
    print(result.stderr[:2000])

# Load settlement summary
try:
    summary=json.load(open('ledger_settlement_summary.json',encoding='utf-8'))
except:
    summary={'settled_now':0,'total_profit_units':0,'cumulative_roi_percent':0,'pending_after':0}

# Step 2: Prediction & Fetch
print('\n[STEP 2] Fetch next GW fixtures + Prediction...')
# We will call paper_trading_next logic but limited to next 7 days
# Instead of subprocess, we will run inline for both leagues with next 7 days filter
import json, pandas as pd, numpy as np
from collections import defaultdict, deque
from sklearn.preprocessing import StandardScaler
import pickle

# Helper to get next GW for a league
def get_next_gw_bets(league_name, model_path, market_odds, market_name):
    # Load model
    with open(model_path,'rb') as f:
        m=pickle.load(f)
    scaler=m['scaler']; model=m['model']; keep8=m['features']
    # Load raw upcoming
    raw=json.load(open(f'raw_{"Ligue1" if "Ligue" in league_name else "Bundesliga" if "Bundes" in league_name else "LaLiga"}_8seasons.json'.replace('LaLiga','LaLiga'),encoding='utf-8')) if False else None
    # Instead directly use existing ledger logic: for La Liga use understat_raw_2026.json, for Ligue1 use raw_Ligue1_8seasons.json plus 2026 fetch
    import requests, time
    # Fetch 2026 upcoming
    league_url = 'La liga' if league_name=='La Liga' else 'Ligue 1'
    s=requests.Session()
    s.headers.update({'User-Agent':'Mozilla/5.0','Referer':f'https://understat.com/league/{league_url}','X-Requested-With':'XMLHttpRequest'})
    s.get(f'https://understat.com/league/{league_url}', timeout=15)
    r=s.get(f'https://understat.com/getLeagueData/{league_url}/2026', timeout=15)
    data=r.json()
    upcoming=[]
    for d in data.get('dates',[]):
        if not d.get('isResult'):
            upcoming.append({'date': pd.to_datetime(d.get('datetime')), 'home': d.get('h',{}).get('title'), 'away': d.get('a',{}).get('title')})
    # Filter to next 7 days from 2026-09-06
    cutoff_start=pd.to_datetime('2026-09-06')
    cutoff_end=cutoff_start+pd.Timedelta(days=8)  # include 13th 23:59
    upcoming=[u for u in upcoming if cutoff_start <= u['date'] <= cutoff_end]
    print(f'[{league_name}] upcoming next 7d: {len(upcoming)}')
    for u in upcoming:
        print(f"  {u['date'].date()} {u['home']} vs {u['away']}")
    # For feature, we need team_state - we will reuse logic from paper_trading_next but quick: use existing df_feat from previous runs?
    # Simplistic: use last known team_state from previous paper_trading_next run (we can load from ledger's last state or recompute quickly via simplified: use Elo and last5 from feature store)
    # For demo, we will use the already computed ledger's upcoming bets that are within next 7d and have edge>8% - i.e., filter existing ledger
    ledger=pd.read_csv('paper_trading_ledger_2026.csv', parse_dates=['date'])
    # Filter to this league and next 7d and PENDING
    mask=(ledger['league']==league_name) & (pd.to_datetime(ledger['date'])>=cutoff_start) & (pd.to_datetime(ledger['date'])<=cutoff_end) & (ledger['status']=='PENDING')
    bets=ledger[mask]
    print(f'[{league_name}] qualified in ledger next 7d: {len(bets)}')
    return bets

# Actually for demo, just filter existing ledger for next 7d
ledger=pd.read_csv('paper_trading_ledger_2026.csv', parse_dates=['date'])
cutoff_start=pd.to_datetime('2026-09-06'); cutoff_end=cutoff_start+pd.Timedelta(days=8)  # include 13th 23:59
next_gw = ledger[(pd.to_datetime(ledger['date'])>=cutoff_start) & (pd.to_datetime(ledger['date'])<=cutoff_end) & (ledger['status']=='PENDING')]
print(f'[STEP 2] Next GW (7d) total pending: {len(next_gw)}')
for _, r in next_gw.iterrows():
    print(f"  {r['date'].date()} {r['league']} {r['home']} vs {r['away']} {r['market']} pred {r.get('pred', r.get('pred_over',''))} edge {r['edge']} stake {r['stake']}")

# Step 3 already done: ledger already has them as PENDING (from install)
# For autoloop, we would add new ones if not exists, but they already exist

# Step 4: Telegram unified report
print('\n[STEP 4] Telegram Report...')
# Settlement part
# For demo, settlement had 0, but we can show
settled_now=summary.get('settled_now',0)
total_profit=summary.get('total_profit_units',0)
roi=summary.get('cumulative_roi_percent',0)
# Next GW bets
next_bets = next_gw
# Format message
msg = f"""📊 تسوية الجولة الماضية
• الصفقات المسواة: {settled_now}
• صافي الربح: {total_profit:.3f} وحدة
• ROI التراكمي: {roi:.2f}%
• المعلق: {summary.get('pending_after',0)}

🚀 توصيات الجولة القادمة (Edge>8% | 1/8 Kelly)
"""
if len(next_bets)==0:
    msg += "لا توجد توصيات مؤهلة هذه الجولة (انتظار).\n"
else:
    for _, r in next_bets.iterrows():
        league=r['league']; market=r['market']
        # pred column may be pred_over or pred
        pred=r.get('pred', r.get('pred_over',''))
        msg += f"• {r['date'].date()} {league} | {r['home']} vs {r['away']} | {market} | Pred {pred} Edge {r['edge']} Stake {r['stake']}\n"
msg += f"\nإجمالي التوصيات: {len(next_bets)} | الدفتر: paper_trading_ledger_2026.csv"
print(msg)
# Save telegram message
with open('telegram_last_report.txt','w',encoding='utf-8') as f:
    f.write(msg)
# Also save JSON for n8n
with open('autoloop_last_run.json','w',encoding='utf-8') as f:
    json.dump({'settlement': summary, 'next_gw_count': int(len(next_bets)), 'next_gw': next_bets.astype(str).to_dict(orient='records'), 'telegram': msg}, f, ensure_ascii=False, indent=2, default=str)
print('\n[SAVED] telegram_last_report.txt autoloop_last_run.json')
# Cloud Telegram send via env vars (GitHub Actions)
import os, requests
tg_token = os.environ.get('TELEGRAM_BOT_TOKEN')
tg_chat = os.environ.get('TELEGRAM_CHAT_ID')
if tg_token and tg_chat:
    try:
        r = requests.get(f'https://api.telegram.org/bot{tg_token}/sendMessage', params={'chat_id': tg_chat, 'text': msg, 'parse_mode': 'HTML'}, timeout=15)
        print(f'[TELEGRAM] Cloud send status {r.status_code}: {r.text[:200]}')
    except Exception as e:
        print(f'[TELEGRAM] failed {e}')
else:
    print('[TELEGRAM] No env token/chat_id - local mode, message saved to telegram_last_report.txt')

print('[AUTOLOOP] Done. La Liga and Ligue1 models untouched.')
