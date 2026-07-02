# Screener — Multi-Source (Upstox • Kite • Yahoo) + Fixed Sector History

## What changed

| File | Status |
|---|---|
| `app.py` | **Rewritten** — Indian-only, multi-source, session-persistent results, auto-saving snapshots, earnings layer removed |
| `data_sources.py` | **New** — Upstox / Kite / Yahoo fetchers with NSE→BSE fallback |
| `sector_history.py` | One-line change — GitHub history cached 5 min |
| `requirements.txt` | Added `pyotp` (for Kite TOTP auto-login) |
| `Stock_List.csv` | Your new 1,922-stock universe — used for symbols AND sector map |
| `vpci_engine.py` | **UNTOUCHED** — zero screening-logic change |
| `earnings_phase.py`, `earnings_overrides_starter.csv` | **Delete from repo** — no longer imported |

Ranker weights revert to your original pre-v4 set (VPCI 25 / RS 25 / 52wH 20 /
Tight 15 / Vol 10 / Mcap 5) since the phase factor is gone. All 7 gates,
freshness detection, G4-pending logic: byte-identical.

## The data chain (why nothing gets skipped now)

For EVERY symbol: `Source 1 NSE → Source 1 BSE → Source 2 NSE → Source 2 BSE → … → Yahoo .NS → .BO`

Tested against your Stock_List.csv: **1,751 resolve on Upstox NSE, 150 more on
Upstox BSE (98.9% total)**; the remaining ~21 (delisted/renamed) fall through to
Kite and Yahoo. The "Data coverage" badges after each scan show exactly which
source served how many stocks, and unresolved symbols get their own
downloadable list.

## Kite daily login

Add to Streamlit Cloud → Settings → Secrets:

```toml
[kite]
api_key     = "your_api_key"
api_secret  = "your_api_secret"
user_id     = "AB1234"          # for TOTP auto-login
password    = "your_password"   # for TOTP auto-login
totp_secret = "BASE32SECRET"    # for TOTP auto-login
```

Then each morning: sidebar → **🔑 Kite Connect — Daily Login** → **Auto-login
with TOTP**. One click generates the day's access token (password → TOTP →
request_token → session/token exchange). If you prefer, paste a manually
generated access_token instead — both are cached until midnight. Kite calls are
internally throttled to ~2.5 req/s (their limit is 3/s), and daily candles are
resampled to Friday-ending weekly bars matching Yahoo's convention.

⚠️ After your recent credential exposure incident: these secrets live only in
Streamlit's secret store — never commit them, and rotate the TOTP secret too if
it was in the exposed file.

## Why sector snapshots never saved — and the fix

The old **Save snapshot** button sat inside `if run_scan:`. Clicking any button
reruns the script with `run_scan = False`, so the save code never executed.
Now:

1. Scan results persist in `st.session_state` — tabs survive every rerun.
2. A snapshot **auto-saves after every scan** (GitHub if `[github]` secrets are
   set, session otherwise). The button remains as a manual re-save.
3. Snapshot market tag is now `IN`; the Rotation tab also reads your legacy
   `NSE` snapshots, so old history still shows.

GitHub secrets (unchanged from before):

```toml
[github]
token  = "github_pat_xxxx"
repo   = "hbtipawan/Screener-Back-UP"
branch = "main"
folder = "history"
```

## Deploy

1. Replace `app.py`, add `data_sources.py`, replace `sector_history.py` and
   `requirements.txt`, add `Stock_List.csv` to the repo root.
2. Delete `earnings_phase.py`, `earnings_overrides_starter.csv`, `us_stocks.csv`,
   `us_etfs.csv`, `bse_stocks.csv` (no longer read).
3. Add/verify the `[kite]` and `[github]` secrets.
4. Reboot the app. Run a scan → check the coverage badges → open Sector
   Leadership and confirm "Snapshot status: ✅ …pushed to …".

---

## Update 2 — Crash fix, True Fresh Signals, New Listings tab

### The ModuleNotFoundError on Streamlit Cloud
The traceback dies at `from vpci_engine import …`, which means one of two
things in your `rojiroti` repo: **(a)** `vpci_engine.py` isn't in the repo
root, or **(b)** it is, but `yfinance` is missing from `requirements.txt`
(vpci_engine imports yfinance at the top, so the error surfaces on that line).
Checklist — repo root must contain: `app.py`, `vpci_engine.py`,
`data_sources.py`, `sector_history.py`, `signal_history.py`,
`Stock_List.csv`, and `requirements.txt` with all five packages
(streamlit, pandas, requests, yfinance, pyotp). The app now has a **startup
guard** that names the exact missing file/package on screen instead of the
redacted crash. Reboot the app after pushing.

### True Fresh Signals (fresh_v2)
Old behaviour: "fresh" = 7/7 now but not on the previous bar — so a stock
already in an open position that dipped to 6/7 for a week and re-passed
showed as fresh. New behaviour: the full weekly history is replayed as a
position state machine — ENTRY on 7/7 (same gates + extension filter, using
the engine's own `_check_gates_at`), trailing stop ratchets at close − 3×ATR,
SELL when close breaks it. **FRESH = 7/7 fires with no open prior position**
(first-ever signal, or first after a sell). Re-triggers inside a live
position are labelled 📌 CONTINUING (7/7) — still in the Buyable tab, with an
audit expander in the Fresh tab. `position_state`, `last_entry_date`,
`last_exit_date` columns show the reasoning for every stock.

### 🌱 New Listings tab
Stocks with 22–41 weekly bars (too young for the 40-week SMA gate) used to
fall into "failed". They're now kept and given the engine's exact VPCI 5/20
computation. Those meeting the G5 condition (VPCI > 0, above signal, rising)
are flagged **Accumulating** — a watch list of promising young listings to
track until they mature into the main scan. Full 7-gate logic is completely
unchanged; this is a separate additive view.
