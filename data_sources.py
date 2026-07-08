"""
data_sources.py
───────────────
Multi-source weekly OHLCV data layer for the VPCI screener.

Sources (in user-selected priority order):
  1. UPSTOX  — public v2 historical-candle API, NO API key needed.
               Instrument master downloaded once/day from Upstox assets CDN.
  2. KITE    — Zerodha Kite Connect (paid API). Daily TOTP auto-login
               generates the day's access_token automatically, or the user
               can paste a manually-generated access_token.
  3. YAHOO   — yfinance fallback (.NS then .BO).

For EVERY symbol the chain is:
    source-1 NSE → source-1 BSE → source-2 NSE → source-2 BSE → ... → Yahoo .NS → .BO

so a stock listed only on BSE (or missing from one vendor's master) is still
found. Returns the SAME weekly DataFrame shape that vpci_engine expects —
no screening logic is touched here, this is purely data plumbing.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import threading
import time
import urllib.parse
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

# Same bar minimum as vpci_engine.MIN_BARS — kept in sync, not imported, so
# this module also works standalone in scripts/tests.
MIN_BARS = 42

# ═══════════════════════════════════════════════════════════════════════
# LIVE / INTRAWEEK SUPPORT
# ═══════════════════════════════════════════════════════════════════════
IST          = timezone(timedelta(hours=5, minutes=30))
SESSION_MINS = 375          # 09:15 → 15:30
# Volume projection guards. Below MIN_PROJ_SESSIONS the sample is too small to
# extrapolate (Mon 09:20 => elapsed 0.02 => a naive 5/elapsed is a 250x scale-up).
MIN_PROJ_SESSIONS = 0.50
MAX_PROJ_MULT     = 5.00
_OPEN        = (9, 15)
_CLOSE       = (15, 30)


def ist_now() -> datetime:
    return datetime.now(IST)


def session_fraction(now: datetime | None = None) -> float:
    """How much of TODAY'S NSE session has elapsed, 0.0 → 1.0.
    Weekend/holiday-agnostic: caller decides whether today is a trading day."""
    now = now or ist_now()
    o = now.replace(hour=_OPEN[0],  minute=_OPEN[1],  second=0, microsecond=0)
    c = now.replace(hour=_CLOSE[0], minute=_CLOSE[1], second=0, microsecond=0)
    if now <= o:
        return 0.0
    if now >= c:
        return 1.0
    return (now - o).total_seconds() / (SESSION_MINS * 60.0)


def market_is_open(now: datetime | None = None) -> bool:
    now = now or ist_now()
    if now.weekday() >= 5:
        return False
    return 0.0 < session_fraction(now) < 1.0


def week_elapsed_sessions(daily: pd.DataFrame) -> float:
    """Sessions elapsed in the CURRENT (last) W-FRI bucket of a daily frame.
    Completed days count 1.0; today counts session_fraction()."""
    if daily is None or daily.empty:
        return 5.0
    idx = pd.DatetimeIndex(daily.index)
    week_end = idx[-1] + pd.offsets.Week(weekday=4) if idx[-1].weekday() != 4 else idx[-1]
    wk_start = week_end - pd.Timedelta(days=4)
    in_week = idx[idx >= wk_start.normalize()]
    if len(in_week) == 0:
        return 5.0
    n = float(len(in_week))
    today = pd.Timestamp(ist_now().date())
    if in_week[-1].normalize() == today and market_is_open():
        n = n - 1.0 + max(session_fraction(), 0.02)
    return max(n, 0.02)

# ═══════════════════════════════════════════════════════════════════════
# RATE LIMITERS (thread-safe) — Kite allows 3 req/s on historical API,
# Upstox is generous but we still keep it polite.
# ═══════════════════════════════════════════════════════════════════════
class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = self._last + self.min_interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


_KITE_LIMITER   = _RateLimiter(0.40)   # ~2.5 req/s, safely under Kite's 3/s
_UPSTOX_LIMITER = _RateLimiter(0.06)


# ═══════════════════════════════════════════════════════════════════════
# SHARED WEEKLY-DF CLEANUP — mirrors vpci_engine.fetch_weekly_data's
# post-processing exactly (drop incomplete week, positive rows, MIN_BARS).
# ═══════════════════════════════════════════════════════════════════════
def _clean_weekly(df: pd.DataFrame | None, min_bars: int = MIN_BARS,
                  keep_partial: bool = False) -> pd.DataFrame | None:
    """keep_partial=True  → KEEP the running week's forming candle (live mode).
       keep_partial=False → legacy behaviour, drop it (Friday-close mode)."""
    if df is None or df.empty or len(df) < min_bars:
        return None
    df = df.copy()
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    for col in ["Close", "High", "Low", "Volume", "Open"]:
        if col not in df.columns:
            return None
    # Drop incomplete current week (same heuristic as vpci_engine)
    if not keep_partial and len(df) > 2:
        last_vol = df["Volume"].iloc[-1]
        prev_vol = df["Volume"].iloc[-2]
        if prev_vol > 0 and last_vol < (prev_vol * 0.25):
            df = df.iloc[:-1]
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)].dropna(subset=["Close", "Volume"])
    return df if len(df) >= min_bars else None


def _finalise_live(weekly: pd.DataFrame, daily: pd.DataFrame,
                   project_volume: bool) -> pd.DataFrame:
    """Tag the frame as partial and (optionally) pro-rate the running week's
    volume up to a full-week equivalent so VPCI's volume terms aren't crushed."""
    weekly = weekly.copy()
    elapsed = week_elapsed_sessions(daily)
    raw_vol = float(weekly["Volume"].iloc[-1])
    mult = 1.0
    do_proj = project_volume and MIN_PROJ_SESSIONS <= elapsed < 5
    if do_proj:
        mult = min(5.0 / elapsed, MAX_PROJ_MULT)
        weekly.iloc[-1, weekly.columns.get_loc("Volume")] = raw_vol * mult
    weekly.attrs["partial_week"]       = elapsed < 5
    weekly.attrs["elapsed_sessions"]   = round(elapsed, 2)
    weekly.attrs["raw_partial_volume"] = raw_vol
    weekly.attrs["volume_projected"]   = bool(do_proj)
    weekly.attrs["projection_mult"]    = round(mult, 2)
    # Confidence in the running-week bar: 1.0 by Friday close, low on Monday am
    weekly.attrs["week_confidence"]    = round(min(elapsed / 5.0, 1.0), 2)
    weekly.attrs["asof"]               = ist_now().strftime("%Y-%m-%d %H:%M IST")
    return weekly


def _daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars (Friday-ending, matching Yahoo)."""
    daily = daily.sort_index()
    wk = pd.DataFrame({
        "Open":   daily["Open"].resample("W-FRI").first(),
        "High":   daily["High"].resample("W-FRI").max(),
        "Low":    daily["Low"].resample("W-FRI").min(),
        "Close":  daily["Close"].resample("W-FRI").last(),
        "Volume": daily["Volume"].resample("W-FRI").sum(),
    }).dropna(subset=["Close"])
    return wk


# ═══════════════════════════════════════════════════════════════════════
# 1. UPSTOX  (no API key needed for historical candles)
# ═══════════════════════════════════════════════════════════════════════
_UPSTOX_ASSETS = "https://assets.upstox.com/market-quote/instruments/exchange/{exch}.csv.gz"


@st.cache_data(ttl=86400, show_spinner=False)
def load_upstox_instruments() -> dict:
    """
    Download Upstox instrument masters for NSE + BSE (once per day) and
    return {"NSE": {tradingsymbol: instrument_key}, "BSE": {...}}.
    """
    out = {"NSE": {}, "BSE": {}}
    for exch in ("NSE", "BSE"):
        try:
            r = requests.get(_UPSTOX_ASSETS.format(exch=exch), timeout=60)
            r.raise_for_status()
            raw = gzip.decompress(r.content)
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            df.columns = [str(c).strip().lower() for c in df.columns]
            key_col = "instrument_key" if "instrument_key" in df.columns else None
            sym_col = "tradingsymbol" if "tradingsymbol" in df.columns else (
                      "trading_symbol" if "trading_symbol" in df.columns else None)
            if not key_col or not sym_col:
                continue
            seg = f"{exch}_EQ|"
            eq = df[df[key_col].astype(str).str.startswith(seg)]
            out[exch] = dict(zip(
                eq[sym_col].astype(str).str.strip().str.upper(),
                eq[key_col].astype(str),
            ))
        except Exception:
            continue
    return out


def _upstox_candles(instrument_key: str) -> pd.DataFrame | None:
    """Fetch ~2y of weekly candles from Upstox public v2 endpoint."""
    to_d   = date.today().isoformat()
    from_d = (date.today() - timedelta(days=760)).isoformat()
    url = (f"https://api.upstox.com/v2/historical-candle/"
           f"{urllib.parse.quote(instrument_key, safe='')}/week/{to_d}/{from_d}")
    _UPSTOX_LIMITER.wait()
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
        if r.status_code != 200:
            return None
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low",
                                            "Close", "Volume", "oi"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        df = df.set_index("ts").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _upstox_daily(instrument_key: str) -> pd.DataFrame | None:
    """~2y of DAILY candles. Note: this endpoint EXCLUDES today's candle."""
    to_d   = date.today().isoformat()
    from_d = (date.today() - timedelta(days=760)).isoformat()
    url = (f"https://api.upstox.com/v2/historical-candle/"
           f"{urllib.parse.quote(instrument_key, safe='')}/day/{to_d}/{from_d}")
    _UPSTOX_LIMITER.wait()
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
        if r.status_code != 200:
            return None
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low",
                                            "Close", "Volume", "oi"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
        df = df.set_index("ts").sort_index()
        df.index = df.index.normalize()
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _upstox_today(instrument_key: str) -> pd.DataFrame | None:
    """TODAY'S forming daily candle. v3 intraday endpoint, no auth needed."""
    url = (f"https://api.upstox.com/v3/historical-candle/intraday/"
           f"{urllib.parse.quote(instrument_key, safe='')}/days/1")
    _UPSTOX_LIMITER.wait()
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        if r.status_code != 200:
            return None
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low",
                                            "Close", "Volume", "oi"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
        df = df.set_index("ts").sort_index()
        df.index = df.index.normalize()
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _upstox_daily_live(instrument_key: str) -> pd.DataFrame | None:
    hist = _upstox_daily(instrument_key)
    if hist is None or hist.empty:
        return None
    today = _upstox_today(instrument_key)
    if today is not None and not today.empty:
        # intraday row wins on collision
        hist = pd.concat([hist[~hist.index.isin(today.index)], today]).sort_index()
    return hist


def fetch_upstox_weekly(symbol: str, min_bars: int = MIN_BARS,
                        live: bool = False, project_volume: bool = True
                        ) -> tuple[pd.DataFrame | None, str]:
    """Try NSE then BSE on Upstox. Returns (df, exchange_used).
    live=True → daily history + today's intraday candle, resampled to weekly,
    running week PRESERVED."""
    instruments = load_upstox_instruments()
    sym = symbol.strip().upper()
    for exch in ("NSE", "BSE"):
        ikey = instruments.get(exch, {}).get(sym)
        if not ikey:
            continue
        if live:
            daily = _upstox_daily_live(ikey)
            if daily is None or daily.empty:
                continue
            df = _clean_weekly(_daily_to_weekly(daily), min_bars, keep_partial=True)
            if df is not None:
                return _finalise_live(df, daily, project_volume), exch
        else:
            df = _clean_weekly(_upstox_candles(ikey), min_bars)
            if df is not None:
                return df, exch
    return None, ""


# ═══════════════════════════════════════════════════════════════════════
# 2. KITE (Zerodha)  — daily TOTP auto-login + historical daily→weekly
# ═══════════════════════════════════════════════════════════════════════
#
# Secrets expected in .streamlit/secrets.toml :
#
#   [kite]
#   api_key     = "xxxx"
#   api_secret  = "xxxx"
#   user_id     = "AB1234"        # optional — needed for TOTP auto-login
#   password    = "xxxx"          # optional — needed for TOTP auto-login
#   totp_secret = "BASE32SECRET"  # optional — needed for TOTP auto-login
#
# If user_id/password/totp_secret are absent, the sidebar lets you paste a
# manually-generated access_token instead.
# ═══════════════════════════════════════════════════════════════════════

def _kite_secrets() -> dict | None:
    try:
        s = st.secrets["kite"]
        return {k: s.get(k, "") for k in
                ("api_key", "api_secret", "user_id", "password", "totp_secret")}
    except Exception:
        return None


def kite_totp_login() -> tuple[str | None, str]:
    """
    Full daily auto-login: password → TOTP → request_token → access_token.
    Returns (access_token, message).
    """
    cfg = _kite_secrets()
    if not cfg or not cfg["api_key"] or not cfg["api_secret"]:
        return None, "No [kite] api_key/api_secret in secrets."
    if not (cfg["user_id"] and cfg["password"] and cfg["totp_secret"]):
        return None, ("TOTP auto-login needs user_id, password and totp_secret "
                      "in [kite] secrets. Alternatively paste an access_token "
                      "in the sidebar.")
    try:
        import pyotp
    except ImportError:
        return None, "pyotp not installed — add `pyotp` to requirements.txt."

    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})

        # Step 1 — password login
        r = s.post("https://kite.zerodha.com/api/login",
                   data={"user_id": cfg["user_id"], "password": cfg["password"]},
                   timeout=20)
        j = r.json()
        if j.get("status") != "success":
            return None, f"Kite password step failed: {j.get('message', r.text[:120])}"
        request_id = j["data"]["request_id"]

        # Step 2 — TOTP
        totp_now = pyotp.TOTP(cfg["totp_secret"].replace(" ", "")).now()
        r = s.post("https://kite.zerodha.com/api/twofa",
                   data={"user_id": cfg["user_id"], "request_id": request_id,
                         "twofa_value": totp_now, "twofa_type": "totp"},
                   timeout=20)
        j = r.json()
        if j.get("status") != "success":
            return None, f"Kite TOTP step failed: {j.get('message', r.text[:120])}"

        # Step 3 — hit connect/login to obtain request_token from redirects
        request_token = None
        url = f"https://kite.zerodha.com/connect/login?v=3&api_key={cfg['api_key']}"
        for _ in range(6):
            r = s.get(url, allow_redirects=False, timeout=20)
            loc = r.headers.get("Location", "")
            if "request_token=" in loc:
                request_token = urllib.parse.parse_qs(
                    urllib.parse.urlparse(loc).query)["request_token"][0]
                break
            if not loc:
                # some flows put it in the final URL after a 200
                if "request_token=" in r.url:
                    request_token = urllib.parse.parse_qs(
                        urllib.parse.urlparse(r.url).query)["request_token"][0]
                break
            url = loc if loc.startswith("http") else ("https://kite.zerodha.com" + loc)
        if not request_token:
            return None, ("Could not capture request_token — check that the app's "
                          "redirect URL is reachable, or paste a manual access_token.")

        # Step 4 — exchange for access_token
        checksum = hashlib.sha256(
            (cfg["api_key"] + request_token + cfg["api_secret"]).encode()
        ).hexdigest()
        r = requests.post("https://api.kite.trade/session/token",
                          data={"api_key": cfg["api_key"],
                                "request_token": request_token,
                                "checksum": checksum},
                          headers={"X-Kite-Version": "3"}, timeout=20)
        j = r.json()
        if j.get("status") != "success":
            return None, f"Token exchange failed: {j.get('message', r.text[:120])}"
        token = j["data"]["access_token"]
        st.session_state["kite_access_token"] = token
        st.session_state["kite_token_date"] = date.today().isoformat()
        return token, "✅ Kite auto-login successful — access token valid for today."
    except Exception as e:
        return None, f"Kite login error: {e}"


def kite_access_token() -> str | None:
    """Return today's cached access token (auto-login or manually pasted)."""
    if st.session_state.get("kite_token_date") == date.today().isoformat():
        return st.session_state.get("kite_access_token")
    return None


def set_kite_manual_token(token: str) -> None:
    st.session_state["kite_access_token"] = token.strip()
    st.session_state["kite_token_date"] = date.today().isoformat()


def _kite_headers() -> dict | None:
    cfg = _kite_secrets()
    token = kite_access_token()
    if not cfg or not cfg.get("api_key") or not token:
        return None
    return {"X-Kite-Version": "3",
            "Authorization": f"token {cfg['api_key']}:{token}"}


@st.cache_data(ttl=86400, show_spinner=False)
def load_kite_instruments(_token_marker: str) -> dict:
    """
    Download Kite instrument dumps for NSE + BSE.
    Returns {"NSE": {tradingsymbol: instrument_token}, "BSE": {...}}.
    _token_marker busts the cache when a new day's token is generated.
    """
    headers = _kite_headers()
    out = {"NSE": {}, "BSE": {}}
    if not headers:
        return out
    for exch in ("NSE", "BSE"):
        try:
            r = requests.get(f"https://api.kite.trade/instruments/{exch}",
                             headers=headers, timeout=60)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
            eq = df[df["instrument_type"] == "EQ"] if "instrument_type" in df.columns else df
            out[exch] = dict(zip(
                eq["tradingsymbol"].astype(str).str.strip().str.upper(),
                eq["instrument_token"].astype(int),
            ))
        except Exception:
            continue
    return out


def _kite_daily(instrument_token: int) -> pd.DataFrame | None:
    headers = _kite_headers()
    if not headers:
        return None
    to_d   = date.today().isoformat()
    from_d = (date.today() - timedelta(days=760)).isoformat()
    url = (f"https://api.kite.trade/instruments/historical/"
           f"{instrument_token}/day?from={from_d}&to={to_d}")
    _KITE_LIMITER.wait()
    try:
        r = requests.get(url, headers=headers, timeout=25)
        if r.status_code != 200:
            return None
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "Open", "High", "Low",
                                            "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
        df = df.set_index("ts").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception:
        return None


def _kite_quote(exch: str, symbol: str) -> dict | None:
    """Live full quote → {'last_price', 'volume', 'ohlc': {...}}."""
    headers = _kite_headers()
    if not headers:
        return None
    inst = f"{exch}:{symbol.strip().upper()}"
    _KITE_LIMITER.wait()
    try:
        r = requests.get("https://api.kite.trade/quote",
                         params={"i": inst}, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        return r.json().get("data", {}).get(inst)
    except Exception:
        return None


def _kite_patch_today(daily: pd.DataFrame, exch: str, symbol: str) -> pd.DataFrame:
    """Overwrite / append today's daily bar with the live quote. Kite's
    historical `day` candle can lag the tape by minutes intraday."""
    q = _kite_quote(exch, symbol)
    if not q:
        return daily
    ohlc = q.get("ohlc") or {}
    ltp  = q.get("last_price")
    vol  = q.get("volume") or q.get("volume_traded")
    if not ltp or not ohlc.get("open"):
        return daily
    today = pd.Timestamp(ist_now().date())
    prev_hi = float(daily["High"].loc[today]) if today in daily.index else 0.0
    prev_lo = float(daily["Low"].loc[today])  if today in daily.index else 1e18
    daily.loc[today, ["Open", "High", "Low", "Close", "Volume"]] = [
        float(ohlc["open"]),
        max(float(ohlc.get("high", ltp)), float(ltp), prev_hi),
        min(float(ohlc.get("low",  ltp)), float(ltp), prev_lo),
        float(ltp),
        float(vol or (daily["Volume"].loc[today] if today in daily.index else 0)),
    ]
    return daily.sort_index()


def fetch_kite_weekly(symbol: str, min_bars: int = MIN_BARS,
                      live: bool = False, project_volume: bool = True
                      ) -> tuple[pd.DataFrame | None, str]:
    """Try NSE then BSE on Kite. Daily candles resampled to weekly.
    live=True → today's bar patched from /quote, running week PRESERVED."""
    token = kite_access_token()
    if not token:
        return None, ""
    instruments = load_kite_instruments(token[:8])
    sym = symbol.strip().upper()
    for exch in ("NSE", "BSE"):
        itok = instruments.get(exch, {}).get(sym)
        if not itok:
            continue
        daily = _kite_daily(itok)
        if daily is None or daily.empty:
            continue
        daily.index = pd.DatetimeIndex(daily.index).normalize()
        if live:
            if market_is_open():
                daily = _kite_patch_today(daily, exch, sym)
            df = _clean_weekly(_daily_to_weekly(daily), min_bars, keep_partial=True)
            if df is not None:
                return _finalise_live(df, daily, project_volume), exch
        else:
            df = _clean_weekly(_daily_to_weekly(daily), min_bars)
            if df is not None:
                return df, exch
    return None, ""


# ═══════════════════════════════════════════════════════════════════════
# 3. YAHOO  (fallback — .NS then .BO)
# ═══════════════════════════════════════════════════════════════════════
def fetch_yahoo_weekly(symbol: str, min_bars: int = MIN_BARS,
                       live: bool = False, project_volume: bool = True
                       ) -> tuple[pd.DataFrame | None, str]:
    """Yahoo fallback. live=True resamples DAILY (Yahoo's today bar is ~15 min
    delayed) instead of using the 1wk interval."""
    import yfinance as yf
    sym = symbol.strip().upper()
    for suffix, exch in ((".NS", "NSE"), (".BO", "BSE")):
        try:
            daily = None
            if live:
                daily = yf.Ticker(sym + suffix).history(period="2y", interval="1d")
                if daily is None or daily.empty:
                    continue
                daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
                df = _clean_weekly(_daily_to_weekly(daily), min_bars, keep_partial=True)
                if df is not None:
                    return _finalise_live(df, daily, project_volume), exch
                continue
            df = yf.Ticker(sym + suffix).history(period="2y", interval="1wk")
        except Exception:
            df = None
        df = _clean_weekly(df, min_bars)
        if df is not None:
            return df, exch
    return None, ""


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED FETCH — the chain every symbol goes through
# ═══════════════════════════════════════════════════════════════════════
SOURCE_FETCHERS = {
    "Upstox": fetch_upstox_weekly,
    "Kite":   fetch_kite_weekly,
    "Yahoo":  fetch_yahoo_weekly,
}


def fetch_weekly_any(symbol: str, source_order: list[str],
                     min_bars: int = MIN_BARS,
                     live: bool = False, project_volume: bool = True
                     ) -> tuple[pd.DataFrame | None, str, str]:
    """
    Fetch weekly OHLCV for one symbol, trying each source in order and,
    inside each source, NSE first then BSE. Yahoo is always the final
    safety net even if the user removed it from the order.

    live=True  → the CURRENT, still-forming week is included as the last bar,
                 built from daily candles + today's live/intraday candle.
                 df.attrs carries: partial_week, elapsed_sessions,
                 raw_partial_volume, volume_projected, asof.

    Returns (df, source_used, exchange_used).
    """
    order = list(source_order)
    if "Yahoo" not in order:
        order.append("Yahoo")
    for src in order:
        fetcher = SOURCE_FETCHERS.get(src)
        if fetcher is None:
            continue
        try:
            df, exch = fetcher(symbol, min_bars, live=live,
                               project_volume=project_volume)
        except Exception:
            df, exch = None, ""
        if df is not None:
            return df, src, exch
    return None, "", ""
