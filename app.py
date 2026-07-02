import json
import re
import warnings
from datetime import datetime, date as _date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

from vpci_engine import analyze_stock_v3, DEFAULT_PARAMS
from data_sources import (
    fetch_weekly_any, load_upstox_instruments,
    kite_totp_login, kite_access_token, set_kite_manual_token,
)
from sector_history import (
    load_sector_map, attach_sector_columns, build_sector_leadership,
    save_weekly_snapshot, load_history_from_github, build_rotation_view,
    init_checkpoint, save_checkpoint, clear_checkpoint,
    checkpoint_to_json, restore_checkpoint_from_upload,
    CHECKPOINT_EVERY,
)

warnings.filterwarnings("ignore")

UNIVERSE_FILES = ["Stock_List.csv", "EQUITY_L_2.csv"]   # first one found wins
MARKET_FLAG = "IN"   # single Indian universe (NSE→BSE→Yahoo fallback per stock)


# ═══════════════════════════════════════════════════════════════════════════════
# RANKING HELPERS  (original pre-v4 weights — earnings phase removed)
# ═══════════════════════════════════════════════════════════════════════════════
def parse_mcap_to_crore(mcap_str):
    """Parse '386K crore inr' or '854 crore inr' or 'N/A' to crore float."""
    if pd.isna(mcap_str) or mcap_str == "N/A" or not str(mcap_str).strip():
        return np.nan
    s = str(mcap_str).lower().replace("crore inr", "").replace("inr", "").strip()
    m = re.match(r"([\d.]+)\s*k", s)
    if m:
        return float(m.group(1)) * 1000
    m = re.match(r"([\d.]+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def mcap_score(crore):
    """Sweet spot: 5,000 - 15,000 cr. Decays outside this band."""
    if pd.isna(crore) or crore <= 0:
        return 0.5
    if 5000 <= crore <= 15000:
        return 1.0
    if 2000 <= crore < 5000:
        return 0.6 + 0.4 * (crore - 2000) / 3000
    if 15000 < crore <= 30000:
        return 1.0 - 0.4 * (crore - 15000) / 15000
    if 1000 <= crore < 2000:
        return 0.3 + 0.3 * (crore - 1000) / 1000
    if 30000 < crore <= 100000:
        return 0.6 - 0.6 * (crore - 30000) / 70000
    return 0.0


def rank_stocks(df, include_relaxed=False, min_gates=7):
    """Rank stocks already passing the gates by composite score."""
    if include_relaxed:
        mask = (df["full_entry"] == True) | (df["relaxed_entry"] == True) | (df["gate_count"] >= min_gates)
    else:
        mask = df["full_entry"] == True

    pool = df[mask].copy().reset_index(drop=True)
    if len(pool) == 0:
        return pool

    pool["score_vpci"] = pool["vpci"].rank(pct=True)
    pool["score_rs"] = pool["rs_return"].rank(pct=True)
    pool["score_52w"] = (pool["pct_near_52w"].clip(0, 100) / 100.0)
    pool["score_tight"] = 1.0 - pool["risk_pct"].rank(pct=True)
    if "vol_ratio" in pool.columns:
        pool["score_vol"] = pool["vol_ratio"].rank(pct=True)
    else:
        pool["score_vol"] = 0.5
    pool["mcap_crore"] = pool["Market Cap"].apply(parse_mcap_to_crore)
    pool["score_mcap"] = pool["mcap_crore"].apply(mcap_score)

    weights = {
        "score_vpci":  0.25,
        "score_rs":    0.25,
        "score_52w":   0.20,
        "score_tight": 0.15,
        "score_vol":   0.10,
        "score_mcap":  0.05,
    }
    pool["composite_score"] = sum(pool[col] * w for col, w in weights.items())

    if "fresh_signal" in pool.columns:
        pool.loc[pool["fresh_signal"] == True, "composite_score"] *= 1.05

    pool = pool.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool


def rank_g4_pending(df):
    """
    Rank stocks where gate_count == 6 and ONLY G4 (breakout) is missing.
    Pre-breakout high-conviction watchlist candidates.
    """
    required_cols = ["gate_count", "g1", "g2", "g3", "g4", "g5", "g6", "g7"]
    for c in required_cols:
        if c not in df.columns:
            return pd.DataFrame()

    mask = (
        (df["gate_count"] == 6) &
        (df["g1"] == True) & (df["g2"] == True) & (df["g3"] == True) &
        (df["g4"] == False) &
        (df["g5"] == True) & (df["g6"] == True) & (df["g7"] == True)
    )

    pool = df[mask].copy().reset_index(drop=True)
    if len(pool) == 0:
        return pool

    pool["score_vpci"] = pool["vpci"].rank(pct=True)
    pool["score_rs"] = pool["rs_return"].rank(pct=True)
    pool["score_52w"] = (pool["pct_near_52w"].clip(0, 100) / 100.0)
    pool["score_tight"] = 1.0 - pool["risk_pct"].rank(pct=True)
    if "vol_ratio" in pool.columns:
        pool["score_vol"] = pool["vol_ratio"].rank(pct=True)
    else:
        pool["score_vol"] = 0.5
    pool["mcap_crore"] = pool["Market Cap"].apply(parse_mcap_to_crore)
    pool["score_mcap"] = pool["mcap_crore"].apply(mcap_score)
    if "pct_near_52w" in pool.columns:
        pool["score_proximity"] = pool["pct_near_52w"].clip(0, 100) / 100.0
    else:
        pool["score_proximity"] = 0.5

    weights = {
        "score_vpci":      0.20,
        "score_rs":        0.20,
        "score_52w":       0.15,
        "score_tight":     0.15,
        "score_vol":       0.10,
        "score_mcap":      0.05,
        "score_proximity": 0.15,
    }
    pool["composite_score"] = sum(pool[col] * w for col, w in weights.items())
    pool = pool.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + HIGH-CONTRAST / LARGE-FONT THEME
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Pawan Chaturvedi Screener", page_icon="📈", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 17px !important; }
    .stDataFrame { font-size: 16px !important; }
    div[data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 16px !important; }
    .stTabs [data-baseweb="tab"] { font-size: 17px !important; padding: 10px 18px; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00b89422, #00b89411);
        border-radius: 8px 8px 0 0;
    }
    .src-badge {
        display: inline-block; padding: 4px 14px; margin: 2px 6px 2px 0;
        border-radius: 14px; font-size: 16px; font-weight: 600;
    }
    .src-ok  { background: #00b89433; color: #00e6b8; border: 1px solid #00b894; }
    .src-off { background: #ff767533; color: #ff9f9e; border: 1px solid #ff7675; }
    .hero {
        text-align: center; padding: 18px 0 8px 0;
        background: linear-gradient(180deg, #00b89412, transparent);
        border-radius: 14px; margin-bottom: 4px;
    }
    .hero h1 { font-size: 3em; margin: 0; letter-spacing: 0.5px; }
    .hero p  { color: #9aa0a6; font-size: 1.25em; margin-top: 6px; }
</style>
<div class="hero">
    <h1>📈 Pawan's Screener</h1>
    <p>VPCI v3 Momentum & Breakout System • Multi-Source Indian Equities (Upstox • Kite • Yahoo)</p>
</div>
""", unsafe_allow_html=True)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE LOADER  (single Indian list — NSE symbols tried on NSE, BSE, Yahoo)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_universe():
    """Load the Indian stock universe. Returns (symbols, path_used)."""
    import os
    for path in UNIVERSE_FILES:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            if "companyId" in df.columns:
                syms = [str(t).strip().upper() for t in df["companyId"].dropna()]
            elif "SYMBOL" in df.columns:
                if "SERIES" in df.columns:
                    df = df[df["SERIES"] == "EQ"]
                syms = [str(t).strip().upper() for t in df["SYMBOL"].dropna()]
            else:
                continue
            syms = [s for s in dict.fromkeys(syms) if s and s != "NAN"]
            return syms, path
        except Exception:
            continue
    st.error("🚨 No universe file found. Upload Stock_List.csv (companyId, Name, Sector, Industry).")
    return ["RELIANCE", "TCS", "HDFCBANK"], None


def universe_path():
    _, p = get_universe()
    return p or UNIVERSE_FILES[0]


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING WORKER  — multi-source chain, screening logic untouched
# ═══════════════════════════════════════════════════════════════════════════════
def process_symbol(symbol, params, source_order):
    df, source, exchange = fetch_weekly_any(symbol, source_order)
    if df is None:
        return None
    r = analyze_stock_v3(symbol, df, params)
    if r:
        r["source"] = source
        r["exchange"] = exchange
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### ⚙️ Scan Configuration")
st.sidebar.info("🇮🇳 Indian equities only. Every stock is tried on **NSE first, "
                "then BSE**, on each data source in your priority order, with "
                "**Yahoo as the final safety net** — so nothing gets skipped.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 Data Source Priority")
source_order = st.sidebar.multiselect(
    "Order matters — first source is tried first",
    options=["Upstox", "Kite", "Yahoo"],
    default=["Upstox", "Kite", "Yahoo"],
    help="Upstox needs no API key. Kite needs today's access token (see panel "
         "below). Yahoo is always appended as the last fallback even if removed here."
)

# ── Kite daily login panel ──────────────────────────────────────────────
with st.sidebar.expander("🔑 Kite Connect — Daily Login", expanded=False):
    _tok = kite_access_token()
    if _tok:
        st.success(f"✅ Access token active for today (…{_tok[-4:]})")
        if st.button("🔄 Re-login (new token)", use_container_width=True):
            st.session_state.pop("kite_access_token", None)
            st.session_state.pop("kite_token_date", None)
            st.rerun()
    else:
        st.caption(
            "Kite tokens expire daily. Either auto-login with TOTP (needs "
            "`user_id`, `password`, `totp_secret` under `[kite]` in secrets) "
            "or paste today's access token manually."
        )
        if st.button("🔐 Auto-login with TOTP", use_container_width=True, type="primary"):
            with st.spinner("Logging into Kite..."):
                token, msg = kite_totp_login()
            (st.success if token else st.error)(msg)
            if token:
                st.rerun()
        _manual = st.text_input("Or paste access_token", type="password", key="kite_manual")
        if _manual:
            set_kite_manual_token(_manual)
            st.success("Manual token saved for today.")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Algorithm Settings")
relaxed_mode = st.sidebar.toggle("Relaxed Mode (Allow 6/7 Gates)", value=False)

with st.sidebar.expander("🛠️ Advanced Settings"):
    workers = st.slider("Parallel Workers", min_value=3, max_value=30, value=12,
                        help="Kite is rate-limited to ~2.5 req/s internally, so "
                             "more workers mainly speed up Upstox/Yahoo fetches.")
    test_limit = st.number_input("Limit Scan (0 = All)", min_value=0, max_value=10000, value=0)

st.sidebar.markdown("---")

# ── Resume / Recovery panel ─────────────────────────────────────────────
with st.sidebar.expander("💾 Resume & Recovery", expanded=False):
    st.caption(
        f"Scans checkpoint to memory every {CHECKPOINT_EVERY} symbols. If your "
        "session disconnects, click **Run Market Scan** again — completed "
        "symbols are skipped."
    )
    _ckpt_bytes = checkpoint_to_json(MARKET_FLAG)
    if _ckpt_bytes:
        st.success("Checkpoint available — the next scan will resume from it.")
        st.download_button(
            "⬇ Download checkpoint (.json)", data=_ckpt_bytes,
            file_name=f"vpci_checkpoint_{MARKET_FLAG}_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json", use_container_width=True,
        )
        if st.button("🗑 Clear checkpoint", use_container_width=True):
            clear_checkpoint(MARKET_FLAG)
            st.rerun()
    else:
        st.info("No checkpoint stored.")
    _restored = st.file_uploader(
        "Restore checkpoint", type=["json"], key="ckpt_uploader",
        help="Upload a previously-downloaded checkpoint to resume across sessions.",
    )
    if _restored is not None:
        ok, msg = restore_checkpoint_from_upload(_restored.read())
        (st.success if ok else st.error)(msg)

st.sidebar.markdown("---")
run_scan = st.sidebar.button("🚀 Run Market Scan", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN EXECUTION  — results stored in st.session_state so the UI (and the
# sector snapshot button) survives Streamlit reruns. This fixes the bug where
# clicking any button wiped the results and snapshots never saved.
# ═══════════════════════════════════════════════════════════════════════════════
if run_scan:
    params = {**DEFAULT_PARAMS, "relaxed": relaxed_mode}
    symbols, upath = get_universe()
    if test_limit > 0:
        symbols = symbols[:test_limit]

    # Pre-warm Upstox instrument master once (avoids 12 threads racing the download)
    if "Upstox" in source_order:
        with st.spinner("📥 Loading Upstox instrument master (cached daily)..."):
            _inst = load_upstox_instruments()
            n_nse, n_bse = len(_inst.get("NSE", {})), len(_inst.get("BSE", {}))
        if n_nse == 0 and n_bse == 0:
            st.warning("Upstox instrument master unavailable — Upstox will be skipped this run.")
        else:
            st.caption(f"Upstox master loaded: {n_nse:,} NSE + {n_bse:,} BSE instruments.")

    if "Kite" in source_order and not kite_access_token():
        st.warning("Kite is in your source order but no access token is active — "
                   "Kite will be skipped. Use the 🔑 sidebar panel to log in.")

    results, failed, remaining = init_checkpoint(MARKET_FLAG, symbols)
    done_symbols = [s for s in symbols if s not in remaining]
    if results or failed:
        st.success(f"♻️ Resumed from checkpoint — {len(done_symbols)} of "
                   f"{len(symbols)} already done, {len(remaining)} to go.")
    else:
        st.info(f"📚 Loaded **{len(symbols)}** symbols from `{upath}`. Commencing analysis...")

    progress_bar = st.progress(len(done_symbols) / max(len(symbols), 1))
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_symbol, s, params, source_order): s for s in remaining}
        done = len(done_symbols)
        since_ckpt = 0
        for f in as_completed(futs):
            done += 1
            since_ckpt += 1
            sym = futs[f]
            progress_bar.progress(done / max(len(symbols), 1))
            status_text.text(f"Scanning... {done} / {len(symbols)} processed")
            try:
                r = f.result()
                if r:
                    results.append(r)
                else:
                    failed.append(sym)
            except Exception:
                failed.append(sym)
            done_symbols.append(sym)
            if since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(MARKET_FLAG, symbols, results, failed, done_symbols)
                since_ckpt = 0

    save_checkpoint(MARKET_FLAG, symbols, results, failed, done_symbols)
    clear_checkpoint(MARKET_FLAG)
    status_text.empty()
    progress_bar.empty()

    if not results:
        st.session_state.pop("scan_df", None)
        st.session_state["scan_failed"] = failed
        st.warning("No candidates found. Check data source connectivity.")
    else:
        df = pd.DataFrame(results)

        def status_label(row):
            if row.get("fresh_signal"): return "🔥 FRESH BUY"
            if row.get("fresh_ext_signal"): return "🔥 FRESH EXT"
            if row.get("full_entry"): return "★ BUYABLE (7/7)"
            if row.get("gates_ready_ext"): return "⚡ 7/7 EXTENDED"
            if row.get("relaxed_entry"): return "★ RELAXED (6/7)"
            if row.get("gate_count", 0) >= 6 and row.get("tier1_pass"): return "◉ WATCHLIST (6/7)"
            if row.get("gate_count", 0) >= 5: return "▲ MOMENTUM (5+)"
            return "Other"

        df["status"] = df.apply(status_label, axis=1)
        df_sorted = df.sort_values(["gate_count", "pct_near_52w"], ascending=[False, False])

        # ── Market-cap enrichment (metadata only — yfinance) ─────────────
        top_symbols = df_sorted[df_sorted["gate_count"] >= 5]
        mcap_dict = {}
        if len(top_symbols) > 0:
            with st.spinner("Fetching live Market Cap metadata..."):
                def fetch_mcap(sym, exch):
                    try:
                        import yfinance as yf
                        yf_sym = f"{sym}.BO" if exch == "BSE" else f"{sym}.NS"
                        tkr = yf.Ticker(yf_sym)
                        try: return sym, tkr.fast_info['marketCap']
                        except Exception:
                            try: return sym, tkr.fast_info.market_cap
                            except Exception: return sym, tkr.info.get('marketCap', 0)
                    except Exception:
                        return sym, 0
                with ThreadPoolExecutor(max_workers=5) as executor:
                    fut_list = [executor.submit(fetch_mcap, r["symbol"], r.get("exchange", "NSE"))
                                for _, r in top_symbols.iterrows()]
                    for future in as_completed(fut_list):
                        sym, mval = future.result()
                        mcap_dict[sym] = mval

        df_sorted['raw_mcap'] = df_sorted['symbol'].map(mcap_dict).fillna(0)

        def format_mcap(mcap):
            if not mcap or mcap == 0: return "N/A"
            crores = mcap / 10000000
            if crores >= 1000: return f"{crores / 1000:.1f}K crore inr".replace(".0K", "K")
            return f"{crores:.0f} crore inr"
        df_sorted["Market Cap"] = df_sorted["raw_mcap"].apply(format_mcap)
        df_sorted = df_sorted.drop(columns=["raw_mcap"])

        # ── Company-name enrichment from the universe file ───────────────
        try:
            uni = pd.read_csv(universe_path())
            uni.columns = [str(c).strip() for c in uni.columns]
            if {"companyId", "Name"}.issubset(uni.columns):
                name_map = dict(zip(uni["companyId"].astype(str).str.strip().str.upper(),
                                    uni["Name"]))
                df_sorted.insert(1, "Company Name",
                                 df_sorted["symbol"].str.upper().map(name_map).fillna(""))
        except Exception:
            pass

        # ── Persist everything the UI needs ──────────────────────────────
        st.session_state["scan_df"] = df_sorted
        st.session_state["scan_failed"] = failed
        st.session_state["scan_time"] = datetime.now().isoformat(timespec="seconds")

        # ── AUTO-SAVE the sector snapshot (this used to require a button
        #    click that never worked across reruns) ──────────────────────
        _sector_map = load_sector_map(universe_path())
        if not _sector_map.empty:
            _df_sect = attach_sector_columns(df_sorted, _sector_map)
            _lead = build_sector_leadership(_df_sect)
            if not _lead["sector_both"].empty and _lead["sector_both"]["stocks_passing"].sum() > 0:
                ok, msg, _snap = save_weekly_snapshot(_lead, MARKET_FLAG)
                st.session_state["last_snapshot_msg"] = ("✅ " if ok else "ℹ️ ") + msg


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS RENDER  — reads session_state, so it survives every rerun
# ═══════════════════════════════════════════════════════════════════════════════
if "scan_df" not in st.session_state:
    st.info("👈 Configure your data sources and click **🚀 Run Market Scan** to begin. "
            "Results stay on screen until the next scan, even if you interact with buttons.")
else:
    df_sorted = st.session_state["scan_df"]
    failed = st.session_state.get("scan_failed", [])

    # ─── SCAN SUMMARY ───
    st.markdown(f"### 📊 Scan Summary "
                f"<span style='font-size:0.55em;color:#9aa0a6'>({st.session_state.get('scan_time','')})</span>",
                unsafe_allow_html=True)
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Total Symbols Analyzed", len(df_sorted) + len(failed))
    met2.metric("✅ Candidates Found", len(df_sorted))
    met3.metric("⚠️ No Data / Excluded", len(failed))
    met4.metric("★ Buyable (7/7)", int(df_sorted.get("full_entry", pd.Series(dtype=bool)).sum()))

    # ─── DATA SOURCE COVERAGE BADGES ───
    if "source" in df_sorted.columns:
        src_counts = df_sorted["source"].value_counts().to_dict()
        exch_counts = df_sorted["exchange"].value_counts().to_dict() if "exchange" in df_sorted.columns else {}
        badges = "".join(
            f"<span class='src-badge src-ok'>{s}: {src_counts.get(s, 0):,}</span>"
            for s in ["Upstox", "Kite", "Yahoo"] if src_counts.get(s, 0) > 0
        )
        badges += "".join(
            f"<span class='src-badge src-ok'>🏛 {e}: {exch_counts.get(e, 0):,}</span>"
            for e in ["NSE", "BSE"] if exch_counts.get(e, 0) > 0
        )
        if failed:
            badges += f"<span class='src-badge src-off'>Not found anywhere: {len(failed)}</span>"
        st.markdown(f"**Data coverage:** {badges}", unsafe_allow_html=True)

    if failed:
        with st.expander(f"⚠️ {len(failed)} symbols not found on any source (Upstox → Kite → Yahoo)"):
            st.dataframe(pd.DataFrame({"symbol": sorted(failed)}),
                         use_container_width=True, hide_index=True, height=250)
            st.download_button("⬇ Download failed list (.csv)",
                               data="\n".join(["symbol"] + sorted(failed)).encode(),
                               file_name=f"unresolved_{_date.today():%Y%m%d}.csv",
                               mime="text/csv")
    st.divider()

    # ─── UI COPY, TRADINGVIEW LINKS (exchange-aware) ───
    df_ui = df_sorted.copy()

    def tv_link(row):
        exch = row.get("exchange", "NSE") or "NSE"
        return f"https://in.tradingview.com/chart/?symbol={exch}:{row['symbol']}"
    df_ui["symbol"] = df_ui.apply(tv_link, axis=1)

    cols = list(df_ui.columns)
    if "Market Cap" in cols:
        cols.remove("Market Cap")
        insert_idx = cols.index("Company Name") + 1 if "Company Name" in cols else (
                     cols.index("symbol") + 1 if "symbol" in cols else 0)
        cols.insert(insert_idx, "Market Cap")
    df_ui = df_ui[[c for c in cols if c in df_ui.columns]]

    tv_config = {
        "symbol": st.column_config.LinkColumn("Symbol", display_text=r".*symbol=(?:NSE:|BSE:)?(.*)")
    }

    def add_tv(frame):
        out = frame.copy()
        out["symbol"] = out.apply(tv_link, axis=1)
        return out

    # ─── TABS ───
    tab1, tab2, tab3, tab4, tab5, tab6, tab8, tab9 = st.tabs([
        "🔥 Fresh Signals", "★ Buyable (7/7)", "◉ Watchlist (6/7)",
        "All Results", "🏆 Ranked", "🎯 G4 Pending",
        "🏭 Sector Leadership", "📈 Sector Rotation"
    ])

    with tab1:
        fresh_df = df_ui[df_ui["status"].isin(["🔥 FRESH BUY", "🔥 FRESH EXT"])]
        if not fresh_df.empty:
            st.dataframe(fresh_df, use_container_width=True, column_config=tv_config, hide_index=True)
        else:
            st.info("No fresh breakout signals detected this week.")

    with tab2:
        buyable_df = df_ui[df_ui["status"] == "★ BUYABLE (7/7)"]
        if not buyable_df.empty:
            st.dataframe(buyable_df, use_container_width=True, column_config=tv_config, hide_index=True)
        else:
            st.info("No stocks passed all 7 criteria.")

    with tab3:
        watch_df = df_ui[df_ui["status"].isin(["★ RELAXED (6/7)", "◉ WATCHLIST (6/7)"])]
        if not watch_df.empty:
            st.dataframe(watch_df, use_container_width=True, column_config=tv_config, hide_index=True)
        else:
            st.info("No watchlist candidates found.")

    with tab4:
        st.dataframe(df_ui, use_container_width=True, column_config=tv_config, hide_index=True)
        all_csv = df_sorted.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Results CSV", data=all_csv,
                           file_name=f"vpci_all_{datetime.now():%Y%m%d_%H%M}.csv",
                           mime="text/csv", key="all_download")

    with tab5:
        st.subheader("All Ranked Buyable Candidates")
        st.caption("Composite score: VPCI 25% + RS 25% + 52wH 20% + Tight base 15% + Volume 10% + Mcap fit 5%")
        try:
            ranked = rank_stocks(df_sorted, include_relaxed=False)
        except Exception as e:
            st.error(f"Ranker error: {e}")
            ranked = pd.DataFrame()

        if len(ranked) > 0:
            ranked_ui = add_tv(ranked)
            display_cols = [
                "rank", "symbol", "Company Name", "Market Cap", "close",
                "composite_score", "score_vpci", "score_rs", "score_52w",
                "score_tight", "score_vol", "score_mcap", "source", "exchange", "status"
            ]
            display_cols = [c for c in display_cols if c in ranked_ui.columns]
            st.dataframe(
                ranked_ui[display_cols].style.format({
                    "composite_score": "{:.3f}", "score_vpci": "{:.2f}",
                    "score_rs": "{:.2f}", "score_52w": "{:.2f}",
                    "score_tight": "{:.2f}", "score_vol": "{:.2f}",
                    "score_mcap": "{:.2f}", "close": "{:.2f}",
                }),
                use_container_width=True, hide_index=True, column_config=tv_config
            )
            st.info(f"Total ranked: {len(ranked)} stocks. Top 5 typically have score > 0.80. "
                    f"Score gap from #1 to #5 indicates conviction strength.")
            ranked_csv = ranked[[c for c in display_cols if c in ranked.columns]] \
                .to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Ranked CSV", data=ranked_csv,
                               file_name=f"vpci_ranked_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv", key="ranked_download")
        else:
            st.warning("No 7/7 stocks to rank this week. Try the Watchlist tab for 6/7 candidates.")

    with tab6:
        st.subheader("🎯 G4 Pending — Pre-Breakout Watchlist")
        st.caption("All 6 other gates passing; only the G4 breakout above the 13-week high is missing. "
                   "Watch these for the breakout.")
        try:
            g4_pending = rank_g4_pending(df_sorted)
        except Exception as e:
            st.error(f"G4 ranker error: {e}")
            g4_pending = pd.DataFrame()

        if len(g4_pending) > 0:
            g4_ui = add_tv(g4_pending)
            g4_display_cols = [
                "rank", "symbol", "Company Name", "Market Cap", "close", "composite_score",
                "score_vpci", "score_rs", "score_52w", "score_tight",
                "score_vol", "score_mcap", "score_proximity", "source", "exchange", "status"
            ]
            g4_display_cols = [c for c in g4_display_cols if c in g4_ui.columns]
            st.dataframe(
                g4_ui[g4_display_cols].style.format({
                    "composite_score": "{:.3f}", "score_vpci": "{:.2f}",
                    "score_rs": "{:.2f}", "score_52w": "{:.2f}",
                    "score_tight": "{:.2f}", "score_vol": "{:.2f}",
                    "score_mcap": "{:.2f}", "score_proximity": "{:.2f}",
                    "close": "{:.2f}",
                }),
                use_container_width=True, hide_index=True, column_config=tv_config
            )
            st.info(f"Total G4-pending: {len(g4_pending)} stocks. Click any symbol to open its "
                    f"chart and watch for the breakout above the 13w high.")
            g4_csv = g4_pending[[c for c in g4_display_cols if c in g4_pending.columns]] \
                .to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download G4 Pending CSV", data=g4_csv,
                               file_name=f"vpci_g4_pending_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv", key="g4_pending_download")
        else:
            st.warning("No stocks meet the strict G4-pending criteria this week "
                       "(all 6 other gates passing, only G4 missing).")

    # ═══════════════════════════════════════════════════════════════════
    # Sector & Industry Leadership (G5 + G6 + G7)
    # ═══════════════════════════════════════════════════════════════════
    with tab8:
        st.subheader("🏭 Leading Sectors & Industries — G5 (VPCI) + G6 (Leader RS) + G7 (Above 40w)")
        st.caption(
            "Sectors and industries ranked by the count of stocks where the institutional "
            "confirmation gates fire together. G5 = VPCI accumulation; G6 = leader RS positive "
            "vs index; G7 = price above the 40-week MA (Stage 2 trend). When several "
            "constituents of the same sector pass all three, accumulation, relative strength "
            "and trend are aligned — the highest-conviction sector for fresh capital this week."
        )
        if st.session_state.get("last_snapshot_msg"):
            st.caption(f"Snapshot status: {st.session_state['last_snapshot_msg']}")

        _sector_map = load_sector_map(universe_path())
        if _sector_map.empty:
            st.warning("Sector map could not be loaded. Make sure the universe CSV contains "
                       "companyId, Name, Sector, Industry columns.")
        else:
            _df_sect = attach_sector_columns(df_sorted, _sector_map)
            _lead = build_sector_leadership(_df_sect)
            _both = _lead["sector_both"]
            _ind = _lead["industry_both"]

            if _both.empty or _both["stocks_passing"].sum() == 0:
                st.info("No stocks pass G5, G6 and G7 together in this scan.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Stocks passing G5 ∧ G6 ∧ G7", int(_both["stocks_passing"].sum()))
                c2.metric("Sectors with ≥1 leader", int((_both["stocks_passing"] > 0).sum()))
                c3.metric("Top sector",
                          _both.iloc[0]["sector"] if len(_both) else "—",
                          f"{int(_both.iloc[0]['stocks_passing'])} stocks" if len(_both) else "")
                st.divider()

                colA, colB = st.columns(2)
                with colA:
                    st.markdown("##### 🥇 Sectors — G5 ∧ G6 ∧ G7 (strict)")
                    st.caption("All three gates passing — institutional confirmation cut.")
                    st.dataframe(_both.head(20), use_container_width=True, hide_index=True)
                with colB:
                    st.markdown("##### 🏷️ Industries — G5 ∧ G6 ∧ G7 (strict)")
                    st.caption("Drill-down within sectors — the actual sub-theme leading.")
                    st.dataframe(_ind.head(20), use_container_width=True, hide_index=True)

                with st.expander("📊 Broader breadth — Sectors / Industries with G5 ∨ G6 ∨ G7 firing"):
                    st.caption("A wider net for early sector accumulation: stocks passing any of the "
                               "three institutional gates — spots sectors just starting to rotate.")
                    cE1, cE2 = st.columns(2)
                    with cE1:
                        st.markdown("**Sectors — G5 ∨ G6 ∨ G7**")
                        st.dataframe(_lead["sector_either"].head(20),
                                     use_container_width=True, hide_index=True)
                    with cE2:
                        st.markdown("**Industries — G5 ∨ G6 ∨ G7**")
                        st.dataframe(_lead["industry_either"].head(20),
                                     use_container_width=True, hide_index=True)

                with st.expander("🔬 Stock-level view — every passer with sector tags"):
                    st.dataframe(_lead["stock_table"], use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("##### 💾 Weekly leadership snapshot")
                st.caption(
                    "A snapshot is now **saved automatically after every scan** (to GitHub if "
                    "secrets are configured, otherwise to this session). The button below "
                    "re-saves manually — e.g. if the auto-save failed on a network blip."
                )
                if st.button("💾 Save snapshot again", key="save_snap"):
                    ok, msg, snap = save_weekly_snapshot(_lead, MARKET_FLAG)
                    st.session_state["last_snapshot_msg"] = ("✅ " if ok else "ℹ️ ") + msg
                    (st.success if ok else st.warning)(msg)

                _snap_payload = {
                    "scan_date": _date.today().isoformat(),
                    "scan_timestamp": datetime.now().isoformat(timespec="seconds"),
                    "market": MARKET_FLAG,
                    "sector_both": _lead["sector_both"].to_dict(orient="records"),
                    "industry_both": _lead["industry_both"].to_dict(orient="records"),
                    "sector_either": _lead["sector_either"].to_dict(orient="records"),
                    "industry_either": _lead["industry_either"].to_dict(orient="records"),
                }
                st.download_button(
                    "⬇ Download snapshot (.json)",
                    data=json.dumps(_snap_payload, indent=2).encode("utf-8"),
                    file_name=f"sector_snapshot_{MARKET_FLAG}_{_date.today():%Y-%m-%d}.json",
                    mime="application/json", key="dl_snap",
                )
                st.download_button(
                    "⬇ Download sector leaders (.csv)",
                    data=_both.to_csv(index=False).encode("utf-8"),
                    file_name=f"sector_leaders_{MARKET_FLAG}_{_date.today():%Y-%m-%d}.csv",
                    mime="text/csv", key="dl_sect_csv",
                )

    # ═══════════════════════════════════════════════════════════════════
    # Sector Rotation (week-over-week history)
    # ═══════════════════════════════════════════════════════════════════
    with tab9:
        st.subheader("📈 Sector Rotation — Week-Over-Week History")
        st.caption(
            "Counts of stocks passing G5 ∧ G6 ∧ G7 per sector across saved weekly snapshots. "
            "A sector accelerating from 2 → 5 → 9 leaders over three weeks is rotating in; "
            "a decaying one is rotating out. Snapshots auto-save after every scan; configure "
            "GitHub secrets to persist them across container restarts."
        )

        _hist = load_history_from_github()
        _hist += st.session_state.get("snapshots", [])

        _seen, _dedup = set(), []
        for s in sorted(_hist, key=lambda d: d.get("scan_date", "")):
            key = (s.get("scan_date"), s.get("market"))
            if key in _seen:
                continue
            _seen.add(key)
            _dedup.append(s)
        # Accept both new "IN" snapshots and legacy "NSE" snapshots
        _hist = [s for s in _dedup if s.get("market") in (MARKET_FLAG, "NSE")]

        if not _hist:
            st.info("No history yet. Run a scan — a snapshot is saved automatically — "
                    "and each saved week will appear here.")
        else:
            st.success(f"Loaded {len(_hist)} weekly snapshot(s).")
            _rot = build_rotation_view(_hist, top_n=12)
            if not _rot.empty:
                st.markdown("##### Sector × Week — stocks passing G5 ∧ G6 ∧ G7")
                st.dataframe(_rot.style.background_gradient(axis=None, cmap="Greens"),
                             use_container_width=True)
                if _rot.shape[1] >= 2:
                    _delta = (_rot.iloc[:, -1] - _rot.iloc[:, -2]).sort_values(ascending=False)
                    cR1, cR2 = st.columns(2)
                    with cR1:
                        st.markdown("##### 🚀 Rotating IN (week-over-week)")
                        st.dataframe(_delta[_delta > 0].head(8).rename("Δ leaders").to_frame(),
                                     use_container_width=True)
                    with cR2:
                        st.markdown("##### 🪂 Rotating OUT (week-over-week)")
                        st.dataframe(_delta[_delta < 0].head(8).rename("Δ leaders").to_frame(),
                                     use_container_width=True)
            else:
                st.info("Need at least one snapshot with sector data to render the rotation grid.")
