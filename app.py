# app.py — LRS 回測系統（自動偵測台股代號 + 真實持倉 + Benchmark 對照）

import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import matplotlib.font_manager as fm
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 字型設定 ===
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="LRS 回測系統", page_icon="📈", layout="wide")
st.markdown(
    "<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — SMA/EMA 回測系統（含 Benchmark）</h1>",
    unsafe_allow_html=True,
)

# === 工具函式 ===
def normalize_symbol(symbol: str) -> str:
    """讓使用者輸入 0050 / 2330 / 00878 時自動補上 .TW，其它代號不動"""
    s = symbol.strip().upper()
    if s.isdigit() or (("." not in s) and (s.startswith("00") or s.startswith("23") or s.startswith("008"))):
        s += ".TW"
    return s


@st.cache_data(show_spinner=False)
def get_available_range(symbol: str):
    """抓該商品最早 / 最晚有資料的日期，用來限制 date_input"""
    hist = yf.Ticker(symbol).history(period="max", auto_adjust=True)
    if hist.empty:
        return pd.to_datetime("1990-01-01").date(), dt.date.today()
    return hist.index.min().date(), hist.index.max().date()


def calc_vol_sharpe_sortino(daily_ret: pd.Series):
    daily = daily_ret.dropna()
    if daily.empty:
        return np.nan, np.nan, np.nan
    avg = daily.mean()
    std = daily.std()
    downside = daily[daily < 0].std()
    vol = std * np.sqrt(252)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
    sortino = (avg / downside) * np.sqrt(252) if downside > 0 else np.nan
    return vol, sharpe, sortino


def fmt_money(x):
    return "—" if pd.isna(x) else f"{x:,.0f} 元"


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x, nd=2):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


# === 使用者輸入 ===
col1, col2, col3 = st.columns(3)
with col1:
    raw_symbol = st.text_input("輸入代號（例：00631L.TW, QQQ, 0050, 2330）", "0050")

symbol = normalize_symbol(raw_symbol)

# 自動偵測可用日期區間（當代號改變時更新）
if "last_symbol" not in st.session_state or st.session_state.last_symbol != symbol:
    st.session_state.last_symbol = symbol
    min_start, max_end = get_available_range(symbol)
    st.session_state.min_start = min_start
    st.session_state.max_end = max_end
else:
    min_start = st.session_state.min_start
    max_end = st.session_state.max_end

st.info(f"🔎 {symbol} 可用資料區間：{min_start} ~ {max_end}")

with col2:
    start = st.date_input(
        "開始日期",
        value=max(min_start, pd.to_datetime("2013-01-01").date()),
        min_value=min_start,
        max_value=max_end,
        format="YYYY/MM/DD",
    )
with col3:
    end = st.date_input(
        "結束日期",
        value=max_end,
        min_value=min_start,
        max_value=max_end,
        format="YYYY/MM/DD",
    )

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 10, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 1000, 1_000_000, 10000, step=1000)

st.markdown("---")

# Benchmark 選項
use_benchmark = st.checkbox("加入大盤 Benchmark 對照")
if use_benchmark:
    b_col1, b_col2 = st.columns([2, 3])
    with b_col1:
        benchmark_raw = st.text_input("Benchmark 代號（例：SPY、VT、0050）", "SPY")
    benchmark_symbol = normalize_symbol(benchmark_raw)
else:
    benchmark_symbol = None

# === 主程式 ===
if st.button("開始回測 🚀"):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    # 暖機：多抓一年資料，只用來算均線，不直接算報酬
    start_early = start_ts - pd.Timedelta(days=365)

    with st.spinner("資料下載中…（自動多抓一年暖機資料）"):
        df_raw = yf.download(symbol, start=start_early, end=end_ts)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        if use_benchmark:
            df_bench_raw = yf.download(benchmark_symbol, start=start_early, end=end_ts)
            if isinstance(df_bench_raw.columns, pd.MultiIndex):
                df_bench_raw.columns = df_bench_raw.columns.get_level_values(0)
        else:
            df_bench_raw = None

    if df_raw.empty:
        st.error(f"⚠️ 無法下載 {symbol} 的資料，請確認代號或時間區間。")
        st.stop()

    # === 主商品：均線、訊號、真實持倉 ===
    df_full = df_raw.copy()
    df_full["MA"] = (
        df_full["Close"].rolling(window=window).mean()
        if ma_type == "SMA"
        else df_full["Close"].ewm(span=window, adjust=False).mean()
    )

    # 訊號：第一天強制買入，其餘依均線黃金/死亡交叉
    df_full["Signal"] = 0
    df_full.iloc[0, df_full.columns.get_loc("Signal")] = 1  # 第一個交易日強制買

    for i in range(1, len(df_full)):
        c_now = df_full["Close"].iloc[i]
        c_prev = df_full["Close"].iloc[i - 1]
        ma_now = df_full["MA"].iloc[i]
        ma_prev = df_full["MA"].iloc[i - 1]

        if pd.isna(ma_now) or pd.isna(ma_prev):
            df_full.iloc[i, df_full.columns.get_loc("Signal")] = 0
            continue

        if (c_now > ma_now) and (c_prev <= ma_prev):
            df_full.iloc[i, df_full.columns.get_loc("Signal")] = 1  # 買進訊號
        elif (c_now < ma_now) and (c_prev >= ma_prev):
            df_full.iloc[i, df_full.columns.get_loc("Signal")] = -1  # 賣出訊號
        else:
            df_full.iloc[i, df_full.columns.get_loc("Signal")] = 0

    # 持倉狀態（1 = 全部投入, 0 = 空手）
    position = []
    current_pos = 1  # 因為第一天已經強制買入
    for sig in df_full["Signal"]:
        if sig == 1:
            current_pos = 1
        elif sig == -1:
            current_pos = 0
        position.append(current_pos)
    df_full["Position"] = position

    # 日報酬 & 策略日報酬
    df_full["Return"] = df_full["Close"].pct_change().fillna(0)
    df_full["Strategy_Return"] = df_full["Return"] * df_full["Position"]

    # 真實持倉資金曲線（空手時不複利）
    df_full["Equity_LRS"] = 1.0
    for i in range(1, len(df_full)):
        if df_full["Position"].iloc[i - 1] == 1:
            df_full.iloc[i, df_full.columns.get_loc("Equity_LRS")] = (
                df_full["Equity_LRS"].iloc[i - 1] * (1 + df_full["Return"].iloc[i])
            )
        else:
            df_full.iloc[i, df_full.columns.get_loc("Equity_LRS")] = df_full["Equity_LRS"].iloc[i - 1]

    # Buy & Hold：單純全程持有
    df_full["Equity_BuyHold"] = (1 + df_full["Return"]).cumprod()

    # 只取使用者指定的區間（暖機不算報酬）
    df = df_full.loc[start_ts:end_ts].copy()

    # 將兩條 Equity 都歸一到區間第一天
    df["Equity_LRS"] /= df["Equity_LRS"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    # === Benchmark 處理 ===
    if df_bench_raw is not None and not df_bench_raw.empty:
        bench_full = df_bench_raw.copy()
        bench_full["Return"] = bench_full["Close"].pct_change().fillna(0)
        bench_full["Equity_Bench"] = (1 + bench_full["Return"]).cumprod()

        # 切同一區間並對齊主商品交易日（用前一日價格補齊）
        bench = bench_full.loc[df.index.min(): df.index.max()].copy()
        bench = bench.reindex(df.index, method="ffill")

        # 歸一
        bench["Equity_Bench"] /= bench["Equity_Bench"].iloc[0]

        # 將 Benchmark 曲線與報酬放回 df
        df["Equity_Bench"] = bench["Equity_Bench"]
        df["Bench_Return"] = df["Equity_Bench"].pct_change().fillna(0)
    else:
        df["Equity_Bench"] = np.nan
        df["Bench_Return"] = np.nan

    # === 投入本金換算成資金曲線 ===
    df["LRS_Capital"] = df["Equity_LRS"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital
    df["Bench_Capital"] = df["Equity_Bench"] * initial_capital if "Equity_Bench" in df else np.nan

    # === 買賣點（只標在選定區間內） ===
    buy_points = [(idx, df.loc[idx, "Close"]) for idx in df.index[1:] if df.loc[idx, "Signal"] == 1]
    sell_points = [(idx, df.loc[idx, "Close"]) for idx in df.index[1:] if df.loc[idx, "Signal"] == -1]
    buy_count, sell_count = len(buy_points), len(sell_points)

    # === 總報酬 / CAGR / MDD ===
    years_len = (df.index[-1] - df.index[0]).days / 365.0 if len(df) > 1 else np.nan

    def calc_from_equity(eq: pd.Series):
        if eq.isna().all():
            return np.nan, np.nan, np.nan
        total = eq.iloc[-1] - 1
        if years_len > 0:
            cagr = eq.iloc[-1] ** (1 / years_len) - 1
        else:
            cagr = np.nan
        mdd = 1 - (eq / eq.cummax()).min()
        return total, cagr, mdd

    final_return_lrs, cagr_lrs, mdd_lrs = calc_from_equity(df["Equity_LRS"])
    final_return_bh, cagr_bh, mdd_bh = calc_from_equity(df["Equity_BuyHold"])
    final_return_bench, cagr_bench, mdd_bench = calc_from_equity(df["Equity_Bench"])

    # 年化波動 / 夏普 / 索提諾
    vol_lrs, sharpe_lrs, sortino_lrs = calc_vol_sharpe_sortino(df["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = calc_vol_sharpe_sortino(df["Return"])
    vol_bench, sharpe_bench, sortino_bench = calc_vol_sharpe_sortino(df["Bench_Return"])

    # 最終資產
    equity_lrs_final = df["LRS_Capital"].iloc[-1]
    equity_bh_final = df["BH_Capital"].iloc[-1]
    equity_bench_final = df["Bench_Capital"].iloc[-1] if not df["Bench_Capital"].isna().all() else np.nan

    # === 視覺化 ===
    st.markdown("<h2 style='margin-top:1em;'>📈 策略績效視覺化</h2>", unsafe_allow_html=True)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("收盤價與均線（含買賣點）", "資金曲線：LRS vs Buy&Hold vs Benchmark"),
    )

    # row 1：價格 + 均線 + 買賣點
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")),
        row=1,
        col=1,
    )
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="markers",
                name="買進",
                marker=dict(color="green", symbol="triangle-up", size=8),
            ),
            row=1,
            col=1,
        )
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="markers",
                name="賣出",
                marker=dict(color="red", symbol="x", size=8),
            ),
            row=1,
            col=1,
        )

    # row 2：資金曲線（歸一後）
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略", line=dict(color="green")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Equity_BuyHold"],
            name="Buy & Hold",
            line=dict(color="gray", dash="dot"),
        ),
        row=2,
        col=1,
    )
    if not df["Equity_Bench"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Equity_Bench"],
                name=f"Benchmark ({benchmark_symbol})",
                line=dict(color="purple", dash="dash"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # === CSS 美化報表 ===
    st.markdown(
        """
    <style>
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1.2em;
        font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        border-radius: 10px;
        overflow: hidden;
    }
    .custom-table th {
        background-color: #f5f6fa;
        color: #2c3e50;
        text-align: center;
        padding: 12px;
        font-weight: 700;
        border-bottom: 2px solid #e0e0e0;
    }
    .custom-table td {
        text-align: center;
        padding: 10px;
        border-bottom: 1px solid #e9e9e9;
        font-size: 15px;
    }
    .custom-table tr:nth-child(even) td {
        background-color: #fafbfc;
    }
    .custom-table tr:hover td {
        background-color: #f1f9ff;
    }
    .custom-table .section-title td {
        background-color: #eef4ff;
        color: #1a237e;
        font-weight: 700;
        font-size: 16px;
        text-align: left;
        padding: 10px 15px;
        border-top: 2px solid #cfd8dc;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # === 綜合績效報表（LRS / Buy&Hold / Benchmark 三欄對照） ===
    html_table = f"""
    <table class='custom-table'>
    <thead>
        <tr>
            <th>指標名稱</th>
            <th>LRS 策略</th>
            <th>Buy & Hold</th>
            <th>Benchmark</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>最終資產</td>
            <td>{fmt_money(equity_lrs_final)}</td>
            <td>{fmt_money(equity_bh_final)}</td>
            <td>{fmt_money(equity_bench_final)}</td>
        </tr>
        <tr><td>總報酬</td>
            <td>{fmt_pct(final_return_lrs)}</td>
            <td>{fmt_pct(final_return_bh)}</td>
            <td>{fmt_pct(final_return_bench)}</td>
        </tr>
        <tr><td>年化報酬</td>
            <td>{fmt_pct(cagr_lrs)}</td>
            <td>{fmt_pct(cagr_bh)}</td>
            <td>{fmt_pct(cagr_bench)}</td>
        </tr>
        <tr><td>最大回撤</td>
            <td>{fmt_pct(mdd_lrs)}</td>
            <td>{fmt_pct(mdd_bh)}</td>
            <td>{fmt_pct(mdd_bench)}</td>
        </tr>
        <tr><td>年化波動率</td>
            <td>{fmt_pct(vol_lrs)}</td>
            <td>{fmt_pct(vol_bh)}</td>
            <td>{fmt_pct(vol_bench)}</td>
        </tr>
        <tr><td>夏普值</td>
            <td>{fmt_num(sharpe_lrs)}</td>
            <td>{fmt_num(sharpe_bh)}</td>
            <td>{fmt_num(sharpe_bench)}</td>
        </tr>
        <tr><td>索提諾值</td>
            <td>{fmt_num(sortino_lrs)}</td>
            <td>{fmt_num(sortino_bh)}</td>
            <td>{fmt_num(sortino_bench)}</td>
        </tr>
        <tr class='section-title'><td colspan='4'>💹 交易統計</td></tr>
        <tr><td>買進次數</td>
            <td>{buy_count}</td>
            <td>—</td>
            <td>—</td>
        </tr>
        <tr><td>賣出次數</td>
            <td>{sell_count}</td>
            <td>—</td>
            <td>—</td>
        </tr>
    </tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    if use_benchmark and not df["Equity_Bench"].isna().all():
        st.success("✅ 回測完成！（含 Benchmark 對照，採真實持倉模擬）")
    else:
        st.success("✅ 回測完成！（採真實持倉模擬）")

