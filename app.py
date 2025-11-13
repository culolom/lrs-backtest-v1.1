# app.py — LRS 回測系統（自動偵測台股代號 + 美化報表 + 真實持倉模擬）

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
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — SMA/EMA 回測系統</h1>", unsafe_allow_html=True)

# === 自動補 .TW 的函式 ===
def normalize_symbol(symbol):
    """讓使用者輸入 0050 / 2330 / 00878 時自動補上 .TW"""
    s = symbol.strip().upper()
    if s.isdigit() or (not "." in s and (s.startswith("00") or s.startswith("23") or s.startswith("008"))):
        s += ".TW"
    return s

# === 函式：取得商品可用資料區間 ===
@st.cache_data(show_spinner=False)
def get_available_range(symbol):
    hist = yf.Ticker(symbol).history(period="max", auto_adjust=True)
    if hist.empty:
        return pd.to_datetime("1990-01-01").date(), dt.date.today()
    return hist.index.min().date(), hist.index.max().date()

# === 使用者輸入 ===
col1, col2, col3 = st.columns(3)
with col1:
    raw_symbol = st.text_input("輸入代號（例：00631L.TW, QQQ, 0050, 2330）", "0050")

symbol = normalize_symbol(raw_symbol)

# 若使用者更換代號，自動偵測日期範圍
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

# === 主程式 ===
if st.button("開始回測 🚀"):
    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)
    with st.spinner("資料下載中…（自動多抓一年暖機資料）"):
        df_raw = yf.download(symbol, start=start_early, end=end)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty:
        st.error(f"⚠️ 無法下載 {symbol} 的資料，請確認代號或時間區間。")
        st.stop()

    df = df_raw.copy()
    df["MA"] = (
        df["Close"].rolling(window=window).mean()
        if ma_type == "SMA"
        else df["Close"].ewm(span=window, adjust=False).mean()
    )

    # === 生成訊號（第一天強制買入） ===
    df["Signal"] = 0
    df.loc[df.index[0], "Signal"] = 1
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["MA"].iloc[i] and df["Close"].iloc[i - 1] <= df["MA"].iloc[i - 1]:
            df.loc[df.index[i], "Signal"] = 1
        elif df["Close"].iloc[i] < df["MA"].iloc[i] and df["Close"].iloc[i - 1] >= df["MA"].iloc[i - 1]:
            df.loc[df.index[i], "Signal"] = -1
        else:
            df.loc[df.index[i], "Signal"] = 0

    # === 持倉 ===
    position, current = [], 1
    for sig in df["Signal"]:
        if sig == 1:
            current = 1
        elif sig == -1:
            current = 0
        position.append(current)
    df["Position"] = position

    # === 回報 ===
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    # === 真實資金曲線 ===
    df["Equity_LRS"] = 1.0
    for i in range(1, len(df)):
        if df["Position"].iloc[i - 1] == 1:
            df.loc[df.index[i], "Equity_LRS"] = df["Equity_LRS"].iloc[i - 1] * (1 + df["Return"].iloc[i])
        else:
            df.loc[df.index[i], "Equity_LRS"] = df["Equity_LRS"].iloc[i - 1]

    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["Equity_LRS"] /= df["Equity_LRS"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    df["LRS_Capital"] = df["Equity_LRS"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital

    # === 買賣點 ===
    buy_points = [(df.index[i], df["Close"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == 1]
    sell_points = [(df.index[i], df["Close"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == -1]
    buy_count, sell_count = len(buy_points), len(sell_points)

    # === 指標 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = (df.index[-1] - df.index[0]).days / 365
    cagr_lrs = (1 + final_return_lrs) ** (1 / years_len) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    def calc_metrics(series):
        daily = series.dropna()
        avg = daily.mean()
        std = daily.std()
        downside = daily[daily < 0].std()
        vol = std * np.sqrt(252)
        sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
        sortino = (avg / downside) * np.sqrt(252) if downside > 0 else np.nan
        return vol, sharpe, sortino

    vol_lrs, sharpe_lrs, sortino_lrs = calc_metrics(df["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = calc_metrics(df["Return"])

    equity_lrs_final = df["LRS_Capital"].iloc[-1]
    equity_bh_final = df["BH_Capital"].iloc[-1]

    # === 圖表 ===
    st.markdown("<h2 style='margin-top:1em;'>📈 策略績效視覺化</h2>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("收盤價與均線（含買賣點）", "資金曲線：LRS vs Buy&Hold"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")), row=1, col=1)
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進",
                                 marker=dict(color="green", symbol="triangle-up", size=8)), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出",
                                 marker=dict(color="red", symbol="x", size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略",
                             line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], name="Buy & Hold",
                             line=dict(color="gray", dash="dot")), row=2, col=1)
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # === 美化報表 ===
    st.markdown("""
    <style>
    .custom-table { width:100%; border-collapse:collapse; margin-top:1.2em; font-family:"Noto Sans TC"; }
    .custom-table th { background:#f5f6fa; padding:12px; font-weight:700; border-bottom:2px solid #ddd; }
    .custom-table td { text-align:center; padding:10px; border-bottom:1px solid #eee; font-size:15px; }
    .custom-table tr:nth-child(even) td { background-color:#fafbfc; }
    .custom-table tr:hover td { background-color:#f1f9ff; }
    .section-title td { background:#eef4ff; color:#1a237e; font-weight:700; font-size:16px; text-align:left; padding:10px 15px; }
    </style>
    """, unsafe_allow_html=True)

    html_table = f"""
    <table class='custom-table'>
    <thead><tr><th>指標名稱</th><th>LRS 策略</th><th>Buy & Hold</th></tr></thead>
    <tbody>
    <tr><td>最終資產</td><td>{equity_lrs_final:,.0f} 元</td><td>{equity_bh_final:,.0f} 元</td></tr>
    <tr><td>總報酬</td><td>{final_return_lrs:.2%}</td><td>{final_return_bh:.2%}</td></tr>
    <tr><td>年化報酬</td><td>{cagr_lrs:.2%}</td><td>{cagr_bh:.2%}</td></tr>
    <tr><td>最大回撤</td><td>{mdd_lrs:.2%}</td><td>{mdd_bh:.2%}</td></tr>
    <tr><td>年化波動率</td><td>{vol_lrs:.2%}</td><td>{vol_bh:.2%}</td></tr>
    <tr><td>夏普值</td><td>{sharpe_lrs:.2f}</td><td>{sharpe_bh:.2f}</td></tr>
    <tr><td>索提諾值</td><td>{sortino_lrs:.2f}</td><td>{sortino_bh:.2f}</td></tr>
    <tr class='section-title'><td colspan='3'>💹 交易統計</td></tr>
    <tr><td>買進次數</td><td>{buy_count}</td><td>—</td></tr>
    <tr><td>賣出次數</td><td>{sell_count}</td><td>—</td></tr>
    </tbody></table>
    """
    st.markdown(html_table, unsafe_allow_html=True)
    st.success("✅ 回測完成！（支援自動辨識台股代號）")
