import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

# =====================
# 字體設定（避免中文亂碼）
# =====================
font_path = "./NotoSansTC-Bold.ttf"
if fm.findSystemFonts(fontpaths=['.'], fontext='ttf'):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = 'Noto Sans TC'
else:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

matplotlib.rcParams['axes.unicode_minus'] = False

# =====================
# Streamlit UI
# =====================
st.title("📈 SMA200 趨勢策略回測系統")

symbol = st.text_input("輸入股票代號（如：00631L.TW, TQQQ）", "00631L.TW")
start_date = st.date_input("回測開始日期", pd.to_datetime("2023-01-01"))
end_date = st.date_input("回測結束日期", pd.to_datetime("2025-01-01"))
initial_capital = st.number_input("初始投入金額", value=10000)

if st.button("開始回測"):

    # ========= 抓資料 + 暖機 =========
    warmup_start = pd.to_datetime(start_date) - pd.Timedelta(days=400)

    df = yf.download(symbol, start=warmup_start, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["SMA200"] = df["Close"].rolling(200).mean()
    df = df.dropna().copy()

    # ========= 計算報酬 =========
    df["Return"] = df["Close"].pct_change().fillna(0)

    # ========= LRS 持倉規則 =========
    df["Position"] = 0

    # 第一筆強制買入
    df.loc[df.index[0], "Position"] = 1

    for i in range(1, len(df)):
        close = df["Close"].iloc[i]
        sma200 = df["SMA200"].iloc[i]

        if close > sma200:
            df.loc[df.index[i], "Position"] = 1
        else:
            df.loc[df.index[i], "Position"] = 0

    # ========= 現金版資金曲線（空倉不動）=========
    df["Equity_LRS"] = initial_capital
    cash_LRS = initial_capital
    holding_LRS = 0

    for i in range(1, len(df)):
        prev_pos = df["Position"].iloc[i - 1]
        today_pos = df["Position"].iloc[i]

        price_yesterday = df["Close"].iloc[i - 1]
        price_today = df["Close"].iloc[i]

        # 若昨天持有 → 今天依漲跌變化
        if prev_pos == 1:
            holding_LRS = holding_LRS * (price_today / price_yesterday)

        # 若昨天空倉 → 資金維持不變
        if prev_pos == 0:
            holding_LRS = 0  # 沒有部位

        # 若今天轉為持有（由 0 變 1）
        if prev_pos == 0 and today_pos == 1:
            holding_LRS = cash_LRS  # 把現金全部買進
            cash_LRS = 0

        # 若今天轉為空倉（由 1 變 0）
        if prev_pos == 1 and today_pos == 0:
            cash_LRS = holding_LRS
            holding_LRS = 0

        df.loc[df.index[i], "Equity_LRS"] = (holding_LRS + cash_LRS)

    # ========= Buy & Hold =========
    buy_price = df["Close"].iloc[0]
    shares = initial_capital / buy_price

    df["Equity_BH"] = shares * df["Close"]

    # ========= 切回使用者選的回測期間 =========
    df = df.loc[start_date:end_date].copy()

    # ========= 績效摘要 =========
    bh_final = df["Equity_BH"].iloc[-1]
    lrs_final = df["Equity_LRS"].iloc[-1]

    bh_return = (bh_final / initial_capital - 1) * 100
    lrs_return = (lrs_final / initial_capital - 1) * 100

    # ========= 印出結果 =========
    st.subheader("📊 最終績效比較")
    st.markdown(f"""
    **Buy & Hold 最終資產：** {bh_final:,.0f} 元  
    **LRS 最終資產：** {lrs_final:,.0f} 元  

    **Buy & Hold 報酬率：** {bh_return:.2f}%  
    **LRS 報酬率：** {lrs_return:.2f}%  
    """)

    # ========= 圖表 =========
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Equity_BH"], label="Buy & Hold")
    ax.plot(df.index, df["Equity_LRS"], label="LRS 趨勢策略")
    ax.set_title("資金曲線比較")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # ========= 買賣紀錄 =========
    df["Signal"] = df["Position"].diff().fillna(0)

    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]

    st.subheader("📌 交易次數")
    st.write(f"買進次數：{len(buys)} | 賣出次數：{len(sells)}")

    st.subheader("🟢 買進紀錄")
    st.dataframe(buys[["Close"]])

    st.subheader("🔴 賣出紀錄")
    st.dataframe(sells[["Close"]])
