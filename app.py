import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import math

# 1. 페이지 설정 (Wide 모드)
st.set_page_config(page_title="My Quant Portfolio", layout="wide")

# 2. DB 초기화 및 함수 (포트폴리오 저장용)
def init_db():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS holdings
                 (ticker TEXT PRIMARY KEY, shares INTEGER, avg_price REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS cash
                 (currency TEXT PRIMARY KEY, amount REAL)''')
    conn.commit()
    conn.close()

def get_portfolio():
    conn = sqlite3.connect('portfolio.db')
    df_holdings = pd.read_sql("SELECT * FROM holdings", conn)
    df_cash = pd.read_sql("SELECT * FROM cash", conn)
    conn.close()
    return df_holdings, df_cash

def update_holding(ticker, shares, avg_price):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    if shares == 0:
        c.execute("DELETE FROM holdings WHERE ticker=?", (ticker,))
    else:
        c.execute("INSERT OR REPLACE INTO holdings VALUES (?, ?, ?)", (ticker, shares, avg_price))
    conn.commit()
    conn.close()

init_db()

# 3. 사이드바가 아닌 메인 화면 분할 (3:1 비율)
col_main, col_side = st.columns([3, 1])

# --- [우측 패널] 내 투자 현황 (토스증권 우측 UI 모방) ---
with col_side:
    st.subheader("내 투자")
    
    # DB에서 데이터 로드
    my_stocks, my_cash = get_portfolio()
    
    # 예시: 현금이 없으면 초기화
    if my_cash.empty:
        current_cash = 10000.0 # 기본 $10,000
    else:
        current_cash = my_cash.iloc[0]['amount']
        
    total_value = current_cash
    
    # 보유 종목 표시 및 가치 계산
    if not my_stocks.empty:
        for index, row in my_stocks.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            try:
                cur_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                val = cur_price * shares
                total_value += val
                
                # 종목 카드 UI
                with st.container(border=True):
                    c1, c2 = st.columns([1, 1])
                    c1.markdown(f"**{ticker}**")
                    c1.caption(f"{shares}주")
                    profit = (cur_price - row['avg_price']) / row['avg_price'] * 100
                    color = "red" if profit > 0 else "blue"
                    c2.markdown(f"${val:,.2f}")
                    c2.markdown(f":{color}[{profit:.2f}%]")
            except:
                pass

    st.metric(label="총 자산 (USD)", value=f"${total_value:,.2f}")
    st.divider()
    
    # 포트폴리오 수정/입력 (Expander로 숨김)
    with st.expander("포트폴리오 수동 입력/수정"):
        input_ticker = st.text_input("티커 (예: TQQQ)").upper()
        input_shares = st.number_input("보유 수량", min_value=0, step=1)
        input_avg = st.number_input("평단가 ($)", min_value=0.0)
        if st.button("저장하기"):
            update_holding(input_ticker, input_shares, input_avg)
            st.rerun()

# --- [좌측 패널] 차트 및 분석 ---
with col_main:
    # 검색창 (헤더처럼)
    search_ticker = st.text_input("종목 검색", value="QQQ" if my_stocks.empty else my_stocks.iloc[0]['ticker'])
    
    # 데이터 가져오기
    stock = yf.Ticker(search_ticker)
    hist = stock.history(period="6mo")
    
    # 헤더 정보
    last_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[-2]
    change = last_price - prev_price
    pct_change = (change / prev_price) * 100
    
    st.markdown(f"## {search_ticker} ${last_price:.2f} <span style='color:{'red' if change>0 else 'blue'}'>({pct_change:.2f}%)</span>", unsafe_allow_html=True)

    # 1. 차트 (Plotly)
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 2. 기능 탭
    tab1, tab2, tab3 = st.tabs(["🔄 리밸런싱", "📢 매매 추천", "📈 추세 예측"])
    
    with tab1:
        st.write("### 리밸런싱 시뮬레이터")
        col_r1, col_r2 = st.columns(2)
        target_ratio = col_r1.slider("목표 주식 비중 (%)", 0, 100, 50)
        rebal_cond = col_r2.number_input("리밸런싱 트리거 (±%)", value=5.0)
        
        st.info(f"💡 {search_ticker} 비중이 {target_ratio}%에서 ±{rebal_cond}% 벗어나면 알림을 줍니다.")
        # 여기에 구체적인 온주 단위 계산 로직 추가 예정
        
    with tab2:
        st.write("### 퀀트 기반 매매 신호")
        # RSI 계산 예시
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        st.metric("현재 RSI (14일)", f"{rsi:.2f}")
        if rsi < 30:
            st.success("🟢 과매도 구간! (매수 검토)")
        elif rsi > 70:
            st.error("🔴 과매수 구간! (매도 검토)")
        else:
            st.warning("⚪ 중립 구간")

    with tab3:
        st.write("### 향후 변동성 예측")
        # 간단한 통계적 예측 예시
        daily_volatility = hist['Close'].pct_change().std()
        next_day_range = last_price * daily_volatility
        st.write(f"내일 예상 변동폭: ±${next_day_range:.2f}")
        st.caption("과거 6개월 변동성 기준 통계적 추정치입니다.")