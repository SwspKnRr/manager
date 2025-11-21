import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import math

# --- [백테스팅 엔진] ---
def run_backtest(df, initial_cash, trigger_up, sell_pct, trigger_down, buy_pct):
    """
    df: 데이터프레임 (Close 컬럼 필수)
    initial_cash: 초기 자본금 (USD)
    trigger_up: 익절 기준 (예: 10 -> 10% 상승시)
    sell_pct: 익절 물량 (예: 50 -> 보유량의 50% 매도)
    trigger_down: 추매 기준 (예: 10 -> 10% 하락시)
    buy_pct: 추매 물량 (예: 50 -> 현금의 50% 투입)
    """
    cash = initial_cash
    shares = 0
    avg_price = 0
    
    # 기록용 리스트
    history = [] 
    trade_log = [] # 매매 일지

    # 첫 날 종가로 50% 매수하고 시작한다고 가정 (혹은 100% 현금 시작 등 설정 가능)
    # 여기서는 시뮬레이션의 명확성을 위해 '100% 현금 시작 -> 첫 매수 기회를 기다림' 
    # 또는 '첫날 50:50 진입' 중 선택해야 하는데, 보통 리밸런싱은 보유 상태를 가정하므로
    # 첫날 자산의 50%를 시가에 매수한 것으로 세팅합니다.
    start_price = df.iloc[0]['Close']
    shares = math.floor((cash * 0.5) / start_price)
    cash -= shares * start_price
    avg_price = start_price
    
    for date, row in df.iterrows():
        price = row['Close']
        action = None
        trade_amt = 0
        
        # 1. 매도 조건 (익절) check
        # 평단가 대비 trigger_up% 이상 올랐는가?
        if shares > 0 and price >= avg_price * (1 + trigger_up/100):
            # 보유 수량의 sell_pct% 만큼 매도 (소수점 버림)
            sell_qty = math.floor(shares * (sell_pct / 100))
            if sell_qty > 0:
                shares -= sell_qty
                cash += sell_qty * price
                action = "SELL"
                trade_amt = sell_qty
                # 매도시 평단가는 변하지 않음 (FIFO 기준이 아니면)
                trade_log.append({"date": date, "type": "🔴 매도", "price": price, "qty": sell_qty, "profit": (price - avg_price)/avg_price*100})

        # 2. 매수 조건 (추매) check
        # 평단가(없으면 전날 종가 기준) 대비 trigger_down% 이하로 떨어졌는가?
        # 주식을 다 팔아서 shares가 0일 때는 직전 고점 대비 등을 따져야 하나, 
        # 여기서는 단순화하여 '직전 체결 평단가' 혹은 '보유 없으면 진입' 로직 적용 필요.
        # 편의상 shares가 0이면 무조건 진입하도록 설정하거나, 기준점을 잡아야 함.
        elif price <= avg_price * (1 - trigger_down/100) or (shares == 0 and cash > price):
            # 보유 현금의 buy_pct% 만큼 매수
            invest_amt = cash * (buy_pct / 100)
            buy_qty = math.floor(invest_amt / price)
            
            if buy_qty > 0:
                # 평단가 갱신 (이동평균법)
                total_val = (shares * avg_price) + (buy_qty * price)
                shares += buy_qty
                cash -= buy_qty * price
                avg_price = total_val / shares
                action = "BUY"
                trade_amt = buy_qty
                trade_log.append({"date": date, "type": "🔵 매수", "price": price, "qty": buy_qty, "new_avg": avg_price})

        # 일별 자산 가치 기록
        total_asset = cash + (shares * price)
        history.append(total_asset)

    df['Strategy_Asset'] = history
    
    # 결과 계산
    final_return = ((history[-1] - initial_cash) / initial_cash) * 100
    buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    
    return df, trade_log, final_return, buy_hold_return

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

# --- [우측 패널] 내 투자 현황 ---
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
        st.markdown("### 🛠️ 리밸런싱 & 변동성 수확 시뮬레이터")
        st.caption("과거 데이터를 바탕으로 '규칙 기반 매매'를 했을 때의 결과를 시뮬레이션합니다.")
        
        # 데이터 준비 (1년치)
        hist_1y = stock.history(period="1y")
        
        col_inputs, col_results = st.columns([1, 2])
        
        with col_inputs:
            st.markdown("#### ⚙️ 규칙 설정")
            
            # Session State 초기화 (슬라이더 값을 제어하기 위함)
            if 'up_a' not in st.session_state: st.session_state['up_a'] = 10.0
            if 'sell_b' not in st.session_state: st.session_state['sell_b'] = 50
            if 'down_c' not in st.session_state: st.session_state['down_c'] = 10.0
            if 'buy_d' not in st.session_state: st.session_state['buy_d'] = 50

            with st.container(border=True):
                st.markdown("**1. 익절(Sell) 규칙**")
                # key를 지정하여 session_state와 연동
                in_up_A = st.slider("A: 상승 트리거 (%)", 1.0, 30.0, key='up_a', step=0.5)
                in_sell_B = st.slider("B: 매도 비중 (%)", 10, 100, key='sell_b', step=10)
                
                st.divider()
                
                st.markdown("**2. 추매(Buy) 규칙**")
                in_down_C = st.slider("C: 하락 트리거 (%)", 1.0, 30.0, key='down_c', step=0.5)
                in_buy_D = st.slider("D: 현금 투입 비중 (%)", 10, 100, key='buy_d', step=10)

            # --- [최적 파라미터 찾기 로직] ---
            if st.button("✨ 최적 파라미터 찾기 (Auto-Tune)"):
                if len(hist_1y) < 10:
                    st.error("데이터가 부족합니다.")
                else:
                    best_ret = -9999
                    best_params = (0, 0)
                    
                    # 진행률 표시바
                    progress_text = "최적의 A(상승), C(하락) 트리거를 찾는 중..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # 탐색 범위 설정 (예: 3% ~ 20% 구간을 1%~2.5% 단위로 탐색)
                    # 너무 촘촘하면 느려지므로 적당한 간격 설정
                    search_ranges = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0]
                    total_steps = len(search_ranges) ** 2
                    current_step = 0
                    
                    # Grid Search 시작
                    for a_val in search_ranges:
                        for c_val in search_ranges:
                            # B와 D는 현재 설정된 값을 고정하고 A, C만 최적화
                            _, _, ret, _ = run_backtest(
                                hist_1y.copy(), 10000, 
                                a_val, in_sell_B, 
                                c_val, in_buy_D
                            )
                            
                            if ret > best_ret:
                                best_ret = ret
                                best_params = (a_val, c_val)
                            
                            current_step += 1
                            my_bar.progress(current_step / total_steps, text=progress_text)
                    
                    my_bar.empty()
                    
                    # 결과 적용 (Session State 업데이트)
                    st.session_state['up_a'] = best_params[0]
                    st.session_state['down_c'] = best_params[1]
                    
                    st.success(f"최적값 발견! 수익률: {best_ret:.2f}% (A={best_params[0]}%, C={best_params[1]}%)")
                    
                    # 화면 새로고침하여 슬라이더 값 반영
                    st.rerun()

        with col_results:
            # 현재 슬라이더 값으로 시뮬레이션 실행 및 결과 표시
            if len(hist_1y) > 0:
                # 초기 자본금 $10,000 가정
                df_res, logs, final_ret, bh_ret = run_backtest(
                    hist_1y.copy(), 10000, in_up_A, in_sell_B, in_down_C, in_buy_D
                )
                
                # 1. 수익률 비교 지표
                m1, m2, m3 = st.columns(3)
                m1.metric("내 전략 수익률", f"{final_ret:.2f}%", delta=f"{final_ret - bh_ret:.2f}%p (vs존버)")
                m2.metric("단순 보유(존버) 수익률", f"{bh_ret:.2f}%")
                m3.metric("매매 횟수", f"{len(logs)}회")
                
                # 2. 그래프 그리기 (Plotly)
                fig_back = go.Figure()
                # 전략 자산
                fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Strategy_Asset'], 
                                    mode='lines', name='전략 자산', line=dict(color='#ef4444', width=2))) # 토스 레드
                # 단순 보유
                norm_factor = 10000 / df_res['Close'].iloc[0]
                fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Close']*norm_factor, 
                                    mode='lines', name='단순 보유', line=dict(color='#e5e7eb', dash='dot')))
                
                # 매매 타점
                buy_dates = [x['date'] for x in logs if '매수' in x['type']]
                buy_prices = [df_res.loc[d]['Strategy_Asset'] for d in buy_dates]
                sell_dates = [x['date'] for x in logs if '매도' in x['type']]
                sell_prices = [df_res.loc[d]['Strategy_Asset'] for d in sell_dates]

                fig_back.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                                              name='매수', marker=dict(color='#3b82f6', symbol='triangle-up', size=12)))
                fig_back.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', 
                                              name='매도', marker=dict(color='#ef4444', symbol='triangle-down', size=12)))

                fig_back.update_layout(
                    title="자산 증감 추이 (1년)", 
                    xaxis_title="", 
                    yaxis_title="자산 가치 ($)", 
                    hovermode="x unified",
                    template="plotly_white",
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_back, use_container_width=True)
                
                # 3. 로그
                with st.expander("📋 매매 기록 상세"):
                    if logs:
                        st.dataframe(pd.DataFrame(logs).style.format({'price': '${:.2f}', 'profit': '{:.2f}%', 'new_avg': '${:.2f}'}), use_container_width=True)
                    else:
                        st.caption("매매 기록이 없습니다.")
        
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