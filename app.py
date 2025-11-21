import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import math

# ---------------------------------------------------------
# 1. 페이지 설정 및 DB 초기화
# ---------------------------------------------------------
st.set_page_config(page_title="My Quant Portfolio", layout="wide")

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
    try:
        df_holdings = pd.read_sql("SELECT * FROM holdings", conn)
        df_cash = pd.read_sql("SELECT * FROM cash", conn)
    except:
        df_holdings = pd.DataFrame()
        df_cash = pd.DataFrame()
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

# ---------------------------------------------------------
# 2. 핵심 로직 함수들 (UI보다 반드시 위에 정의되어야 함)
# ---------------------------------------------------------

# [백테스팅 엔진]
def run_backtest(df, initial_cash, mode, target_weight, trigger_up, sell_pct, trigger_down, buy_pct):
    """
    mode: 'RATE' (평단가 기준 수익률) or 'WEIGHT' (전체 포트폴리오 내 비중)
    """
    cash = initial_cash
    start_price = df.iloc[0]['Close']
    
    # 초기 세팅: 자산의 'target_weight'% 만큼 매수하고 시작
    # WEIGHT 모드일 때 의미가 크며, RATE 모드일 때도 초기 진입 비중으로 활용
    initial_invest = (initial_cash * (target_weight / 100))
    shares = math.floor(initial_invest / start_price)
    cash -= shares * start_price
    avg_price = start_price
    
    history = [] 
    trade_log = []

    for date, row in df.iterrows():
        price = row['Close']
        
        # 현재 자산 상태 계산
        stock_val = shares * price
        total_val = cash + stock_val
        current_weight = (stock_val / total_val * 100) if total_val > 0 else 0
        
        # --- 매도(익절) 조건 체크 ---
        should_sell = False
        
        if mode == 'RATE': # 1. 평단가 기준 수익률
            if shares > 0 and price >= avg_price * (1 + trigger_up/100):
                should_sell = True
        elif mode == 'WEIGHT': # 2. 포트폴리오 비중 기준
            if current_weight >= target_weight + trigger_up:
                should_sell = True
                
        if should_sell:
            sell_qty = math.floor(shares * (sell_pct / 100))
            if sell_qty > 0:
                shares -= sell_qty
                cash += sell_qty * price
                profit_rate = (price - avg_price)/avg_price*100 if avg_price > 0 else 0
                trade_log.append({
                    "date": date, "type": "🔴 매도", "price": price, "qty": sell_qty, 
                    "cause": f"{'수익률' if mode=='RATE' else '비중'}({profit_rate:.1f}%/{current_weight:.1f}%)"
                })

        # --- 매수(추매) 조건 체크 ---
        should_buy = False
        
        if mode == 'RATE':
            # 평단가 대비 하락 or 보유량 0일때
            if price <= avg_price * (1 - trigger_down/100) or (shares == 0 and cash > price):
                should_buy = True
        elif mode == 'WEIGHT':
            # 목표 비중보다 낮아지면
            if current_weight <= target_weight - trigger_down:
                should_buy = True
        
        if should_buy:
            invest_amt = cash * (buy_pct / 100)
            buy_qty = math.floor(invest_amt / price)
            
            if buy_qty > 0:
                total_val_temp = (shares * avg_price) + (buy_qty * price)
                shares += buy_qty
                cash -= buy_qty * price
                avg_price = total_val_temp / shares
                
                trade_log.append({
                    "date": date, "type": "🔵 매수", "price": price, "qty": buy_qty, 
                    "cause": f"{'저가' if mode=='RATE' else '비중미달'}"
                })

        # 자산 기록
        total_asset = cash + (shares * price)
        history.append(total_asset)

    df['Strategy_Asset'] = history
    final_return = ((history[-1] - initial_cash) / initial_cash) * 100
    buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    
    return df, trade_log, final_return, buy_hold_return


# [최적화 콜백 함수] - 이 부분이 누락되어 에러가 났었습니다.
def optimize_params(df, fixed_b, fixed_d, target_w):
    if len(df) < 10:
        st.toast("❌ 데이터가 부족합니다.")
        return

    best_ret = -9999
    # 기본값 저장 (검색 실패 시 유지를 위해)
    best_params = (st.session_state.get('mode', 'RATE'), 
                   st.session_state.get('up_a', 10.0), 
                   st.session_state.get('down_c', 10.0))
    
    modes = ['RATE', 'WEIGHT']
    search_ranges = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
    
    st.toast("🤖 매매 기준과 파라미터를 전체 탐색 중입니다...")
    
    # Grid Search
    for m in modes:
        for a_val in search_ranges:
            for c_val in search_ranges:
                _, _, ret, _ = run_backtest(
                    df.copy(), 10000, 
                    m, target_w,
                    a_val, fixed_b, 
                    c_val, fixed_d
                )
                if ret > best_ret:
                    best_ret = ret
                    best_params = (m, a_val, c_val)
    
    # Session State 업데이트 (화면 리프레시 전 값 변경)
    st.session_state['mode'] = best_params[0]
    st.session_state['up_a'] = best_params[1]
    st.session_state['down_c'] = best_params[2]
    
    mode_kor = "평가액(수익률)" if best_params[0] == 'RATE' else "포트폴리오 비중"
    st.toast(f"✅ 최적값 발견! [{mode_kor}] 상한:{best_params[1]}% / 하한:{best_params[2]}%")


# ---------------------------------------------------------
# 3. UI 레이아웃 구성
# ---------------------------------------------------------

col_main, col_side = st.columns([3, 1])

# --- [우측 패널] 내 투자 현황 ---
with col_side:
    st.subheader("내 투자")
    my_stocks, my_cash = get_portfolio()
    
    current_cash = 10000.0 if my_cash.empty else my_cash.iloc[0]['amount']
    total_value = current_cash
    
    if not my_stocks.empty:
        for index, row in my_stocks.iterrows():
            ticker = row['ticker']
            shares = row['shares']
            try:
                cur_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                val = cur_price * shares
                total_value += val
                
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
    
    with st.expander("포트폴리오 수동 입력/수정"):
        input_ticker = st.text_input("티커 (예: TQQQ)").upper()
        input_shares = st.number_input("보유 수량", min_value=0, step=1)
        input_avg = st.number_input("평단가 ($)", min_value=0.0)
        if st.button("저장하기"):
            update_holding(input_ticker, input_shares, input_avg)
            st.rerun()

# --- [좌측 패널] 차트 및 분석 ---
with col_main:
    search_ticker = st.text_input("종목 검색", value="TQQQ" if my_stocks.empty else my_stocks.iloc[0]['ticker'])
    
    stock = yf.Ticker(search_ticker)
    hist = stock.history(period="6mo")
    
    last_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[-2]
    change = last_price - prev_price
    pct_change = (change / prev_price) * 100
    
    st.markdown(f"## {search_ticker} ${last_price:.2f} <span style='color:{'red' if change>0 else 'blue'}'>({pct_change:.2f}%)</span>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["🔄 리밸런싱", "📢 매매 추천", "📈 추세 예측"])
    
    # === Tab 1: 리밸런싱 ===
    with tab1:
        st.markdown("### 🛠️ 리밸런싱 & 변동성 수확 시뮬레이터")
        st.caption("과거 데이터를 바탕으로 '규칙 기반 매매' 시뮬레이션을 돌려보세요.")
        
        hist_1y = stock.history(period="1y")
        
        col_inputs, col_results = st.columns([1, 2])
        
        with col_inputs:
            st.markdown("#### ⚙️ 전략 설정")
            
            # Session State 초기화
            if 'mode' not in st.session_state: st.session_state['mode'] = 'RATE'
            if 'target_w' not in st.session_state: st.session_state['target_w'] = 50
            if 'up_a' not in st.session_state: st.session_state['up_a'] = 10.0
            if 'sell_b' not in st.session_state: st.session_state['sell_b'] = 50
            if 'down_c' not in st.session_state: st.session_state['down_c'] = 10.0
            if 'buy_d' not in st.session_state: st.session_state['buy_d'] = 50

            with st.container(border=True):
                mode_options = {'RATE': '📊 주식 평가액 (수익률) 기준', 'WEIGHT': '⚖️ 포트폴리오 비중 기준'}
                selected_mode = st.radio(
                    "매매 기준 선택", 
                    options=list(mode_options.keys()), 
                    format_func=lambda x: mode_options[x],
                    key='mode'
                )
                
                if selected_mode == 'WEIGHT':
                    st.session_state['target_w'] = st.slider("목표 주식 비중 (%)", 10, 90, key='target_w', step=10)
                
                st.divider()

                lbl_up = "A: 익절 기준 (+%)" if selected_mode == 'RATE' else "A: 비중 초과 허용 (+%p)"
                lbl_down = "C: 추매 기준 (-%)" if selected_mode == 'RATE' else "C: 비중 미달 허용 (-%p)"

                st.markdown("**매도(Sell) 규칙**")
                in_up_A = st.slider(lbl_up, 1.0, 30.0, key='up_a', step=0.5)
                in_sell_B = st.slider("B: 매도 물량 (보유량의 %)", 10, 100, key='sell_b', step=10)
                
                st.markdown("**매수(Buy) 규칙**")
                in_down_C = st.slider(lbl_down, 1.0, 30.0, key='down_c', step=0.5)
                in_buy_D = st.slider("D: 매수 물량 (현금의 %)", 10, 100, key='buy_d', step=10)

            # [중요] 위에서 정의한 함수를 연결
            st.button(
                "✨ 전략 완전 탐색 (Auto-Tune)", 
                on_click=optimize_params, 
                args=(hist_1y, in_sell_B, in_buy_D, st.session_state['target_w'])
            )
            
            if selected_mode == 'RATE':
                st.caption(f"💡 **해석**: 평단가 대비 **{in_up_A}%** 오르면 팔고, **{in_down_C}%** 내리면 삽니다.")
            else:
                tgt = st.session_state['target_w']
                st.caption(f"💡 **해석**: 주식 비중이 **{tgt + in_up_A:.1f}%**가 되면 팔고, **{tgt - in_down_C:.1f}%**가 되면 삽니다.")

        with col_results:
            if len(hist_1y) > 0:
                df_res, logs, final_ret, bh_ret = run_backtest(
                    hist_1y.copy(), 10000, 
                    st.session_state['mode'], st.session_state['target_w'],
                    in_up_A, in_sell_B, in_down_C, in_buy_D
                )
                
                m1, m2, m3 = st.columns(3)
                m1.metric("내 전략 수익률", f"{final_ret:.2f}%", delta=f"{final_ret - bh_ret:.2f}%p")
                m2.metric("단순 보유 수익률", f"{bh_ret:.2f}%")
                m3.metric("총 매매 횟수", f"{len(logs)}회")
                
                fig_back = go.Figure()
                fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Strategy_Asset'], 
                                    mode='lines', name='내 전략 자산', line=dict(color='#ef4444', width=2)))
                
                norm_factor = 10000 / df_res['Close'].iloc[0]
                fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Close']*norm_factor, 
                                    mode='lines', name='단순 보유', line=dict(color='#e5e7eb', dash='dot')))
                
                buy_dates = [x['date'] for x in logs if '매수' in x['type']]
                buy_prices = [df_res.loc[d]['Strategy_Asset'] for d in buy_dates]
                sell_dates = [x['date'] for x in logs if '매도' in x['type']]
                sell_prices = [df_res.loc[d]['Strategy_Asset'] for d in sell_dates]

                fig_back.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                                              name='매수', marker=dict(color='#3b82f6', symbol='triangle-up', size=10)))
                fig_back.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', 
                                              name='매도', marker=dict(color='#ef4444', symbol='triangle-down', size=10)))

                fig_back.update_layout(
                    title="자산 증감 추이 (1년)", 
                    margin=dict(t=30, b=0, l=0, r=0), 
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
                )
                st.plotly_chart(fig_back, use_container_width=True)
                
                with st.expander("📋 매매 기록 및 원인 상세"):
                    if logs:
                        st.dataframe(
                            pd.DataFrame(logs).style.format({'price': '${:.2f}'}), 
                            use_container_width=True
                        )
                    else:
                        st.info("설정된 조건에 맞는 매매 기록이 없습니다.")
                        
    # === Tab 2: 매매 추천 ===
    with tab2:
        st.write("### 퀀트 기반 매매 신호")
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

    # === Tab 3: 추세 예측 ===
    with tab3:
        st.write("### 향후 변동성 예측")
        daily_volatility = hist['Close'].pct_change().std()
        next_day_range = last_price * daily_volatility
        st.write(f"내일 예상 변동폭: ±${next_day_range:.2f}")
        st.caption("과거 6개월 변동성 기준 통계적 추정치입니다.")