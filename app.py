import streamlit as st
import pandas as pd
import os
import yfinance as yf

# --- 1. 폰트 설정 (굴림) ---
def set_font_gulim():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Gulim', '굴림', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 데이터 저장/로드 설정 ---
CSV_FILE = "my_portfolio.csv"

def load_data():
    # 1. 파일이 아예 없으면 -> 빈 데이터프레임 생성
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["종목코드", "종목명", "매수수량", "평균단가"])
    
    # 2. 파일 읽기 시도
    try:
        df = pd.read_csv(CSV_FILE)
        
        # [핵심 수정] 필수 컬럼인 '종목코드'가 있는지 확인
        if "종목코드" not in df.columns:
            # 예전 형식의 파일이라면 -> 깡통으로 리셋 (에러 방지)
            return pd.DataFrame(columns=["종목코드", "종목명", "매수수량", "평균단가"])
            
        return df
        
    except Exception:
        # 파일이 깨졌거나 읽을 수 없으면 -> 빈 것으로 리셋
        return pd.DataFrame(columns=["종목코드", "종목명", "매수수량", "평균단가"])

def save_data(code, name, amount, price):
    df = load_data()
    new_data = pd.DataFrame({
        "종목코드": [code],
        "종목명": [name],
        "매수수량": [amount],
        "평균단가": [price]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def delete_file():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

# --- 3. 현재가 가져오기 및 수익률 계산 함수 ---
def get_market_data(df):
    if df.empty:
        return df

    current_prices = []
    
    # 진행상황 바 (로딩 중 표시)
    progress_text = "현재가 조회 중입니다..."
    my_bar = st.progress(0, text=progress_text)
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        ticker = row['종목코드']
        try:
            # yfinance로 데이터 가져오기
            stock = yf.Ticker(ticker)
            # 가장 최근 1일치 데이터 조회
            hist = stock.history(period="1d")
            
            if not hist.empty:
                # 종가(Close) 가져오기
                current_price = hist['Close'].iloc[-1]
            else:
                current_price = row['평균단가'] # 조회 실패 시 평단가로 대체 (에러 방지)
                
        except Exception:
            current_price = row['평균단가']
            
        current_prices.append(current_price)
        # 진행률 업데이트
        my_bar.progress((idx + 1) / total_rows, text=progress_text)

    my_bar.empty() # 로딩 바 제거

    # 데이터프레임에 계산 결과 추가
    df['현재가'] = current_prices
    df['평가금액'] = df['현재가'] * df['매수수량']
    df['투자원금'] = df['평균단가'] * df['매수수량']
    df['평가손익'] = df['평가금액'] - df['투자원금']
    df['수익률(%)'] = ((df['현재가'] - df['평균단가']) / df['평균단가']) * 100
    
    return df

# --- 4. 메인 UI ---
def main():
    set_font_gulim()
    st.title("📈 내 주식 포트폴리오 (수익률 Ver.)")

    # 사이드바에 입력 폼 배치
    with st.sidebar:
        st.header("➕ 종목 추가")
        with st.form("input_form"):
            st.info("💡 한국 주식은 끝에 .KS(코스피), .KQ(코스닥)을 붙여주세요.\n예: 삼성전자(005930.KS), 에코프로(086520.KQ), 애플(AAPL)")
            code = st.text_input("종목코드 (예: 005930.KS, AAPL)")
            name = st.text_input("종목명 (예: 삼성전자)")
            amount = st.number_input("수량", min_value=1, step=1)
            price = st.number_input("평단가", min_value=0.0, step=100.0)
            
            if st.form_submit_button("저장하기"):
                if code and name:
                    save_data(code, name, amount, price)
                    st.success(f"{name} 저장 완료!")
                    st.rerun()
                else:
                    st.warning("종목코드와 종목명을 입력해주세요.")
        
        st.markdown("---")
        if st.button("⚠️ 데이터 전체 초기화"):
            delete_file()
            st.rerun()

    # 메인 화면: 포트폴리오 현황
    st.subheader("📊 나의 자산 현황")
    
    raw_df = load_data()
    
    if not raw_df.empty:
        # 계산 로직 실행
        result_df = get_market_data(raw_df)

        # 총 자산 요약 보여주기 (Metrics)
        total_invest = result_df['투자원금'].sum()
        total_eval = result_df['평가금액'].sum()
        total_profit = result_df['평가손익'].sum()
        total_rate = (total_profit / total_invest * 100) if total_invest > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("총 투자원금", f"{total_invest:,.0f}원")
        col2.metric("총 평가금액", f"{total_eval:,.0f}원")
        col3.metric("총 손익", f"{total_profit:,.0f}원", f"{total_rate:.2f}%")

        st.markdown("---")
        
        # 상세 표 보여주기 (수익률에 색상 입히기)
        # 보기 좋게 컬럼 순서 및 포맷 정리
        display_df = result_df[['종목명', '매수수량', '평균단가', '현재가', '수익률(%)', '평가손익']]
        
        st.dataframe(
            display_df.style.format({
                '평균단가': '{:,.0f}',
                '현재가': '{:,.0f}',
                '수익률(%)': '{:.2f}%',
                '평가손익': '{:,.0f}'
            }).background_gradient(subset=['수익률(%)'], cmap='RdYlGn', vmin=-30, vmax=30),
            use_container_width=True
        )
        
        st.caption("* '새로고침'을 하거나 종목을 추가하면 현재가가 업데이트됩니다.")

    else:
        st.info("👈 왼쪽 사이드바에서 종목을 추가해주세요.")

if __name__ == "__main__":
    main()