import streamlit as st
import pandas as pd
import os

# --- 설정: 파일 이름 ---
CSV_FILE = "my_portfolio.csv"

# --- [추가됨] 폰트 설정 (CSS 주입) ---
def set_font_gulim():
    st.markdown("""
    <style>
    /* 전체 폰트 강제 적용 */
    html, body, [class*="css"] {
        font-family: 'Gulim', '굴림', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    set_font_gulim()  # <-- 실행 시 폰트 함수 호출
    st.title("📈 내 주식 포트폴리오 (심플 버전)")
    
# --- 1. 데이터 불러오기 함수 ---
def load_data():
    if not os.path.exists(CSV_FILE):
        # 파일이 없으면 빈 데이터프레임 생성
        return pd.DataFrame(columns=["종목명", "매수수량", "평균단가"])
    
    # 파일이 있으면 읽어오기
    return pd.read_csv(CSV_FILE)

# --- 2. 데이터 저장하기 함수 ---
def save_data(ticker, amount, price):
    df = load_data()
    
    # 새로운 데이터 한 줄 만들기
    new_data = pd.DataFrame({
        "종목명": [ticker],
        "매수수량": [amount],
        "평균단가": [price]
    })
    
    # 기존 데이터에 합치기
    df = pd.concat([df, new_data], ignore_index=True)
    
    # CSV 파일로 저장
    df.to_csv(CSV_FILE, index=False)

# --- 3. 메인 화면 ---
def main():
    st.title("📈 내 주식 포트폴리오 (심플 버전)")

    # 1. 현재 저장된 목록 보여주기
    st.subheader("📋 현재 보유 종목")
    df = load_data()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # (옵션) 초기화 버튼
        if st.button("⚠️ 전체 삭제 (초기화)"):
            if os.path.exists(CSV_FILE):
                os.remove(CSV_FILE)
                st.rerun()
    else:
        st.info("아직 저장된 종목이 없습니다.")

    st.markdown("---")

    # 2. 입력 폼
    st.subheader("➕ 종목 추가")
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        ticker = col1.text_input("종목명 (예: 삼성전자, AAPL)")
        amount = col2.number_input("수량", min_value=1, step=1)
        price = col3.number_input("평단가", min_value=0.0, step=100.0)
        
        if st.form_submit_button("저장하기"):
            if ticker:
                save_data(ticker, amount, price)
                st.success(f"{ticker} 저장 완료!")
                st.rerun() # 화면 새로고침해서 표 업데이트
            else:
                st.warning("종목명을 입력해주세요.")

if __name__ == "__main__":
    main()