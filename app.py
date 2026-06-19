"""
도금 폐수 Cr(VI) 처리용 활성탄 산정 시뮬레이터
Author: 김남윤 (충남대 환경공학과)
Data: AC-O3-Al 흡착제 기반 실험 데이터
Version: 2.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 실험 기반 파라미터
# ============================================
PARAMS = {
    'qmax': 19.9030,           # mg/g - Langmuir 최대 흡착 용량
    'KL': 0.0284,              # L/mg - Langmuir 친화도
    'KF': 1.0660,              # Freundlich KF
    'n_freundlich': 1.6103,    # Freundlich n
    'k2': 0.096044,            # g/(mg·min) - PSO 속도 상수
    'qe_kinetic': 2.008,       # mg/g - PSO qe (200ppm 기준)
    't_eq_hr': 4,              # 평형 시간
    'pH_opt': 3,               # 최적 pH
    'column_efficiency': 1.3,  # 컬럼 효율 (회분식 대비)
}

STANDARDS = {
    '청정지역': 0.1,
    '가지역': 0.5,
    '나지역': 0.5,
    '특례지역': 2.0,
}

# ============================================
# 흡착 모델 함수
# ============================================

def langmuir(Ce):
    return (PARAMS['qmax'] * PARAMS['KL'] * Ce) / (1 + PARAMS['KL'] * Ce)

def freundlich(Ce):
    return PARAMS['KF'] * Ce ** (1/PARAMS['n_freundlich'])

def pH_correction(pH):
    sigma = 2.5
    return np.exp(-(pH - PARAMS['pH_opt'])**2 / (2 * sigma**2))

def calculate_treatment(C_in, AC_dose_g_per_L, pH, model='Langmuir'):
    """질량 보존 방정식: C_in - Ce = qe(Ce) * dose"""
    pH_factor = pH_correction(pH)
    
    def mass_balance(Ce):
        if model == 'Langmuir':
            qe = langmuir(Ce) * pH_factor
        else:
            qe = freundlich(Ce) * pH_factor
        return C_in - Ce - qe * AC_dose_g_per_L
    
    try:
        Ce = brentq(mass_balance, 1e-6, C_in - 1e-9, maxiter=200)
    except (ValueError, RuntimeError):
        Ce = C_in * 0.5
    
    if model == 'Langmuir':
        qe = langmuir(Ce) * pH_factor
    else:
        qe = freundlich(Ce) * pH_factor
    
    efficiency = (C_in - Ce) / C_in * 100 if C_in > 0 else 0
    return Ce, efficiency, qe

def calculate_AC_requirement(C_in, flow_m3_day, pH, region='가지역', 
                              safety_factor=2.0, model='Langmuir'):
    """현실적인 활성탄 산정 (컬럼 운전 가정)"""
    target = STANDARDS[region]
    
    # 회분식 기준 dose 산정 (이분법)
    AC_dose_low = 0.01
    AC_dose_high = 500
    AC_dose_mid = 1.0
    
    for _ in range(60):
        AC_dose_mid = (AC_dose_low + AC_dose_high) / 2
        Ce, _, _ = calculate_treatment(C_in, AC_dose_mid, pH, model)
        if Ce > target * 0.8:
            AC_dose_low = AC_dose_mid
        else:
            AC_dose_high = AC_dose_mid
        if abs(AC_dose_high - AC_dose_low) < 0.005:
            break
    
    # 컬럼 효율 고려 + 안전인자 적용
    AC_dose_design = (AC_dose_mid / PARAMS['column_efficiency']) * safety_factor
    
    Ce_final, eff_final, qe_final = calculate_treatment(
        C_in, AC_dose_design * PARAMS['column_efficiency'], pH, model)
    
    # 일일 활성탄 소비량 = 일일 Cr 부하 / qe_design
    daily_Cr_g = C_in * flow_m3_day  # g/day
    qe_design = qe_final / safety_factor  # 안전인자 반영
    
    if qe_design > 0:
        daily_AC_kg = (daily_Cr_g / qe_design) / 1000
    else:
        daily_AC_kg = 0
    
    monthly_AC_kg = daily_AC_kg * 30
    yearly_AC_kg = daily_AC_kg * 365
    
    meets_standard = Ce_final <= target
    n_stages = 2 if (C_in > 200 and not meets_standard) else 1
    
    return {
        'AC_dose_g_per_L': AC_dose_design,
        'daily_AC_kg': daily_AC_kg,
        'monthly_AC_kg': monthly_AC_kg,
        'yearly_AC_kg': yearly_AC_kg,
        'yearly_AC_ton': yearly_AC_kg / 1000,
        'Ce_final': Ce_final,
        'efficiency': eff_final,
        'qe': qe_final,
        'qe_design': qe_design,
        'meets_standard': meets_standard,
        'target_standard': target,
        'n_stages': n_stages,
        'daily_Cr_load_kg': daily_Cr_g / 1000,
        'pH_factor': pH_correction(pH),
    }

# ============================================
# Streamlit GUI
# ============================================

st.set_page_config(
    page_title="활성탄 산정 시뮬레이터",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 도금 폐수 Cr(VI) 처리용 활성탄 산정 시뮬레이터")
st.markdown("**AC-O3-Al 흡착제 기반 사업장 맞춤형 의사결정 도구 v2.0**")
st.markdown("---")

with st.sidebar:
    st.header("📋 사업장 정보 입력")
    
    st.subheader("폐수 특성")
    C_in = st.slider(
        "Cr(VI) 평균 농도 (mg/L)", 
        min_value=1, max_value=500, value=100, step=5,
        help="도금 폐수 일반: 50~200 mg/L"
    )
    flow = st.number_input(
        "일일 폐수 유량 (m³/day)", 
        min_value=0.1, max_value=1000.0, value=10.0, step=1.0,
        help="소형: 1~10, 중형: 10~50, 대형: 50~500"
    )
    pH = st.slider(
        "폐수 pH", 
        min_value=1.0, max_value=14.0, value=3.0, step=0.1,
        help="도금 폐수: 2~4 (산성), 최적 pH: 3"
    )
    
    st.subheader("처리 조건")
    region = st.selectbox(
        "사업장 지역", 
        ['청정지역', '가지역', '나지역', '특례지역'], 
        index=1,
        help="청정: 0.1, 가/나: 0.5, 특례: 2.0 mg/L"
    )
    safety_factor = st.slider(
        "안전 인자", 
        min_value=1.0, max_value=5.0, value=2.0, step=0.1,
        help="설계 안전 여유. 표준 2.0"
    )
    model = st.radio("흡착 모델", ['Langmuir', 'Freundlich'])
    
    st.markdown("---")
    calc_button = st.button("🔍 활성탄 산정 계산", type="primary", use_container_width=True)

if calc_button or 'result' in st.session_state:
    if calc_button:
        result = calculate_AC_requirement(C_in, flow, pH, region, safety_factor, model)
        st.session_state['result'] = result
        st.session_state['inputs'] = {
            'C_in': C_in, 'flow': flow, 'pH': pH, 
            'region': region, 'model': model, 'safety_factor': safety_factor
        }
    else:
        result = st.session_state['result']
        inputs = st.session_state['inputs']
        C_in = inputs['C_in']
        pH = inputs['pH']
        model = inputs['model']
        region = inputs['region']
    
    # 상단 — 핵심 결과
    st.header("📊 활성탄 산정 결과")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("일일 활성탄", f"{result['daily_AC_kg']:.2f} kg")
    col2.metric("월간 활성탄", f"{result['monthly_AC_kg']:.1f} kg")
    col3.metric("연간 활성탄", f"{result['yearly_AC_ton']:.2f} 톤")
    col4.metric("일일 Cr 부하", f"{result['daily_Cr_load_kg']:.2f} kg")
    
    st.markdown("---")
    
    # 처리 성능
    st.header("🎯 처리 성능")
    col1, col2, col3 = st.columns(3)
    col1.metric("유입 농도", f"{C_in} mg/L")
    col2.metric("처리 후 농도", f"{result['Ce_final']:.3f} mg/L",
                delta=f"-{C_in - result['Ce_final']:.2f}")
    col3.metric("제거 효율", f"{result['efficiency']:.2f}%")
    
    st.subheader(f"📋 배출 기준 평가 — {region}")
    st.write(f"적용 기준: **{result['target_standard']} mg/L**")
    
    if result['meets_standard']:
        st.success(
            f"✅ 배출 기준 만족 (예상 {result['Ce_final']:.3f} mg/L < 기준 {result['target_standard']} mg/L)"
        )
    else:
        st.error(f"❌ 배출 기준 미달")
        st.warning("⚠️ 안전 인자 증가 또는 다단 처리 권장")
    
    st.markdown("---")
    
    # 운영 권장 사항
    st.header("🛠️ 운영 권장 사항")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("처리 시스템 구성")
        n_columns = max(1, int(np.ceil(result['monthly_AC_kg'] / 100)))
        st.info(f"""
        - **처리 단수**: {result['n_stages']}단
        - **권장 컬럼 규모**: 직경 50cm × 높이 1.5m
        - **컬럼당 충진량**: 약 100 kg
        - **필요 컬럼 수**: {n_columns}개 (월 교체 기준)
        - **평형 시간**: {PARAMS['t_eq_hr']} 시간
        - **설계 qe**: {result['qe_design']:.2f} mg/g
        """)
    
    with col2:
        st.subheader("운영 정보 (비용 분석)")
        AC_price = 5000
        daily_cost = result['daily_AC_kg'] * AC_price
        monthly_cost = result['monthly_AC_kg'] * AC_price
        yearly_cost = result['yearly_AC_kg'] * AC_price
        st.info(f"""
        - **일일 비용**: ₩{daily_cost:,.0f}
        - **월간 비용**: ₩{monthly_cost:,.0f}
        - **연간 비용**: ₩{yearly_cost:,.0f}
        - **활성탄 단가**: ₩5,000/kg
        - **폐기물 처리비**: 별도 산정 필요
        """)
    
    st.markdown("---")
    
    # 그래프 시각화
    st.header("📈 데이터 시각화")
    tab1, tab2, tab3 = st.tabs(["등온선 모델", "동역학 모델", "농도별 효율"])
    
    with tab1:
        st.subheader("Cr(VI) 흡착 등온선 (AC-O3-Al)")
        fig, ax = plt.subplots(figsize=(10, 6))
        Ce_range = np.linspace(0.1, 100, 200)
        ax.plot(Ce_range, langmuir(Ce_range), '-', 
                label=f'Langmuir (qmax={PARAMS["qmax"]:.2f} mg/g, R²=0.985)',
                linewidth=2, color='red')
        ax.plot(Ce_range, freundlich(Ce_range), '--', 
                label=f'Freundlich (n={PARAMS["n_freundlich"]:.2f}, R²=0.989)',
                linewidth=2, color='blue')
        ax.axhline(y=result['qe'], color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=result['Ce_final'], color='gray', linestyle=':', alpha=0.5)
        ax.plot(result['Ce_final'], result['qe'], 'ko', markersize=12, 
                label=f'운전점 (Ce={result["Ce_final"]:.2f}, qe={result["qe"]:.2f})')
        ax.set_xlabel('Ce (mg/L)', fontsize=12)
        ax.set_ylabel('qe (mg-Cr/g-AC)', fontsize=12)
        ax.set_title('AC-O3-Al의 Cr(VI) 흡착 등온선', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Pseudo-Second Order 동역학 모델")
        fig, ax = plt.subplots(figsize=(10, 6))
        t_range = np.linspace(0, 720, 200)
        qe_kin = PARAMS['qe_kinetic']
        k2 = PARAMS['k2']
        qt = (k2 * qe_kin**2 * t_range) / (1 + k2 * qe_kin * t_range)
        ax.plot(t_range/60, qt, '-', linewidth=2, color='green',
                label=f'PSO Model (qe={qe_kin:.2f}, k2={k2:.4f})')
        ax.axhline(y=qe_kin, color='red', linestyle='--', alpha=0.5, 
                   label=f'qe = {qe_kin:.2f} mg/g')
        ax.axvline(x=PARAMS['t_eq_hr'], color='gray', linestyle=':', alpha=0.5,
                   label=f'평형 시간 = {PARAMS["t_eq_hr"]} hr')
        ax.set_xlabel('시간 (hr)', fontsize=12)
        ax.set_ylabel('qt (mg/g)', fontsize=12)
        ax.set_title('Cr(VI) 흡착 동역학 (200 ppm, AC-O3-Al)', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.info("📌 **PSO 모델 R² = 1.0000** — 화학 흡착 메커니즘 입증")
    
    with tab3:
        st.subheader("Cr(VI) 농도별 처리 효율 예측")
        fig, ax = plt.subplots(figsize=(10, 6))
        C_range = np.linspace(10, 500, 50)
        eff_range = []
        Ce_range_calc = []
        dose = result['AC_dose_g_per_L'] * PARAMS['column_efficiency']
        for c in C_range:
            ce, eff, _ = calculate_treatment(c, dose, pH, model)
            eff_range.append(eff)
            Ce_range_calc.append(ce)
        ax2 = ax.twinx()
        line1 = ax.plot(C_range, eff_range, 'b-', linewidth=2, label='제거 효율 (%)')
        line2 = ax2.plot(C_range, Ce_range_calc, 'r-', linewidth=2, label='처리 후 농도 (mg/L)')
        ax2.axhline(y=result['target_standard'], color='orange', linestyle='--', 
                    alpha=0.7, label=f'배출 기준 ({result["target_standard"]} mg/L)')
        ax.axvline(x=C_in, color='gray', linestyle=':', alpha=0.5, 
                   label=f'현재 운전 ({C_in} mg/L)')
        ax.set_xlabel('유입 Cr(VI) 농도 (mg/L)', fontsize=12)
        ax.set_ylabel('제거 효율 (%)', color='blue', fontsize=12)
        ax2.set_ylabel('처리 후 농도 (mg/L)', color='red', fontsize=12)
        ax.set_title(f'농도별 처리 성능 (흡착제 {dose:.1f} g/L)', fontsize=13)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    with st.expander("📚 모델 정보 및 가정"):
        st.markdown(f"""
        ### 실험 기반 파라미터
        
        **흡착 등온선** (R² > 0.98):
        - Langmuir qmax = {PARAMS['qmax']:.4f} mg/g
        - Langmuir KL = {PARAMS['KL']:.4f} L/mg
        - Freundlich KF = {PARAMS['KF']:.4f}
        - Freundlich n = {PARAMS['n_freundlich']:.4f}
        
        **동역학** (PSO R² = 1.0):
        - k₂ = {PARAMS['k2']:.6f} g/(mg·min)
        - 평형 시간: {PARAMS['t_eq_hr']} 시간
        
        ### 모델 가정
        1. 25°C 상온 운전
        2. pH 3 최적 조건 (가우시안 보정)
        3. 단일 오염물 (Cr(VI))
        4. 회분식 흡착 데이터 기반
        5. 컬럼 효율 = 회분식의 {PARAMS['column_efficiency']:.1f}배 (선행 연구)
        6. 안전 인자 적용 (설계 qe = 평형 qe / 안전인자)
        
        ### 한계 사항
        - 컬럼 흡착 실측 데이터 부족 (향후 연구 필요)
        - 다중 오염물질 경쟁 흡착 미반영
        - 온도 영향 미반영
        - Al 담지량 매우 낮음 (0.03 wt%, 향후 최적화 필요)
        - pH 영향 가우시안 가정 (실험적 검증 필요)
        
        ### 데이터 출처
        - 실험: AC-O3-Al 흡착제 (오존 산화 + 0.01M AlCl₃ 개질)
        - 측정: UV-Vis (DPC법, 540nm), ICP-OES
        - 작성: 김남윤 (충남대 환경공학과, 2026)
        """)

else:
    st.info("""
    👈 **사이드바에서 사업장 정보를 입력하고 '활성탄 산정 계산' 버튼을 클릭하세요.**
    
    ### 사용 방법
    1. **폐수 특성** 입력: Cr(VI) 농도, 유량, pH
    2. **처리 조건** 선택: 지역, 안전 인자, 흡착 모델
    3. **계산 버튼** 클릭
    4. 결과 확인: 활성탄 필요량, 처리 효율, 비용
    
    ### 시뮬레이터 특징
    - ✅ 실제 실험 데이터 기반 (AC-O3-Al 흡착제)
    - ✅ Langmuir/Freundlich 모델 적용 (R² > 0.98)
    - ✅ PSO 동역학 (R² = 1.0)
    - ✅ 한국 환경부 배출 기준 자동 평가
    - ✅ 비용 분석 포함
    - ✅ 컬럼 운전 효율 반영
    """)
    
    st.subheader("📋 가상 도금 공장 시나리오 예시")
    scenarios = pd.DataFrame({
        '사업장': ['소형 도금', '중형 도금', '대형 표면처리', '시화 도금단지', '인천 남동공단'],
        'Cr(VI) 농도 (mg/L)': [50, 100, 200, 80, 120],
        '유량 (m³/day)': [5, 20, 100, 50, 30],
        'pH': [3.0, 3.5, 2.5, 3.0, 3.5],
        '지역': ['일반', '특례', '일반', '특례', '일반'],
    })
    st.dataframe(scenarios, use_container_width=True)
    
    st.markdown("""
    ### 졸업논문 정보
    - **연구**: AC-O3-Al 흡착제를 이용한 Cr(VI) 흡착 연구
    - **작성**: 김남윤 (충남대 환경공학과)
    - **연도**: 2026
    """)

st.markdown("---")
st.caption("🧪 도금 폐수 Cr(VI) 처리용 활성탄 산정 시뮬레이터 v2.0 | 충남대학교 환경공학과 김남윤")
