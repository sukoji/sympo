"""
symPO: 계층적 WBS 자동 생성 시스템
Streamlit 메인 UI
"""
import sys
import os
import time
import json
import traceback
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ─── eDISC 프로파일 캐시 (Streamlit 재실행 시 재파싱 방지) ────
@st.cache_resource
def _load_disc_profiles_cached():
    """앱 기동 시 1회만 파싱, 이후 캐시 반환"""
    try:
        from data_pipeline.disc_parser import load_all_disc_profiles
        _dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
        return load_all_disc_profiles(_dir)
    except Exception:
        return {}

# ─── AGENT STYLES & RENDERING ──────────────────────
AGENT_STYLES = {
    "슈퍼바이저(PM)": ("🎯", "bubble-supervisor", "#667eea"),
    "플래너": ("📅", "bubble-planner", "#4fd1c5"),
    "프론트엔드 개발자": ("💻", "bubble-frontend", "#f6ad55"),
    "백엔드 개발자": ("⚙️", "bubble-backend", "#b794f4"),
    "디자이너": ("🎨", "bubble-designer", "#f093fb"),
    "QA 엔지니어": ("🔍", "bubble-qa", "#fc8181"),
    "시스템": ("🤖", "bubble-system", "#48bb78"),
    "WBS Gen Agent": ("🤖", "bubble-system", "#059669"),
}

def render_debate_message(m):
    import html as _html
    import re as _re
    role_name = str(getattr(m.agent_role, 'value', m.agent_role))
    style_info = AGENT_STYLES.get(role_name, ("👤", "bubble-default", "#64748b"))
    icon, _, color = style_info

    _clean = _re.sub(r'\bNEW_TASK:\s*\{.*?\}', '', m.message, flags=_re.DOTALL).strip()
    safe_msg = _html.escape(_clean).replace('\n', '<br>')
    safe_name = _html.escape(m.agent_name)
    msg_type = getattr(m, 'message_type', '') or ''

    # PASS 메시지
    if msg_type == 'pass':
        st.markdown(f"""
        <div class="timeline-item" style="margin-bottom:0.3rem;opacity:0.7;">
            <div class="agent-avatar" style="width:28px;height:28px;font-size:0.78rem;
                background:rgba(248,250,252,0.8);color:#94a3b8;border:1px solid #e2e8f0;flex-shrink:0;">{icon}</div>
            <div style="display:flex;align-items:center;gap:8px;background:rgba(248,250,252,0.6);
                border:1px solid #f1f5f9;border-radius:6px;padding:0.35rem 0.75rem;flex-grow:1;">
                <span style="background:#e2e8f0;color:#64748b;padding:1px 6px;border-radius:4px;
                    font-weight:700;font-size:0.7rem;letter-spacing:0.05em;flex-shrink:0;">PASS</span>
                <span style="font-size:0.8rem;color:#94a3b8;"><strong style="color:#64748b;">{safe_name}</strong>: {safe_msg}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # mediation 메시지 (L2 구분선)
    if msg_type == 'mediation' and getattr(m, 'related_task_id', None):
        st.markdown(f"""
        <div style="margin:1rem 0 0.5rem 0;padding:8px 14px;
            background:linear-gradient(90deg,rgba(238,242,255,0.9),rgba(237,233,254,0.5),transparent);
            border-left:4px solid #6366f1;border-radius:0 8px 8px 0;
            font-size:0.82rem;color:#4338ca;font-weight:700;
            display:flex;align-items:center;gap:8px;letter-spacing:0.02em;">
            <span style="width:6px;height:6px;border-radius:50%;background:#6366f1;flex-shrink:0;"></span>
            {safe_msg}
        </div>
        """, unsafe_allow_html=True)
        return

    # proposal/comment 메시지 → 핵심 카드
    is_proposal = msg_type in ('proposal', 'decision')
    accent_bg = f"linear-gradient(135deg, {color}08 0%, {color}04 100%)"
    border_style = f"2px solid {color}30" if is_proposal else f"1px solid {color}18"

    st.markdown(f"""
    <div class="timeline-item">
        <div class="agent-avatar" style="
            border: 2px solid {color}35;
            background: linear-gradient(135deg, {color}12, {color}06);
            color: {color};
            box-shadow: 0 2px 8px {color}20;">
            {icon}
        </div>
        <div class="msg-content" style="
            border-left: {border_style};
            background: {accent_bg};
            border-color: {color}18;">
            <div class="msg-header">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span class="agent-label" style="color:{color};">{safe_name}</span>
                    {f'<span style="font-size:0.68rem;background:{color}15;color:{color};padding:1px 7px;border-radius:99px;font-weight:700;border:1px solid {color}25;">{"📋 제안" if msg_type=="proposal" else "✅ 결정"}</span>' if is_proposal else ''}
                </div>
                <span class="msg-time">{m.timestamp}</span>
            </div>
            <div class="msg-text" style="color:#1e293b;">{safe_msg}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def _extract_text_from_pdf(pdf_file) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return f"[PDF 추출 오류] {str(e)}"

def _render_member_profile_card(profile):
    """MemberProfile 객체를 미려한 카드 형태로 렌더링합니다."""
    skills = ", ".join(profile.primary_skills[:4])
    strengths = ", ".join(profile.strengths[:3])
    
    st.markdown(f"""
    <div class="member-profile-card">
        <div class="member-header">
            <span class="member-name">👤 {profile.name}</span>
        </div>
        <div class="member-stats">
            <span>📅 {profile.years_of_experience}년차</span>
            <span>⚡ 역량: {skills}</span>
        </div>
        <div class="member-skills-row">
            <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">핵심 강점</div>
            <span class="member-trait-tag">💪 {strengths}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def _clean_member_name(filename: str) -> str:
    """파일명에서 이름만 추출 (확장자, 각종 접두어 제거)"""
    import os
    import re
    # 1. 확장자 제거
    name = os.path.splitext(filename)[0]
    # 2. 공통 접두어/패턴 제거 (이력서_, resume_, CV_ 등)
    name = re.sub(r'^(이력서|resume|cv|portfolio|포트폴리오|eDISC)[_ \-]*', '', name, flags=re.IGNORECASE)
    # 3. 괄호 및 내부 텍스트 제거 (예: (최종), (수정본) 등)
    name = re.sub(r'\s*[\(\[].*?[\)\]]', '', name)
    # 4. 언더바, 대시 등 공백화 후 트리밍
    name = name.replace('_', ' ').replace('-', ' ').strip()
    return name if name else filename

# ─── Streamlit 페이지 설정 ────────────────────────────
st.set_page_config(
    page_title="symPO | WBS 자동 생성",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 🎨 Premium UI CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── CSS 변수 ── */
    :root {
        --primary: #4f46e5;
        --primary-light: #818cf8;
        --primary-dark: #3730a3;
        --secondary: #7c3aed;
        --accent: #06b6d4;
        --success: #059669;
        --warning: #d97706;
        --danger: #dc2626;
        --surface: #ffffff;
        --surface-2: #f8fafc;
        --surface-3: #f1f5f9;
        --border: #e2e8f0;
        --border-light: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.1), 0 4px 12px rgba(0,0,0,0.05);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
    }

    /* ── Global Reset ── */
    .stApp {
        background: linear-gradient(160deg, #f0f2ff 0%, #f5f0ff 40%, #f0f7ff 100%);
        font-family: 'Inter', 'Pretendard', -apple-system, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── 메인 헤더 ── */
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 50%, #0891b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        margin: 0 0 0.25rem 0;
        line-height: 1;
        filter: drop-shadow(0 2px 8px rgba(79,70,229,0.2));
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0 0 1.5rem 0;
        letter-spacing: 0.01em;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        border-right: none !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.15) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #a5b4fc !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #a5b4fc !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, rgba(99,102,241,0.5), rgba(124,58,237,0.5)) !important;
        border: 1px solid rgba(165,180,252,0.3) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] .stSuccess {
        background: rgba(52, 211, 153, 0.12) !important;
        border: 1px solid rgba(52,211,153,0.25) !important;
        border-radius: 10px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(8px);
        padding: 6px;
        border-radius: 16px;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(226,232,240,0.8);
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 10px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 0 18px;
        border: none !important;
        letter-spacing: 0.01em;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.35);
    }

    /* ── Cards ── */
    .glass-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(12px);
        border-radius: var(--radius-lg);
        border: 1px solid rgba(226,232,240,0.6);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    .info-card {
        background: linear-gradient(135deg, rgba(238,242,255,0.9) 0%, rgba(240,249,255,0.9) 100%);
        border-radius: var(--radius-md);
        border: 1px solid #c7d2fe;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* ── Section Headers ── */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.65rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #4f46e5, #7c3aed, transparent) 1;
    }

    /* ── Timeline / Chat UI ── */
    .debate-timeline {
        display: flex;
        flex-direction: column;
        gap: 0.875rem;
        padding: 0.5rem 0;
    }
    .timeline-item {
        display: flex;
        gap: 0.875rem;
        animation: fadeSlideInUp 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    }
    @keyframes fadeSlideInUp {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes agentPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.5), 0 2px 8px rgba(0,0,0,0.1); }
        50% { box-shadow: 0 0 0 8px rgba(79, 70, 229, 0), 0 4px 12px rgba(79,70,229,0.2); }
    }
    @keyframes shimmerLoad {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    @keyframes pulseGlow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(0.97); }
    }
    @keyframes spinOrbit {
        0% { transform: rotate(0deg) translateX(14px) rotate(0deg); }
        100% { transform: rotate(360deg) translateX(14px) rotate(-360deg); }
    }
    .agent-avatar {
        flex-shrink: 0;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        background: white;
        box-shadow: var(--shadow-sm);
        position: relative;
        transition: all 0.3s ease;
    }
    .msg-content {
        flex-grow: 1;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(226,232,240,0.8);
        border-radius: 4px 14px 14px 14px;
        padding: 0.875rem 1.1rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
        backdrop-filter: blur(4px);
    }
    .msg-content:hover {
        box-shadow: var(--shadow-md);
    }
    .msg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .agent-label {
        font-weight: 700;
        font-size: 0.875rem;
        letter-spacing: 0.01em;
    }
    .msg-time {
        font-size: 0.72rem;
        color: #94a3b8;
        background: #f8fafc;
        padding: 2px 8px;
        border-radius: 99px;
        font-variant-numeric: tabular-nums;
        border: 1px solid #f1f5f9;
    }
    .msg-text {
        color: #374151;
        font-size: 0.9rem;
        line-height: 1.7;
        white-space: pre-wrap;
        letter-spacing: 0.01em;
    }
    .typing-indicator {
        padding: 0.7rem 1.1rem;
        background: linear-gradient(90deg, #f8fafc, #f0f4ff, #f8fafc);
        background-size: 200% 100%;
        border-radius: 12px;
        color: #6366f1;
        font-size: 0.875rem;
        font-weight: 500;
        border: 1px solid #e0e7ff;
        animation: shimmerLoad 2s linear infinite;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Metrics ── */
    .metric-card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(226,232,240,0.7);
        box-shadow: var(--shadow-sm);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #06b6d4);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(99,102,241,0.2);
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 900;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── 버튼 ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border-radius: var(--radius-md) !important;
        padding: 0.65rem 1.5rem !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4), 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover {
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.55), 0 4px 8px rgba(0,0,0,0.1) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4) !important;
    }

    /* ── 입력 필드 ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.9) !important;
        color: #0f172a !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.12) !important;
        background: white !important;
    }

    /* ── 섹션 구분선 ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(124,58,237,0.3), transparent);
        margin: 1.75rem 0;
    }

    /* ── 빈 상태 ── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-xl);
        border: 2px dashed #c7d2fe;
        margin: 1rem 0;
    }
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: pulseGlow 3s ease-in-out infinite;
    }
    .empty-state h3 {
        color: #374151;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .empty-state p {
        color: #94a3b8;
        font-size: 0.875rem;
    }

    /* ── Phase 배지 ── */
    .phase-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .phase-1 { background: linear-gradient(135deg, #fef9c3, #fef3c7); color: #92400e; border: 1px solid #fde68a; }
    .phase-2 { background: linear-gradient(135deg, #ede9fe, #e0e7ff); color: #5b21b6; border: 1px solid #ddd6fe; }
    .phase-3 { background: linear-gradient(135deg, #d1fae5, #dcfce7); color: #065f46; border: 1px solid #6ee7b7; }

    /* ── 진행 상태 카드 ── */
    .progress-card {
        background: linear-gradient(135deg, rgba(238,242,255,0.95) 0%, rgba(250,245,255,0.95) 100%);
        border: 1px solid rgba(167,139,250,0.3);
        border-left: 4px solid #6366f1;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(4px);
        animation: fadeSlideInUp 0.3s ease;
    }
    .progress-agent {
        font-weight: 700;
        font-size: 0.95rem;
        color: #4338ca;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .progress-detail {
        color: #6b7280;
        font-size: 0.82rem;
        margin-top: 0.3rem;
        line-height: 1.5;
    }

    /* ── Control Room ── */
    .control-room {
        background: linear-gradient(135deg, rgba(238,242,255,0.9) 0%, rgba(224,242,254,0.9) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-left: 4px solid #6366f1;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(4px);
        box-shadow: var(--shadow-sm);
    }
    .control-room-title {
        font-weight: 700;
        color: #3730a3;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .control-room-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #e0e7ff, #ede9fe);
        color: #3730a3;
        padding: 3px 12px;
        border-radius: 99px;
        font-size: 0.77rem;
        font-weight: 700;
        margin-right: 5px;
        margin-bottom: 3px;
        border: 1px solid rgba(99,102,241,0.2);
        box-shadow: 0 1px 3px rgba(99,102,241,0.1);
    }

    /* ── Member Profile Cards ── */
    .member-profile-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(226,232,240,0.8);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    .member-profile-card:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(99,102,241,0.2);
    }
    .member-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .member-name {
        font-weight: 700;
        font-size: 1rem;
        color: #0f172a;
    }
    .member-role-badge {
        background: linear-gradient(135deg, #eef2ff, #ede9fe);
        color: #4338ca;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 700;
        border: 1px solid #e0e7ff;
    }
    .member-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 0.6rem;
    }
    .member-trait-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #e2e8f0;
    }
    .member-skills-row {
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px dashed #e2e8f0;
    }

    /* ── Agent Call Tree 노드 스타일 ── */
    @keyframes nodeActiveRing {
        0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.6), 0 4px 12px rgba(0,0,0,0.1); }
        70% { box-shadow: 0 0 0 10px rgba(99,102,241,0), 0 4px 12px rgba(99,102,241,0.2); }
        100% { box-shadow: 0 0 0 0 rgba(99,102,241,0), 0 4px 12px rgba(0,0,0,0.1); }
    }
    @keyframes flowDot {
        0% { left: 0%; opacity: 1; transform: translateX(0); }
        100% { left: 100%; opacity: 0; transform: translateX(-4px); }
    }
    @keyframes successPop {
        0% { transform: scale(0.8); opacity: 0; }
        70% { transform: scale(1.1); }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


def _render_agent_call_tree(state: dict):
    """
    에이전트 호출 트리를 인터랙티브 HTML/CSS 애니메이션으로 렌더링합니다.
    각 노드는 idle / active / done 상태에 따라 동적 스타일을 갖습니다.
    """
    import streamlit.components.v1 as _comp
    import html as _h

    # ── 상태 추출 ───────────────────────────────────────────
    current_wbs = bool(state.get("current_wbs_draft"))
    called_agents = state.get("called_agents") or []
    final_wbs = bool(state.get("final_wbs"))
    acting = str(state.get("_current_agent_acting") or "")
    debate_log = state.get("debate_log") or []
    current_round = int(state.get("current_round") or 0)
    consensus = bool(state.get("consensus_reached"))

    # 각 에이전트 역할이 토론 로그에 등장했는지 확인
    spoken_roles = set()
    spoken_names = set()
    for msg in debate_log:
        r = str(getattr(msg, 'agent_role', '') or '')
        n = str(getattr(msg, 'agent_name', '') or '')
        spoken_roles.add(r); spoken_names.add(n)

    def _spoke(*keys):
        return any(k in spoken_roles or k in spoken_names for k in keys)

    wbs_done     = current_wbs
    task_done    = bool(called_agents) and current_wbs
    planner_done = _spoke("플래너", "Planner", "planner")
    fe_done      = _spoke("프론트엔드 개발자", "Frontend", "frontend")
    be_done      = _spoke("백엔드 개발자", "Backend", "backend")
    des_done     = _spoke("디자이너", "Designer", "designer")
    qa_done      = _spoke("QA 엔지니어", "QA", "qa")
    any_sub_done = any([planner_done, fe_done, be_done, des_done, qa_done])
    mediate_done = any_sub_done and current_round > 0
    final_done   = final_wbs or consensus

    # 노드 상태 결정
    def _st(is_done, agent_key=""):
        if acting and agent_key and agent_key in acting:
            return "active"
        return "done" if is_done else "idle"

    node_input    = "done"   # 항상 완료 (입력은 이미 있음)
    node_wbs      = _st(wbs_done, "wbs_gen")
    node_task     = _st(task_done, "supervisor_task_match")
    node_planner  = _st(planner_done, "planner")
    node_fe       = _st(fe_done, "frontend")
    node_be       = _st(be_done, "backend")
    node_des      = _st(des_done, "designer")
    node_qa       = _st(qa_done, "qa")
    node_mediate  = _st(mediate_done, "supervisor_mediat")
    node_final    = _st(final_done, "supervisor_finaliz")

    # ── CSS ─────────────────────────────────────────────────
    CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: transparent; }

    .tree-wrap {
        padding: 12px 8px 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0;
        min-width: 480px;
    }
    .tree-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        width: 100%;
    }
    .tree-row.sub { gap: 6px; }

    /* ── 노드 공통 ── */
    .node {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        padding: 8px 14px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.01em;
        border: 1.5px solid transparent;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative;
        text-align: center;
        min-width: 80px;
    }
    .node-icon { font-size: 18px; margin-bottom: 4px; }
    .node-label { line-height: 1.3; }
    .node-phase {
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 3px;
        opacity: 0.7;
    }

    /* ── idle ── */
    .idle {
        background: #f8fafc;
        border-color: #e2e8f0;
        color: #94a3b8;
        opacity: 0.65;
    }
    .idle .node-icon { filter: grayscale(1); }

    /* ── done ── */
    .done {
        background: linear-gradient(135deg, #d1fae5, #ecfdf5);
        border-color: #6ee7b7;
        color: #065f46;
    }
    .done::after {
        content: '✓';
        position: absolute;
        top: -6px; right: -6px;
        width: 16px; height: 16px;
        background: #059669;
        color: white;
        border-radius: 50%;
        font-size: 9px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800;
        box-shadow: 0 1px 4px rgba(5,150,105,0.4);
        animation: successPop 0.35s ease;
    }
    @keyframes successPop {
        0% { transform: scale(0); opacity: 0; }
        70% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }

    /* ── active ── */
    .active {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-color: #6366f1;
        color: #3730a3;
        animation: nodeRing 1.5s ease-in-out infinite;
        box-shadow: 0 0 0 0 rgba(99,102,241,0.5);
    }
    @keyframes nodeRing {
        0% { box-shadow: 0 0 0 0 rgba(99,102,241,0.5), 0 4px 12px rgba(99,102,241,0.15); }
        60% { box-shadow: 0 0 0 10px rgba(99,102,241,0), 0 4px 12px rgba(99,102,241,0.15); }
        100% { box-shadow: 0 0 0 0 rgba(99,102,241,0), 0 4px 12px rgba(99,102,241,0.15); }
    }
    .active .node-icon { animation: iconBounce 1s ease-in-out infinite; }
    @keyframes iconBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }

    /* ── 입력/출력 특수 노드 ── */
    .node-input {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border-color: #fcd34d;
        color: #92400e;
        border-radius: 10px;
        width: 180px;
    }
    .node-output {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-color: #86efac;
        color: #166534;
        border-radius: 10px;
        width: 200px;
    }
    .node-output.done { background: linear-gradient(135deg, #d1fae5, #bbf7d0); }

    /* ── 커넥터 ── */
    .connector {
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        height: 28px;
        width: 100%;
    }
    .connector-line {
        width: 2px;
        height: 28px;
        background: linear-gradient(180deg, #c7d2fe, #e0e7ff);
        position: relative;
        overflow: hidden;
        border-radius: 1px;
    }
    .connector-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #6366f1;
        position: absolute;
        left: -2px;
        animation: flowDown 1.2s linear infinite;
    }
    @keyframes flowDown {
        0% { top: -8px; opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { top: 28px; opacity: 0; }
    }

    /* ── 팬아웃 ── */
    .fan-area {
        width: 100%;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .fan-line {
        width: 70%;
        height: 24px;
        border: 2px solid #c7d2fe;
        border-bottom: none;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
        background: transparent;
    }
    .fan-line-in {
        width: 70%;
        height: 24px;
        border: 2px solid #c7d2fe;
        border-top: none;
        border-radius: 0 0 12px 12px;
        margin-top: 0;
    }

    /* ── 라운드 배지 ── */
    .round-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 3px 12px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 700;
        margin: 4px 0;
        box-shadow: 0 2px 6px rgba(99,102,241,0.3);
    }
    .sub-node { min-width: 72px; font-size: 10.5px; padding: 7px 8px; }
    </style>
    """

    # ── 노드 HTML 생성기 ───────────────────────────────────
    def node(icon, label, status, phase="", extra_class="", extra_style=""):
        phase_html = f'<div class="node-phase">{phase}</div>' if phase else ""
        return (
            f'<div class="node {status} {extra_class}" style="{extra_style}">'
            f'<div class="node-icon">{icon}</div>'
            f'<div class="node-label">{label}</div>'
            f'{phase_html}'
            f'</div>'
        )

    def connector(active=True):
        dot = '<div class="connector-dot"></div>' if active else ""
        return f'<div class="connector"><div class="connector-line">{dot}</div></div>'

    flow_active = wbs_done  # 데이터 흐름 애니메이션 표시 여부

    sub_nodes_html = (
        node("📅", "플래너", node_planner, extra_class="sub-node") +
        node("💻", "프론트엔드", node_fe, extra_class="sub-node") +
        node("⚙️", "백엔드", node_be, extra_class="sub-node") +
        node("🎨", "디자이너", node_des, extra_class="sub-node") +
        node("🔍", "QA", node_qa, extra_class="sub-node")
    )

    round_badge = (
        f'<div style="display:flex;justify-content:center;margin:4px 0;">'
        f'<div class="round-badge">🔁 Round {current_round} / {"합의 완료" if consensus else "진행 중"}</div>'
        f'</div>'
    ) if current_round > 0 else ""

    HTML = f"""
    {CSS}
    <div class="tree-wrap">
        <!-- 입력 -->
        <div class="tree-row">
            {node("📄", "PRD + 팀 프로필", "done", extra_class="node-input")}
        </div>
        {connector(True)}

        <!-- Phase 1: WBS Gen -->
        <div class="tree-row">
            {node("🤖", "WBS Gen Agent", node_wbs, phase="Phase 1")}
        </div>
        {connector(wbs_done)}

        <!-- Phase 2: Task Manager -->
        <div class="tree-row">
            {node("🎯", "Task Manager", node_task, phase="Phase 2")}
        </div>

        <!-- 팬아웃 -->
        <div class="fan-area">
            <div class="fan-line"></div>
        </div>

        <!-- Phase 3: Sub-agents -->
        {round_badge}
        <div class="tree-row sub">
            {sub_nodes_html}
        </div>

        <!-- 팬인 -->
        <div class="fan-area">
            <div class="fan-line-in"></div>
        </div>
        {connector(any_sub_done)}

        <!-- Supervisor 중재 -->
        <div class="tree-row">
            {node("🎯", "Supervisor 중재", node_mediate, phase="Phase 3")}
        </div>
        {connector(mediate_done)}

        <!-- Final -->
        <div class="tree-row">
            {node("✅", "최종 WBS 확정", node_final, extra_class="node-output")}
        </div>
    </div>
    """

    _comp.html(HTML, height=540, scrolling=False)


def main():
    # ─── 헤더 ─────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:flex-end;gap:16px;margin-bottom:0.25rem;">
        <h1 class="main-header">🎯 symPO</h1>
        <div style="margin-bottom:10px;display:flex;gap:6px;align-items:center;">
            <span style="background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;
                padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700;
                letter-spacing:0.05em;">v2.0</span>
            <span style="background:rgba(16,185,129,0.1);color:#059669;border:1px solid #6ee7b7;
                padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700;">MULTI-AGENT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">다중 에이전트 오케스트레이션 · 계층적 WBS 자동 생성 · LangGraph 기반</p>',
        unsafe_allow_html=True,
    )

    # ─── 사이드바 ─────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ 시스템 설정")

        # ENV 기반 기본 선택 인덱스 결정
        current_backend_env = os.getenv("LLM_BACKEND", "mock").lower()
        backend_options = [
            "gemini (Google)",
            "gemma4 (Local)",
            "gemma4-api (Colab)",
            "llama4 (Local)",
            "mock (API 불필요)",
            "openai (GPT-4o)",
            "anthropic (Claude)",
            "ollama (로컬)",
        ]
        # 기본: mock
        default_index = 4
        if current_backend_env in ("gemma4-api", "gemma4_api"): default_index = 2
        elif "gemini" in current_backend_env: default_index = 0
        elif "gemma" in current_backend_env: default_index = 1
        elif "llama" in current_backend_env: default_index = 3
        elif "openai" in current_backend_env: default_index = 5
        elif "anthropic" in current_backend_env: default_index = 6
        elif "ollama" in current_backend_env: default_index = 7

        llm_mode = st.selectbox(
            "LLM 백엔드",
            backend_options,
            index=default_index,
            help="gemini는 Cloud API, gemma4/llama4는 로컬 GPU, gemma4-api는 Colab ngrok HTTP 서버. API 키 없이 테스트하려면 mock."
        )

        backend_map = {
            "gemini (Google)": "gemini",
            "gemma4 (Local)": "gemma4",
            "gemma4-api (Colab)": "gemma4-api",
            "llama4 (Local)": "llama4",
            "mock (API 불필요)": "mock",
            "openai (GPT-4o)": "openai",
            "anthropic (Claude)": "anthropic",
            "ollama (로컬)": "ollama",
        }
        os.environ["LLM_BACKEND"] = backend_map.get(llm_mode, "gemini")

        backend_name = os.environ['LLM_BACKEND'].upper()
        st.success(f"✓ 활성 모델: **{backend_name}**")

        with st.expander("🤖 에이전트별 모델 상세 설정"):
            st.markdown('<div style="font-size:0.75rem;color:#a5b4fc;margin-bottom:8px;">각 에이전트별 백본 모델을 개별 지정할 수 있습니다 (실험용).</div>', unsafe_allow_html=True)
            
            agent_model_config = {}
            # node_name: wbs_gen, supervisor, planner, frontend, backend, designer, qa
            
            m_opts = ["기본값"] + backend_options
            
            m_wbs = st.selectbox("WBS 생성 (Phase 1)", m_opts, key="m_wbs")
            m_sup = st.selectbox("슈퍼바이저 (PM)", m_opts, key="m_sup")
            m_pla = st.selectbox("플래너", m_opts, key="m_pla")
            m_fe = st.selectbox("프론트엔드 개발자", m_opts, key="m_fe")
            m_be = st.selectbox("백엔드 개발자", m_opts, key="m_be")
            m_des = st.selectbox("디자이너", m_opts, key="m_des")
            m_qa = st.selectbox("QA 엔지니어", m_opts, key="m_qa")

            def _get_val(v): return backend_map.get(v) if v != "기본값" else None
            
            if _get_val(m_wbs): agent_model_config["wbs_gen"] = _get_val(m_wbs)
            if _get_val(m_sup): agent_model_config["supervisor"] = _get_val(m_sup)
            if _get_val(m_pla): agent_model_config["planner"] = _get_val(m_pla)
            if _get_val(m_fe): agent_model_config["frontend"] = _get_val(m_fe)
            if _get_val(m_be): agent_model_config["backend"] = _get_val(m_be)
            if _get_val(m_des): agent_model_config["designer"] = _get_val(m_des)
            if _get_val(m_qa): agent_model_config["qa"] = _get_val(m_qa)
            
            st.session_state["agent_model_config"] = agent_model_config

        st.markdown("---")
        min_rounds = st.slider("최소 토론 라운드", 1, 3, 2)
        max_rounds = st.slider("최대 토론 라운드", min_rounds, 10, 5)

        st.markdown("---")
        st.markdown("### 🔄 에이전트 플로우")
        st.markdown("""
<div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
    border-radius:14px; padding:16px; font-size:11.5px; color:#e0e7ff !important; line-height:1.8;">
<div style="display:flex;flex-direction:column;gap:4px;">
  <div style="background:rgba(255,215,0,0.12);border:1px solid rgba(255,215,0,0.25);border-radius:8px;padding:6px 10px;">
    📄 <b style="color:#fde68a;">PRD + 팀 프로필 분석</b>
  </div>
  <div style="text-align:center;color:#818cf8;font-size:14px;">↓</div>
  <div style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:8px;padding:6px 10px;">
    🤖 <b style="color:#a5b4fc;">Phase 1: WBS 초안 생성</b><br>
    <span style="color:#818cf8;font-size:10.5px;">&nbsp;└ L1(7~9)→L2(5~7)→L3(5~8) 계층</span>
  </div>
  <div style="text-align:center;color:#818cf8;font-size:14px;">↓</div>
  <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:6px 10px;">
    🎯 <b style="color:#6ee7b7;">Phase 2: Task Manager</b><br>
    <span style="color:#818cf8;font-size:10.5px;">&nbsp;├ eDISC 기반 R&R 자동 배분<br>&nbsp;└ L2별 전문가 매칭</span>
  </div>
  <div style="text-align:center;color:#818cf8;font-size:14px;">↓</div>
  <div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:6px 10px;">
    🔁 <b style="color:#fde68a;">Phase 3: 집중 토론</b><br>
    <span style="color:#818cf8;font-size:10.5px;">&nbsp;├ 📅 플래너 / 💻 FE / ⚙️ BE<br>&nbsp;├ 🎨 디자이너 / 🔍 QA<br>&nbsp;└ 슈퍼바이저 실시간 중재</span>
  </div>
  <div style="text-align:center;color:#818cf8;font-size:14px;">↓</div>
  <div style="background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.3);border-radius:8px;padding:6px 10px;">
    ✅ <b style="color:#c4b5fd;">Phase 4: 최종 WBS 확정</b><br>
    <span style="color:#818cf8;font-size:10.5px;">&nbsp;└ 간트 + 평가지표 리포트</span>
  </div>
</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗂 샘플 데이터 로드", use_container_width=True):
            st.session_state["load_sample"] = True
            st.rerun()

        # eDISC 프로파일 상태 표시 (캐시 사용 — 재클릭 시 재파싱 없음)
        st.markdown("---")
        st.markdown("### 🧬 eDISC 프로파일")
        _sb_profiles = _load_disc_profiles_cached()
        if _sb_profiles:
            _type_colors = {"C": "🟣", "D": "🔴", "I": "🟡", "S": "🟢"}
            _rows = []
            for _n, _p in _sb_profiles.items():
                _icon = _type_colors.get(_p.type_code[:1] if _p.type_code else "?", "⚪")
                _rows.append(f"{_icon} <b>{_n}</b> — {_p.primary_type} | {_p.team_role}")
            st.markdown(
                "<div style='background:#f8f9fa;border-radius:8px;padding:10px 14px;"
                "font-size:11px;color:#1e293b;line-height:2;'>"
                + "<br>".join(_rows) + "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='color:#64748b;font-size:11px;'>sample_data/ 에 eDISC_*.pdf 없음</div>",
                unsafe_allow_html=True
            )

    # Session state 초기화
    if "selected_tab" not in st.session_state:
        st.session_state["selected_tab"] = 0
    if "wbs_output" not in st.session_state:
        st.session_state["wbs_output"] = None
    if "debate_log" not in st.session_state:
        st.session_state["debate_log"] = []
    if "team_members" not in st.session_state:
        st.session_state["team_members"] = []

    # ─── 메인 탭 ──────────────────────────────────────
    tabs = st.tabs([
        "📝 1. 데이터 입력",
        "🤖 2. 에이전트 토론",
        "📊 3. WBS 결과",
        "📖 4. 설명 가능성",
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 탭 1: 데이터 입력
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[0]:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-title">📋 PRD 입력</div>', unsafe_allow_html=True)

            prd_input_mode = st.radio(
                "입력 방식",
                ["📝 직접 입력", "📄 텍스트 파일 업로드", "🗂 개발 샘플", "🏪 P마켓 샘플"],
                horizontal=True,
            )

            # 샘플 로드 플래그 (사이드바 버튼)
            sample_loaded = st.session_state.get("load_sample", False)

            # ── 샘플 프리셋 정의 ──────────────────────────────
            _DEV_SAMPLE = dict(
                project_name="대규모 실시간 로그 분석 플랫폼 구축",
                project_goal="분산 처리 아키텍처 기반으로 일일 10TB 이상의 로그를 실시간 파이프라인으로 분석하여 인사이트 도출",
                target_users="Data Scientist, 데이터 분석가, 전략 기획팀",
                scope="포함: Kafka 기반 스트리밍, Spark 데이터파이프라인 구축, 시각화 대시보드 / 제외: 데이터 소스 자체 수집 모듈 구축",
                key_features="Kafka 스트리밍 파이프라인\nSpark 분산 병렬 연산 처리\nAirflow 데이터 카탈로그 및 스케줄링\nTableau/BI 연동 시각화 API",
                tech_stack="Apache Kafka\nApache Spark\nAirflow\nPython/FastAPI\nPostgreSQL\nDocker/K8s",
                deadline="2025-09-30",
                budget_weeks=16,
                team_size=4,
                constraints="로그 유실률 0.001% 이하 엄수\n데이터 조회 API 응답속도 100ms 이내\n데이터 보안 규제(GDPR/개인정보) 마스킹 처리 필수",
            )
            _PMARKET_SAMPLE = dict(
                project_name="P마켓 빅데이터 기반 매출 증대 전략",
                project_goal="2025년 판매·고객 데이터 빅데이터 분석으로 고객 세그먼트별 맞춤 프로모션을 실행하고, 모바일 앱·매장 진열 개선을 통해 전년 대비 매출 20% 이상 회복",
                target_users="수지지역 P마켓 방문 고객 (학부모, 노인·실버, 직장인, 학생)",
                scope="포함: 판매·고객 데이터 분석, 맞춤형 프로모션 기획·실행, 모바일 앱 기능 개선, 매장 진열 리디자인 / 제외: 신규 매장 개점, 공급업체 교체",
                key_features="소매업 트렌드·소비자 수요 조사 및 데이터 분석\n고객 특성(세그먼트)별 맞춤형 서비스·프로모션 설계\n모바일 알림·배송·커뮤니티 서비스 고도화\n매장 실내 진열 및 고객 동선 리디자인",
                tech_stack="Python/Pandas (데이터 분석)\nTableau / Power BI (시각화)\nMySQL (판매 데이터)\n모바일 앱 (Android/iOS)\nCRM 플랫폼",
                deadline="2025-12-31",
                budget_weeks=24,
                team_size=5,
                constraints="개인정보보호법 준수 (고객 데이터 익명화 필수)\n기존 POS·재고 시스템과 데이터 연동\n예산 범위 내 외부 광고비 최소화\n파일럿 프로모션은 1개 매장 우선 적용 후 전점 확대",
            )

            # 어떤 샘플을 사용할지 결정
            _is_dev = sample_loaded or prd_input_mode == "🗂 개발 샘플"
            _is_pmarket = prd_input_mode == "🏪 P마켓 샘플"
            _preset = _DEV_SAMPLE if _is_dev else (_PMARKET_SAMPLE if _is_pmarket else None)

            if prd_input_mode in ("📝 직접 입력", "🗂 개발 샘플", "🏪 P마켓 샘플") or sample_loaded:
                def _v(key, default=""):
                    return _preset[key] if _preset else default

                project_name = st.text_input(
                    "프로젝트명 *",
                    value=_v("project_name"),
                    placeholder="예: 대규모 이벤트 로그 파이프라인",
                )
                project_goal = st.text_area(
                    "프로젝트 목표 *",
                    value=_v("project_goal"),
                    height=80,
                )
                target_users = st.text_input(
                    "타깃 사용자",
                    value=_v("target_users"),
                )
                scope = st.text_area(
                    "프로젝트 범위",
                    value=_v("scope"),
                    height=60,
                )
                key_features = st.text_area(
                    "핵심 기능/과제 (줄바꿈 구분) *",
                    value=_v("key_features"),
                    height=100,
                    placeholder="기능/과제 1\n기능/과제 2\n기능/과제 3",
                )
                tech_stack = st.text_area(
                    "기술 스택 / 활용 도구 (줄바꿈 구분)",
                    value=_v("tech_stack"),
                    height=80,
                )

                col_d, col_w = st.columns(2)
                with col_d:
                    deadline = st.text_input("마감일", value=_v("deadline"))
                with col_w:
                    budget_weeks = st.number_input(
                        "예산 기준 주수", min_value=1, max_value=52,
                        value=int(_preset["budget_weeks"]) if _preset else 12,
                    )

                team_size = st.number_input(
                    "팀원 수", min_value=1, max_value=30,
                    value=int(_preset["team_size"]) if _preset else 4,
                )
                constraints = st.text_area(
                    "특수 제약사항 (줄바꿈 구분)",
                    value=_v("constraints"),
                    height=70,
                )
                prd_raw_text = None

            elif prd_input_mode == "📄 텍스트 파일 업로드":
                uploaded = st.file_uploader("PRD 텍스트 파일", type=["txt", "md"])
                if uploaded:
                    prd_raw_text = uploaded.read().decode("utf-8")
                    st.text_area("파일 내용 미리보기", prd_raw_text, height=200)
                    project_name = st.text_input("프로젝트명 확인/수정")
                else:
                    prd_raw_text = None
                    project_name = ""
                project_goal = target_users = scope = key_features = tech_stack = ""
                deadline = constraints = ""
                budget_weeks = 12
                team_size = 4

        with col2:
            st.markdown('<div class="section-title">👥 팀원 데이터 입력</div>', unsafe_allow_html=True)

            team_input_mode = st.radio(
                "입력 방식",
                ["📝 직접 입력", "📄 이력서 파일 업로드", "🗂 샘플 팀 사용", "🛒 P마켓 샘플팀 사용"],
                horizontal=True,
            )

            members_data = []

            if team_input_mode == "🗂 샘플 팀 사용":
                sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_members")
                if os.path.exists(sample_dir):
                    sample_files = sorted(os.listdir(sample_dir))
                    st.success(f"✅ {len(sample_files)}명의 샘플 팀원 로드됨")
                    for f in sample_files:
                        fpath = os.path.join(sample_dir, f)
                        with open(fpath, encoding="utf-8") as fi:
                            content = fi.read()
                        
                        # 세션 캐시 초기화
                        if "resume_cache" not in st.session_state:
                            st.session_state["resume_cache"] = {}
                        
                        file_key = f"sample_base_{f}"
                        if file_key in st.session_state["resume_cache"]:
                            profile = st.session_state["resume_cache"][file_key]["profile"]
                            content = st.session_state["resume_cache"][file_key]["content"]
                        else:
                            from data_pipeline.member_parser import MemberParser
                            # LLM 기반 정밀 파싱 수행 (Fallback 포함)
                            profile = MemberParser.from_resume_text_llm(content, name="미정")
                            
                            # 텍스트에서 이름 추출 시도
                            lines = content.splitlines()
                            name_line = next((l for l in lines if "이름:" in l), f"이름: {profile.name}")
                            name = name_line.replace("이름:", "").strip()
                            profile.name = name
                            
                            st.session_state["resume_cache"][file_key] = {"profile": profile, "content": content}

                        members_data.append({
                            "name": profile.name,
                            "role": "",  
                            "resume": content,
                            "mode": "resume",
                            "profile": profile
                        })
                        with st.expander(f"👤 {profile.name} (분석 완료)", expanded=False):
                            _render_member_profile_card(profile)

            elif team_input_mode == "🛒 P마켓 샘플팀 사용":
                sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_pmarket_members")
                if os.path.exists(sample_dir):
                    sample_files = sorted(os.listdir(sample_dir))
                    st.success(f"✅ {len(sample_files)}명의 P마켓 맞춤 샘플 팀원 로드됨")
                    for f in sample_files:
                        fpath = os.path.join(sample_dir, f)
                        with open(fpath, encoding="utf-8") as fi:
                            content = fi.read()
                        
                        # 세션 캐시 초기화
                        if "resume_cache" not in st.session_state:
                            st.session_state["resume_cache"] = {}
                        
                        file_key = f"sample_{f}"
                        if file_key in st.session_state["resume_cache"]:
                            profile = st.session_state["resume_cache"][file_key]["profile"]
                            content = st.session_state["resume_cache"][file_key]["content"]
                        else:
                            from data_pipeline.member_parser import MemberParser
                            profile = MemberParser.from_resume_text_llm(content, name="미정")
                            
                            lines = content.splitlines()
                            name_line = next((l for l in lines if "이름:" in l), f"이름: {profile.name}")
                            name = name_line.replace("이름:", "").strip()
                            profile.name = name
                            
                            st.session_state["resume_cache"][file_key] = {"profile": profile, "content": content}
                        
                        members_data.append({
                            "name": profile.name,
                            "role": "",  
                            "resume": content,
                            "mode": "resume",
                            "profile": profile
                        })
                        with st.expander(f"👤 {profile.name} (분석 완료)", expanded=False):
                            _render_member_profile_card(profile)

            elif team_input_mode == "📄 이력서 파일 업로드":
                uploaded_files = st.file_uploader(
                    "이력서 파일 (복수 업로드 가능)",
                    type=["txt", "md", "pdf"],
                    accept_multiple_files=True,
                )
                
                if "resume_cache" not in st.session_state:
                    st.session_state["resume_cache"] = {}

                for uf in uploaded_files:
                    file_key = f"{uf.name}_{uf.size}"
                    if file_key in st.session_state["resume_cache"]:
                        profile = st.session_state["resume_cache"][file_key]["profile"]
                        content = st.session_state["resume_cache"][file_key]["content"]
                    else:
                        if uf.name.lower().endswith(".pdf"):
                            content = _extract_text_from_pdf(uf)
                        else:
                            content = uf.read().decode("utf-8")
                            
                        from data_pipeline.member_parser import MemberParser
                        cleaned_name = _clean_member_name(uf.name)
                        profile = MemberParser.from_resume_text_llm(content, name=cleaned_name)
                        
                        # 텍스트에서 이름 추출 시도 (패턴 다양화)
                        import re
                        name_match = re.search(r'(?:이름|성명|Name|이 름)\s*[:\s]\s*([가-힣a-zA-Z\s]{2,10})', content)
                        if name_match:
                            name = name_match.group(1).strip()
                        else:
                            # 기존 "이름:" 방식 폴백
                            lines = content.splitlines()
                            name_line = next((l for l in lines[:10] if "이름:" in l), None)
                            if name_line:
                                name = name_line.replace("이름:", "").strip()
                            else:
                                # LLM이 '미정'을 반환했거나 추출 실패 시 파일명에서 가져온 이름 사용
                                name = profile.name if profile.name != "미정" else cleaned_name
                            
                        profile.name = name
                        
                        st.session_state["resume_cache"][file_key] = {"profile": profile, "content": content}
                    
                    with st.expander(f"👤 {profile.name} (분석 완료)", expanded=False):
                        _render_member_profile_card(profile)

                    members_data.append({
                        "name": profile.name, 
                        "role": profile.role.value if hasattr(profile.role, 'value') else "미정", 
                        "resume": content, 
                        "mode": "resume",
                        "profile": profile
                    })
                    st.success(f"✅ {profile.name} 로드됨")

            else:  # 직접 입력
                num_members = st.number_input("팀원 수", min_value=1, max_value=10, value=4)
                for i in range(num_members):
                    with st.expander(f"팀원 {i+1}", expanded=(i == 0)):
                        c1, c2 = st.columns(2)
                        with c1:
                            m_name = st.text_input(f"이름", key=f"mname_{i}", placeholder="홍길동")
                            m_years = st.number_input(f"경력(년)", key=f"myears_{i}", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
                        with c2:
                            m_avail = st.slider(f"투입률(%)", key=f"mavail_{i}", min_value=10, max_value=100, value=100, step=10)
                        m_tech = st.text_input(f"기술스택 (쉼표 구분)", key=f"mtech_{i}", placeholder="Python, FastAPI, PostgreSQL")
                        m_str = st.text_input(f"강점 (쉼표 구분)", key=f"mstr_{i}", placeholder="체계적인 문서화, 빠른 개발")
                        m_weak = st.text_input(f"약점 (쉼표 구분)", key=f"mweak_{i}", placeholder="러닝커브, 소통 어려움")
                        if m_name:
                            members_data.append({
                                "name": m_name, "role": None,
                                "years": m_years, "tech": m_tech,
                                "strengths": m_str, "weaknesses": m_weak,
                                "availability": m_avail, "mode": "form"
                            })

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📚 참고 DB (RAG)</div>', unsafe_allow_html=True)

            # ── RAG 전략 선택 ─────────────────────────────────
            from data_pipeline.rag_strategies import STRATEGY_INFO, list_strategy_keys
            _rag_keys = list_strategy_keys()
            _rag_labels = {
                k: f"{STRATEGY_INFO[k]['icon']}  {STRATEGY_INFO[k]['name']}  —  {STRATEGY_INFO[k]['description']}"
                for k in _rag_keys
            }
            selected_rag_strategy = st.radio(
                "🔬 RAG 검색 전략",
                options=_rag_keys,
                format_func=lambda x: _rag_labels[x],
                index=0,
                key="rag_strategy",
                help="검색 전략에 따라 WBS 생성에 참고되는 문서가 달라집니다. 결과 탭에서 비교 분석을 확인하세요.",
            )
            with st.expander(f"ℹ️ {STRATEGY_INFO[selected_rag_strategy]['name']} 설명", expanded=False):
                st.markdown(STRATEGY_INFO[selected_rag_strategy]["detail"])
                col_p, col_c = st.columns(2)
                with col_p:
                    st.markdown("**장점**")
                    for p in STRATEGY_INFO[selected_rag_strategy]["pros"]:
                        st.markdown(f"- ✅ {p}")
                with col_c:
                    st.markdown("**단점**")
                    for c in STRATEGY_INFO[selected_rag_strategy]["cons"]:
                        st.markdown(f"- ⚠️ {c}")

            ref_mode = st.radio("참고 데이터", ["🗂 샘플 WBS 사용", "🗂 샘플 회의록 사용", "📄 텍스트 파일 업로드", "🎙️ 회의 음성 업로드 (STT 파싱)", "건너뜀"], horizontal=True)
            ref_wbs_text = ""
            meeting_text = ""

            if ref_mode == "🗂 샘플 WBS 사용":
                sample_ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_reference_wbs.txt")
                if os.path.exists(sample_ref_path):
                    with open(sample_ref_path, encoding="utf-8") as f:
                        ref_wbs_text = f.read()
                    st.success("✅ 샘플 참고 WBS 로드됨")
            elif ref_mode == "🗂 샘플 회의록 사용":
                sample_meeting_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_meeting_transcript.txt")
                if os.path.exists(sample_meeting_path):
                    with open(sample_meeting_path, encoding="utf-8") as f:
                        meeting_text = f.read()
                    st.success("✅ 샘플 회의록 로드됨 (화자 분리 텍스트)")
                    with st.expander("📝 회의록 미리보기", expanded=False):
                        st.text(meeting_text[:800] + "..." if len(meeting_text) > 800 else meeting_text)
            elif ref_mode == "📄 텍스트 파일 업로드":
                uploaded_wbs = st.file_uploader("참고 WBS 파일 (txt, md)", type=["txt", "md"])
                if uploaded_wbs:
                    ref_wbs_text = uploaded_wbs.read().decode("utf-8")
                    st.success("✅ 참고 WBS 업로드됨")
            elif ref_mode == "🎙️ 회의 음성 업로드 (STT 파싱)":
                st.markdown("💡 **오디오 파일을 업로드하면 AI가 자동으로 화자를 분리하고 텍스트로 변환하여 회의록을 추출합니다.**")
                uploaded_audio = st.file_uploader("음성 파일 업로드 (mp3, wav, m4a 등)", type=["mp3", "wav", "m4a", "flac"])
                if uploaded_audio:
                    import tempfile
                    
                    if st.button("음성 텍스트화(STT) 돌리기", use_container_width=True):
                        with st.spinner("GPU를 사용하여 화자를 정밀 분리하는 중... (음성 길이에 따라 1~2분 소요)"):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
                                tmp_file.write(uploaded_audio.read())
                                tmp_audio_path = tmp_file.name
                                
                            try:
                                from STT_DONGK import process_meeting_audio
                                from agents.llm_config import HF_TOKEN
                                
                                hf_token = HF_TOKEN if HF_TOKEN else os.environ.get("HF_TOKEN")
                                transcript = process_meeting_audio(tmp_audio_path, hf_token)
                                
                                st.session_state["stt_meeting_text"] = transcript
                                st.success("✅ STT 기반 화자 분리 완료!")
                            except Exception as e:
                                st.error(f"STT 오류: {e}")
                                
                if st.session_state.get("stt_meeting_text"):
                    meeting_text = st.session_state["stt_meeting_text"]
                    with st.expander("📝 파싱된 회의록 확인 (RAG 반영 예정)", expanded=True):
                        st.text(meeting_text)

            st.markdown("---")
            # 세션 상태에서 필요한 변수들이 정의되어 있는지 확인 (없으면 초기값)
            if st.button("🚀 WBS 워크플로우 시작", use_container_width=True):
                if not project_name or not key_features:
                    st.error("⚠️ 프로젝트명과 핵심 기능은 필수 입력 사항입니다.")
                else:
                    _run_wbs_generation(
                        project_name, project_goal, target_users, scope,
                        key_features, tech_stack, deadline, budget_weeks, team_size, constraints,
                        members_data, ref_wbs_text, meeting_text, min_rounds, max_rounds,
                        prd_raw_text=prd_raw_text,
                        wbs_live_placeholder=st.session_state.get("wbs_live_placeholder_ref"),
                        debate_live_placeholder=st.session_state.get("debate_live_placeholder_ref"),
                    )
    with tabs[1]:
        # 실시간 업데이트용 플레이스홀더 준비
        debate_live_placeholder_ref = st.empty()
        st.session_state["debate_live_placeholder_ref"] = debate_live_placeholder_ref

        debate_log = st.session_state.get("debate_log", [])

        if not debate_log:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">🤖</span>
                <h3>에이전트 토론 대기 중</h3>
                <p>WBS 생성을 시작하면 에이전트들의 실시간 토론이 이곳에 표시됩니다.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            import html as _html

            # ── 에이전트 호출 트리 (결과 뷰) ──
            col_tree, col_chat = st.columns([1, 2], gap="large")

            with col_tree:
                st.markdown('<div class="section-title">🌐 에이전트 호출 트리</div>', unsafe_allow_html=True)
                # 세션에서 마지막 상태 복원 (실행 후 완료 상태)
                _last_state = {
                    "current_wbs_draft": st.session_state.get("wbs_output") and True,
                    "called_agents": st.session_state.get("called_agents_final", []),
                    "final_wbs": bool(st.session_state.get("wbs_output")),
                    "current_round": len([m for m in debate_log if getattr(m,'message_type','')=='mediation']),
                    "consensus_reached": True,
                    "debate_log": debate_log,
                    "_current_agent_acting": "",
                }
                _render_agent_call_tree(_last_state)

            with col_chat:
                st.markdown(
                    f'<div class="section-title">💬 에이전트 전략 토론 타임라인 '
                    f'<span style="font-size:0.75rem;font-weight:500;color:#94a3b8;margin-left:8px;">'
                    f'총 {len(debate_log)}개 메시지</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="debate-timeline">', unsafe_allow_html=True)
                for m in debate_log:
                    render_debate_message(m)
                st.markdown('</div>', unsafe_allow_html=True)
                st.write("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 탭 3: WBS 결과
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[2]:
        # 실시간 업데이트용 플레이스홀더 준비
        wbs_live_placeholder_ref = st.empty()
        st.session_state["wbs_live_placeholder_ref"] = wbs_live_placeholder_ref

        wbs_output = st.session_state.get("wbs_output")
        gen_success = st.session_state.get("wbs_generation_success", False)

        if gen_success:
            st.success("✅ **WBS 생성 및 파이프라인 처리가 완료되었습니다!**")
            st.session_state["wbs_generation_success"] = False

        if not wbs_output:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">📊</span>
                <h3>WBS 결과 대기 중</h3>
                <p>WBS 생성을 완료하면 계층적 WBS 테이블이 이곳에 표시됩니다.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            _render_wbs_result(wbs_output)

            # ─── RAG 분석 패널 ────────────────────────
            rag_meta = st.session_state.get("rag_metadata")
            if rag_meta:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                _render_rag_analysis(rag_meta)

            # ─── 평가 지표 패널 ────────────────────────
            metrics_data = st.session_state.get("metrics")
            if metrics_data and not metrics_data.get("error"):
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                _render_metrics_table(metrics_data)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 탭 4: 설명 가능성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[3]:
        wbs_output = st.session_state.get("wbs_output")

        if not wbs_output:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-state-icon">🔍</span>
                <h3>설명 가능성 대기 중</h3>
                <p>WBS를 먼저 생성하면 각 태스크의 일정 산출 근거를 확인할 수 있습니다.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-title">🔍 태스크별 일정 산출 근거 조회</div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">'
                '각 태스크를 선택하면 배정 일정·버퍼 근거·관련 에이전트 토론을 확인할 수 있습니다.</p>',
                unsafe_allow_html=True
            )

            # 레벨 필터
            filter_col, select_col = st.columns([1, 3])
            with filter_col:
                level_filter = st.selectbox("레벨 필터", ["전체", "L1만", "L2만", "L3만"])
            with select_col:
                level_map = {"전체": None, "L1만": "L1", "L2만": "L2", "L3만": "L3"}
                filtered_tasks = [
                    t for t in wbs_output.tasks
                    if level_map[level_filter] is None or t.level.value == level_map[level_filter]
                ]
                task_ids = [f"[{t.level.value}] {t.task_id}: {t.title}" for t in filtered_tasks]
                selected = st.selectbox("태스크 선택", task_ids, label_visibility="collapsed")

            if selected:
                # "[L1] L1-01: 기획 및 설계" → "L1-01"
                task_id = selected.split(":")[0].split("]")[-1].strip()
                from output.report_writer import ReportWriter

                # 팀원 이름 반영을 위해 assigned_to를 이름으로 치환
                team_members_ss = st.session_state.get("team_members", [])
                member_map_ss = {m.member_id: f"{m.name} ({m.role.value})" for m in team_members_ss}

                # 태스크의 assigned_to를 이름으로 교체 (display용 복사본)
                task_obj = next((t for t in wbs_output.tasks if t.task_id == task_id), None)
                if task_obj and task_obj.assigned_to:
                    named_assignees = [member_map_ss.get(mid, mid) for mid in task_obj.assigned_to]
                    task_obj = task_obj.model_copy(update={"assigned_to": named_assignees})
                    # wbs_output을 임시로 수정
                    tasks_replaced = [
                        task_obj if t.task_id == task_id else t for t in wbs_output.tasks
                    ]
                    from schemas.wbs_schema import WBSOutput
                    wbs_display = wbs_output.model_copy(update={"tasks": tasks_replaced})
                else:
                    wbs_display = wbs_output

                writer = ReportWriter()
                explanation = writer.explain_task_schedule(task_id, wbs_display)
                st.markdown(explanation)


def _run_wbs_generation(
    project_name, project_goal, target_users, scope,
    key_features, tech_stack, deadline, budget_weeks, team_size, constraints,
    members_data, ref_wbs_text, meeting_text, min_rounds, max_rounds,
    prd_raw_text=None,
    wbs_live_placeholder=None,
    debate_live_placeholder=None,
):
    """WBS 생성 파이프라인 실행"""
    from data_pipeline.prd_parser import PRDParser
    from data_pipeline.member_parser import MemberParser
    from data_pipeline.vector_store import WBSVectorStore
    from data_pipeline.rag_strategies import get_strategy, STRATEGY_INFO
    from persona_engine.persona_builder import PersonaBuilder
    from agents.state import create_initial_state
    from output.wbs_generator import WBSGenerator
    from output.report_writer import ReportWriter

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    live_debate_container = debate_live_placeholder if debate_live_placeholder else st.container()

    def update_progress(step: str, detail: str = ""):
        import html as _h
        progress_placeholder.markdown(f"""
        <div class="progress-card">
            <div class="progress-agent">⚡ {_h.escape(step)}</div>
            <div class="progress-detail">{_h.escape(detail)}</div>
        </div>
        """, unsafe_allow_html=True)

    try:
        # ─── 1단계: PRD 파싱 ────────────────────────
        update_progress("1단계: PRD 분석 중...", "프로젝트 요구사항을 구조화합니다")
        time.sleep(0.5)

        if prd_raw_text:
            prd = PRDParser.from_text(prd_raw_text, project_name or "미정")
        else:
            prd = PRDParser.from_form(
                project_name=project_name or "미정",
                project_goal=project_goal or "미정",
                target_users=target_users or "미정",
                scope=scope or "전체",
                key_features_text=key_features or "기능 미정",
                tech_stack_text=tech_stack or "",
                deadline=deadline or None,
                team_size=team_size,
                budget_weeks=int(budget_weeks),
                constraints_text=constraints or "",
            )

        # ─── 2단계: 팀원 파싱 ───────────────────────
        update_progress("2단계: 팀원 메타데이터 분석 중...", "역량, 강점, 약점을 파악합니다")
        time.sleep(0.5)

        team_members = []
        if not members_data:
            # 기본 더미 팀원 생성
            from schemas.member_schema import MemberProfile, MemberRole
            team_members = [
                MemberProfile(
                    member_id="MBR-001",
                    name="BE 개발자",
                    role=MemberRole.BACKEND,
                    years_of_experience=4.0,
                    tech_stack=["Python", "FastAPI", "PostgreSQL"],
                    primary_skills=["API 개발", "DB 설계"],
                    strengths=["체계적인 개발"],
                    weaknesses=["러닝커브"],
                ),
                MemberProfile(
                    member_id="MBR-002",
                    name="FE 개발자",
                    role=MemberRole.FRONTEND,
                    years_of_experience=3.0,
                    tech_stack=["React", "TypeScript"],
                    primary_skills=["UI 구현", "컴포넌트 설계"],
                    strengths=["빠른 구현"],
                    weaknesses=["API 연동 이슈"],
                ),
            ]
        else:
            for m in members_data:
                if m.get("mode") == "resume":
                    # 이미 UI 로드 시점에 정밀 파싱된 profile이 있다면 재사용
                    if "profile" in m:
                        member = m["profile"]
                    else:
                        member = MemberParser.from_resume_text(m["resume"], m["name"])
                    team_members.append(member)
                else:
                    tech_list = [t.strip() for t in (m.get("tech", "") or "").split(",") if t.strip()]
                    str_list = [s.strip() for s in (m.get("strengths", "") or "").split(",") if s.strip()]
                    weak_list = [w.strip() for w in (m.get("weaknesses", "") or "").split(",") if w.strip()]

                    from schemas.member_schema import MemberProfile, MemberRole
                    import uuid
                    member = MemberProfile(
                        member_id=f"MBR-{uuid.uuid4().hex[:4].upper()}",
                        name=m.get("name", "팀원"),
                        role=MemberParser._map_role(m.get("role")) if m.get("role") else None,
                        years_of_experience=m.get("years", 3.0),
                        tech_stack=tech_list or ["미정"],
                        primary_skills=tech_list[:3] or ["미정"],
                        strengths=str_list or ["성실함"],
                        weaknesses=weak_list or ["미정"],
                        availability_percent=m.get("availability", 100),
                    )
                    team_members.append(member)

        st.session_state["team_members"] = team_members

        # ─── 3단계: 벡터 DB 구축 ───────────────────
        update_progress("3단계: Vector DB 구축 중...", "문서를 인덱싱하고 RAG를 준비합니다")
        time.sleep(0.5)

        vector_store = WBSVectorStore()
        vector_store.add_prd(prd)
        for m in team_members:
            vector_store.add_member(m)
        if ref_wbs_text:
            vector_store.add_reference_wbs(ref_wbs_text, prd.project_name)
        if meeting_text:
            vector_store.add_meeting_log(meeting_text)

        # eDISC 프로파일 로드 및 벡터 DB 추가 (캐시 재사용)
        disc_agent_contexts = {}
        try:
            _disc_profiles = _load_disc_profiles_cached()
            if _disc_profiles:
                for _dp in _disc_profiles.values():
                    vector_store.add_disc_profile(_dp)
                disc_agent_contexts = {n: p.to_agent_context() for n, p in _disc_profiles.items()}
                update_progress("3단계: Vector DB 구축 중...",
                                f"eDISC 프로파일 {len(_disc_profiles)}명 로드됨: {', '.join(_disc_profiles.keys())}")
        except Exception:
            pass  # eDISC 없어도 정상 동작

        # RAG 검색 (선택된 전략 사용)
        _rag_strategy_key = st.session_state.get("rag_strategy", "vanilla")
        _rag_strategy = get_strategy(_rag_strategy_key)
        update_progress(
            "3단계: Vector DB 구축 중...",
            f"RAG 전략: {STRATEGY_INFO[_rag_strategy_key]['name']} — {STRATEGY_INFO[_rag_strategy_key]['description']}",
        )

        _common_kwargs = dict(
            documents=vector_store._documents,
            vectorstore=vector_store._vectorstore,
            embeddings=vector_store._embeddings,
        )
        rag_wbs = _rag_strategy.retrieve(
            f"{prd.project_name} WBS 일정", "reference_wbs", k=3, **_common_kwargs
        )
        rag_meetings = _rag_strategy.retrieve(
            "일정 버퍼 교훈", "meeting_log", k=3, **_common_kwargs
        )
        rag_wbs_texts = [d["content"] for d in rag_wbs]
        rag_meeting_texts = [d["content"] for d in rag_meetings]

        # RAG 메타데이터 저장 (결과 탭 분석 패널용)
        st.session_state["rag_metadata"] = {
            "strategy_key": _rag_strategy_key,
            "strategy_name": STRATEGY_INFO[_rag_strategy_key]["name"],
            "strategy_icon": STRATEGY_INFO[_rag_strategy_key]["icon"],
            "wbs_results": rag_wbs,
            "meeting_results": rag_meetings,
            "doc_stats": vector_store.get_stats(),
        }

        # ─── 4단계: 페르소나 생성 ──────────────────
        update_progress("4단계: 에이전트 페르소나 생성 중...", "팀원 역량을 에이전트에 주입합니다")
        time.sleep(0.5)

        team_summary = PersonaBuilder.generate_team_summary(team_members)
        supervisor_persona = PersonaBuilder.build_supervisor_persona(
            supervisor_name="PM 에이전트", team_summary=team_summary
        )
        member_personas = PersonaBuilder.build_all_personas(team_members)
        all_personas = {"supervisor": supervisor_persona, **member_personas}

        # ─── 5단계: LangGraph 오케스트레이션 ───────
        update_progress("5단계: 다중 에이전트 토론 진행 중...", f"최대 {max_rounds}라운드 토론을 진행합니다")

        from agents.state import create_initial_state

        initial_state = create_initial_state(
            prd=prd,
            team_members=team_members,
            agent_personas=all_personas,
            min_rounds=min_rounds,
            max_rounds=max_rounds,
            model_config=st.session_state.get("agent_model_config", {}),
        )
        initial_state["rag_reference_wbs"] = rag_wbs_texts
        initial_state["rag_meeting_logs"] = rag_meeting_texts
        initial_state["disc_profiles"] = disc_agent_contexts

        # 에이전트 아이콘/스타일 매핑 (실시간 렌더링용) - AGENT_STYLES 상단 정의 사용

        # 실시간 렌더링 레이아웃: 트리(좌) + 토론(우)
        col_live_tree, col_live_chat = st.columns([1, 2], gap="large")

        with col_live_tree:
            st.markdown('<div class="section-title">🌐 에이전트 호출 트리</div>', unsafe_allow_html=True)
            tree_placeholder = st.empty()

        with col_live_chat:
            tracker_placeholder = st.empty()
            if debate_live_placeholder:
                debate_container = debate_live_placeholder.container()
            else:
                debate_container = st.container()
            typing_placeholder = st.empty()

        # LangGraph/Sequential 실행 (Generator 순회)
        final_state = initial_state
        from orchestration.debate_loop import execute_sympo_flow
        import re as _re
        import html as _html

        # 실시간 토론 렌더링: 이전에 렌더링된 메시지 수 추적 (새 메시지만 추가)
        rendered_msg_count = 0

        # Generator를 순회하며 실시간 UI 업데이트
        for current_state in execute_sympo_flow(initial_state, max_rounds):
            final_state = current_state
            all_msgs = current_state.get("debate_log", [])
            acting_agent = current_state.get("_current_agent_acting")
            current_round_n = current_state.get("current_round", 0)
            current_l2 = current_state.get("_current_l2_task_id")

            # ── 에이전트 호출 트리 실시간 업데이트 ─────────────
            with tree_placeholder.container():
                _render_agent_call_tree(current_state)

            # ── WBS 실시간 렌더링 ───────────────────────────
            current_draft = current_state.get("current_wbs_draft")
            if current_draft and wbs_live_placeholder:
                from schemas.wbs_schema import WBSOutput
                draft_output = WBSOutput(
                    project_name=prd.project_name,
                    tasks=[t.model_dump() for t in current_draft],
                    total_weeks=0.0,
                    debate_log=[],
                    summary="실시간 생성 중인 초안입니다."
                )
                with wbs_live_placeholder.container():
                    st.markdown("""
                    <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                        <h4 style="margin: 0; color: #92400e; font-size: 1rem;">🕒 WBS 초안 실시간 생성/수정 중...</h4>
                        <p style="margin: 4px 0 0 0; color: #b45309; font-size: 0.85rem;">에이전트 토론 결과가 반영되고 있습니다. 잠시만 기다려주세요.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    _render_wbs_result(draft_output, is_draft=True, key_suffix=f"_draft_{current_round_n}")

            # ── 진행 상태 업데이트 ──────────────────────────────
            is_wbs_draft_done = bool(current_state.get("current_wbs_draft"))

            if acting_agent == "wbs_gen_node" and not is_wbs_draft_done:
                status_placeholder.markdown("""
                <div class="progress-card">
                    <div class="progress-agent">🤖 Phase 1 — WBS Gen Agent 실행 중</div>
                    <div class="progress-detail">PRD를 분석하여 3단계 계층 WBS 초안을 생성하고 있습니다...</div>
                </div>""", unsafe_allow_html=True)
            elif acting_agent == "supervisor_task_match":
                status_placeholder.markdown(f"""
                <div class="progress-card">
                    <div class="progress-agent">🎯 Phase 2 — Task 매칭 Manager (Round {current_round_n + 1})</div>
                    <div class="progress-detail">팀원 역량 프로필을 분석하여 L2 기능그룹별 R&amp;R을 배분합니다...</div>
                </div>""", unsafe_allow_html=True)
            elif acting_agent and "슈퍼바이저" in str(acting_agent):
                status_placeholder.markdown(f"""
                <div class="progress-card">
                    <div class="progress-agent">🎯 슈퍼바이저(PM) 중재 중</div>
                    <div class="progress-detail">토론 결과를 종합하여 버퍼·재배정을 반영합니다...</div>
                </div>""", unsafe_allow_html=True)
            elif acting_agent:
                safe_agent = _html.escape(str(acting_agent))
                l2_hint = f" — {_html.escape(current_l2)}" if current_l2 else ""
                status_placeholder.markdown(f"""
                <div class="progress-card">
                    <div class="progress-agent">💬 {safe_agent} 검토 중{l2_hint}</div>
                    <div class="progress-detail">담당 관점에서 WBS 태스크의 리스크와 버퍼를 분석하고 있습니다...</div>
                </div>""", unsafe_allow_html=True)
            elif all_msgs:
                last_msg = all_msgs[-1]
                safe_name = _html.escape(last_msg.agent_name)
                status_placeholder.markdown(f"""
                <div class="progress-card">
                    <div class="progress-agent">✅ {safe_name} 응답 완료</div>
                    <div class="progress-detail">총 {len(all_msgs)}개 메시지 누적 · Round {current_round_n}</div>
                </div>""", unsafe_allow_html=True)

            # ── Control Room ──────────────────────────────────
            called_agents = current_state.get("called_agents", [])
            rationale = current_state.get("supervisor_rationale", "")
            l2_mapping = current_state.get("l2_agent_mapping", {})

            if called_agents and rationale:
                agents_html = "".join(
                    f'<span class="control-room-badge">{_html.escape(str(a))}</span>'
                    for a in called_agents
                )
                # L2별 에이전트 매핑 표시
                l2_mapping_html = ""
                if l2_mapping:
                    l2_items = "".join(
                        f'<div style="font-size:0.78rem;color:#475569;padding:2px 0;">'
                        f'<span style="color:#6366f1;font-weight:600;">{_html.escape(l2_id)}</span> → '
                        f'{", ".join(_html.escape(a) for a in agents)}</div>'
                        for l2_id, agents in list(l2_mapping.items())[:6]
                    )
                    more = f'<div style="font-size:0.75rem;color:#94a3b8;">+{len(l2_mapping)-6}개 더...</div>' if len(l2_mapping) > 6 else ""
                    l2_mapping_html = (
                        f'<div style="border-top:1px solid #e0e7ff;padding-top:0.5rem;margin-top:0.4rem;">'
                        f'<span style="color:#6b7280;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">L2별 담당 에이전트</span><br>'
                        f'{l2_items}{more}</div>'
                    )

                with tracker_placeholder.container():
                    st.markdown(f"""
                    <div class="control-room">
                        <div class="control-room-title">📡 실시간 통합 관제실 (Call Tracker)</div>
                        <div style="margin-bottom:0.4rem;font-size:0.85rem;">
                            <span style="color:#6b7280;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">호출 에이전트</span><br>
                            {agents_html}
                        </div>
                        <div style="font-size:0.88rem;color:#374151;line-height:1.5;border-top:1px solid #e0e7ff;padding-top:0.5rem;margin-top:0.4rem;">
                            <span style="color:#6b7280;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">사유</span><br>
                            {_html.escape(str(rationale))}
                        </div>
                        {l2_mapping_html}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                tracker_placeholder.empty()

            # ── 실시간 토론 로그: 새 메시지만 증분 렌더링 ─────
            new_msgs = all_msgs[rendered_msg_count:]
            if new_msgs:
                with debate_container:
                    for m in new_msgs:
                        render_debate_message(m)

                rendered_msg_count = len(all_msgs)

            # ── 타이핑 인디케이터 (현재 작업 중인 에이전트) ──
            if acting_agent:
                safe_agent = _html.escape(str(acting_agent))
                typing_placeholder.markdown(f"""
                <div class="typing-indicator" style="margin-top:0.5rem;">
                    💬 <strong>{safe_agent}</strong> 에이전트가 응답을 작성 중...
                    <span style="display:inline-block;animation:blink 1s step-end infinite;">▋</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                typing_placeholder.empty()

            # 세션 스테이트 업데이트
            st.session_state["debate_log"] = all_msgs
        
        status_placeholder.empty()
        typing_placeholder.empty()
        if wbs_live_placeholder:
            wbs_live_placeholder.empty()

        # ─── 6단계: WBS 생성 ───────────────────────
        update_progress("6단계: 최종 WBS 생성 중...", "합의된 일정으로 계층적 WBS를 산출합니다")
        time.sleep(0.5)

        generator = WBSGenerator()
        # final_wbs/current_wbs_draft 내의 모델 인스턴스들을 dict로 변환하여 전달 (Pydantic v2 호환성 고도화)
        raw_tasks = final_state.get("final_wbs") or final_state.get("current_wbs_draft", [])
        wbs_output = generator.generate(
            tasks=raw_tasks,
            prd=prd,
            team=team_members,
            debate_log=final_state.get("debate_log", []),
            summary=final_state.get("generation_summary", ""),
        )

        # 보고서 저장 및 WBS 산출 (Bug Fix: pass tasks instead of team members)
        writer = ReportWriter()
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
        os.makedirs(output_dir, exist_ok=True)
        writer.output_dir = output_dir
 
        # wbs_output.tasks를 전달하여 MemberProfile 관련 에러 해결
        wbs_path = writer.write_wbs_markdown(wbs_output, wbs_output.tasks)
        log_path = writer.write_debate_log(wbs_output.debate_log, prd.project_name)
        json_path = writer.write_json_output(wbs_output)

        # Session state 저장
        st.session_state["wbs_output"] = wbs_output
        st.session_state["debate_log"] = final_state.get("debate_log", [])
        st.session_state["wbs_generation_success"] = True
        # R&R 가시화를 위한 추가 저장
        st.session_state["calling_context"] = final_state.get("calling_context", {})
        st.session_state["called_agents_final"] = final_state.get("called_agents", [])

        # ─── 7단계: 평가 지표 자동 산출 ──────────────
        update_progress("7단계: 평가 지표 계산 중...", "RAGAS, Planning Score, SR 등 7개 정량 지표를 산출합니다")
        try:
            from metrics import compute_all_metrics
            _exp_config = {
                "rag_strategy": st.session_state.get("rag_strategy", "vanilla"),
                "min_rounds": min_rounds,
                "max_rounds": max_rounds,
                "team_size": len(team_members),
                "budget_weeks": getattr(prd, "budget_weeks", None),
                "llm_backend": os.environ.get("LLM_BACKEND", "unknown"),
                "note": "",
            }
            metrics_result = compute_all_metrics(
                final_state=final_state,
                prd=prd,
                team_members=team_members,
                output_dir=output_dir,
                experiment_config=_exp_config,
            )
            st.session_state["metrics"] = metrics_result
        except Exception as e:
            st.session_state["metrics"] = {"error": str(e)}

        progress_placeholder.empty()
        # 즉시 전체 화면 리렌더링하여 데이터 반영
        st.rerun()

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ 오류 발생: {str(e)}")
        with st.expander("상세 오류 정보"):
            st.code(traceback.format_exc())


def _execute_pipeline(initial_state, max_rounds: int):
    """(더 이상 직접 호출되지 않음, 주석 처리)"""
    pass


def _render_rag_analysis(rag_meta: dict):
    """
    RAG 검색 전략 분석 패널.
    각 전략이 어떤 문서를 어떤 근거로 선택했는지 시각화합니다.
    """
    from data_pipeline.rag_strategies import STRATEGY_INFO

    icon = rag_meta.get("strategy_icon", "🔍")
    name = rag_meta.get("strategy_name", "RAG")
    key = rag_meta.get("strategy_key", "vanilla")
    info = STRATEGY_INFO.get(key, {})

    st.markdown(
        f'<div class="section-title">{icon} RAG 검색 분석 — {name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"> {info.get('detail', '')}")

    # 문서 통계
    doc_stats = rag_meta.get("doc_stats", {})
    if doc_stats:
        cols = st.columns(len(doc_stats))
        labels = {
            "prd": "PRD",
            "member": "팀원 프로필",
            "reference_wbs": "참고 WBS",
            "meeting_log": "회의록",
            "disc_profile": "eDISC",
        }
        for col, (dtype, cnt) in zip(cols, doc_stats.items()):
            with col:
                st.metric(labels.get(dtype, dtype), f"{cnt}건")

    # WBS 참고 검색 결과
    wbs_results = rag_meta.get("wbs_results", [])
    meeting_results = rag_meta.get("meeting_results", [])

    col_wbs, col_meet = st.columns(2)

    with col_wbs:
        st.markdown("#### 📋 참고 WBS 검색 결과")
        if not wbs_results:
            st.info("참고 WBS 데이터 없음")
        for i, doc in enumerate(wbs_results):
            rag_info = doc.get("_rag_info", {})
            score = doc.get("_rag_score", 0)
            method = rag_info.get("method", key)

            # 전략별 배지
            if method == "rrf":
                badge = (
                    f"RRF {rag_info.get('rrf_score', 0):.4f} | "
                    f"BM25순위 {rag_info.get('bm25_rank', '—')} | "
                    f"Dense순위 {rag_info.get('dense_rank', '—')}"
                )
            elif method == "graph_rag":
                badge = rag_info.get("graph_path", "")
            elif method == "agentic":
                hops = rag_info.get("hops", [])
                badge = f"{len(hops)}홉 | 커버리지 {rag_info.get('coverage', 0):.0%}"
            elif method == "faiss_dense":
                badge = f"유사도 {score:.3f} (L2 거리 {rag_info.get('l2_distance', 0):.3f})"
            else:
                badge = f"점수 {score:.1f}"

            with st.expander(f"**[{i+1}]** {badge}", expanded=(i == 0)):
                st.text(doc["content"][:400] + ("..." if len(doc["content"]) > 400 else ""))
                meta = {k: v for k, v in doc.get("metadata", {}).items() if not k.startswith("_")}
                if meta:
                    st.caption(f"메타데이터: {meta}")

                # 그래프 RAG 엔티티 시각화
                if method == "graph_rag" and rag_info.get("entities"):
                    st.markdown(
                        "**추출 엔티티**: " + " · ".join(
                            f"`{e}`" for e in rag_info["entities"]
                        )
                    )
                # Agentic 홉 로그
                if method == "agentic" and rag_info.get("hops"):
                    for h in rag_info["hops"]:
                        st.caption(
                            f"홉 {h['hop']}: 쿼리=`{h['query'][:60]}` | "
                            f"신규문서={h['new_docs']} | "
                            f"커버리지={h['coverage']:.0%}"
                            + (" ✅ 종료" if h.get("stopped") else "")
                        )

    with col_meet:
        st.markdown("#### 💬 회의록 검색 결과")
        if not meeting_results:
            st.info("회의록 데이터 없음")
        for i, doc in enumerate(meeting_results):
            rag_info = doc.get("_rag_info", {})
            score = doc.get("_rag_score", 0)
            method = rag_info.get("method", key)

            if method == "rrf":
                badge = f"RRF {rag_info.get('rrf_score', 0):.4f}"
            elif method == "graph_rag":
                badge = rag_info.get("graph_path", "")
            elif method == "agentic":
                badge = f"커버리지 {rag_info.get('coverage', 0):.0%}"
            else:
                badge = f"점수 {score:.3f}"

            with st.expander(f"**[{i+1}]** {badge}", expanded=(i == 0)):
                st.text(doc["content"][:400] + ("..." if len(doc["content"]) > 400 else ""))
                if method == "graph_rag" and rag_info.get("entities"):
                    st.markdown(
                        "**추출 엔티티**: " + " · ".join(
                            f"`{e}`" for e in rag_info["entities"]
                        )
                    )

    # 전략 비교 안내
    st.info(
        f"💡 다른 RAG 전략으로 비교 실험하려면 **입력 탭 → RAG 검색 전략**을 변경하고 "
        f"WBS를 다시 생성하세요. 현재: **{name}**"
    )


def _render_metrics_table(metrics_data: dict):
    """
    평가 지표 전체를 기준값·판정 포함 표로 렌더링합니다.
    - 요약 표: 7개 핵심 지표 + 기준값 + 판정
    - 세부 표: 각 지표의 세부 수치
    - 이력 표: 과거 실행 누적 비교
    """
    import pandas as pd
    from metrics import METRIC_BENCHMARKS, _judge

    st.markdown('<div class="section-title">📈 평가 지표 (Evaluation Metrics)</div>', unsafe_allow_html=True)

    # ── 실행 메타 정보 ────────────────────────────────────────
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    with meta_col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="font-size:1rem;">'
            f'{metrics_data.get("project_name","—")}</div>'
            f'<div class="metric-label">프로젝트</div></div>',
            unsafe_allow_html=True,
        )
    with meta_col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{metrics_data.get("model_backend","—").upper()}</div>'
            f'<div class="metric-label">LLM 백엔드</div></div>',
            unsafe_allow_html=True,
        )
    with meta_col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{metrics_data.get("total_tasks",0)}</div>'
            f'<div class="metric-label">총 태스크 수</div></div>',
            unsafe_allow_html=True,
        )
    with meta_col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{metrics_data.get("debate_rounds",0)}</div>'
            f'<div class="metric-label">토론 라운드</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 7개 핵심 지표 요약 표 ─────────────────────────────────
    st.markdown("#### 핵심 지표 요약 (기준값 대비 판정)")

    ragas  = metrics_data.get("ragas_faithfulness", {})
    sr     = metrics_data.get("success_rate", {})
    ps     = metrics_data.get("planning_score", {})
    buf    = metrics_data.get("buffer_ratio", {})
    turns  = metrics_data.get("interaction_turns", {})
    sup    = metrics_data.get("supervisor_intervention", {})
    conv   = metrics_data.get("convergence", {})

    ragas_val   = ragas.get("faithfulness", 0.0)
    sr_val      = sr.get("success_rate", 0.0)
    ps_val      = ps.get("planning_score", 0.0)
    buf_val     = buf.get("buffer_ratio_pct", 0.0)
    turns_val   = turns.get("total_messages", 0)
    sup_val     = sup.get("intervention_ratio", 0.0)
    conv_val    = conv.get("is_converging", False)

    summary_rows = [
        {
            "#": 1,
            "지표명": "RAGAS Faithfulness",
            "값": f"{ragas_val:.2%}",
            "기준": "≥ 50%",
            "판정": _judge("ragas_faithfulness", ragas_val),
            "설명": "RAG 컨텍스트 근거 비율",
        },
        {
            "#": 2,
            "지표명": "Success Rate (SR)",
            "값": f"{sr_val:.2%}",
            "기준": "= 100%",
            "판정": _judge("success_rate", sr_val),
            "설명": "PRD 기능 커버리지",
        },
        {
            "#": 3,
            "지표명": "Planning Score",
            "값": "측정 불가 (임베딩 모델 미설치)" if ps_val < 0 else f"{ps_val:.4f}",
            "기준": "≥ 0.5000",
            "판정": "—" if ps_val < 0 else _judge("planning_score", ps_val),
            "설명": "태스크-팀원 역량 코사인 유사도",
        },
        {
            "#": 4,
            "지표명": "Buffer Ratio",
            "값": f"{buf_val:.1f}%",
            "기준": "15% ~ 30%",
            "판정": _judge("buffer_ratio_pct", buf_val),
            "설명": "리스크 버퍼 비율 (L1 기준)",
        },
        {
            "#": 5,
            "지표명": "Interaction Turns",
            "값": f"{turns_val}회",
            "기준": "5 ~ 60회",
            "판정": _judge("interaction_turns", turns_val),
            "설명": "총 토론 메시지 수",
        },
        {
            "#": 6,
            "지표명": "Supervisor 개입율",
            "값": f"{sup_val:.2%}",
            "기준": "≤ 40%",
            "판정": _judge("supervisor_intervention_ratio", sup_val),
            "설명": "슈퍼바이저 메시지 비율",
        },
        {
            "#": 7,
            "지표명": "Convergence",
            "값": "수렴" if conv_val else "미수렴",
            "기준": "수렴",
            "판정": _judge("convergence", conv_val),
            "설명": "토론 합의 수렴 여부",
        },
    ]

    df_summary = pd.DataFrame(summary_rows).set_index("#")

    # 판정별 행 색상을 위해 스타일 적용
    def _color_judge(val):
        if "✅" in str(val):
            return "background-color: #d1fae5; color: #065f46; font-weight:600;"
        elif "⚠️" in str(val):
            return "background-color: #fef3c7; color: #92400e; font-weight:600;"
        elif "❌" in str(val):
            return "background-color: #fee2e2; color: #991b1b; font-weight:600;"
        return ""

    styled_summary = df_summary.style.applymap(_color_judge, subset=["판정"])
    st.dataframe(styled_summary, use_container_width=True, height=290)

    # ── 세부 지표 표 ──────────────────────────────────────────
    st.markdown("#### 세부 지표 (Sub-Metrics)")

    detail_rows = [
        # RAGAS
        {"지표 그룹": "RAGAS Faithfulness", "세부 항목": "근거 확인된 주장 수",  "값": ragas.get("supported_claims", 0),  "단위": "건"},
        {"지표 그룹": "RAGAS Faithfulness", "세부 항목": "전체 주장 수",          "값": ragas.get("total_claims", 0),      "단위": "건"},
        # Interaction Turns
        {"지표 그룹": "Interaction Turns",  "세부 항목": "참여 에이전트 수",      "값": turns.get("unique_agents", 0),     "단위": "명"},
        {"지표 그룹": "Interaction Turns",  "세부 항목": "에이전트별 발화 수",    "값": str(turns.get("messages_by_agent", {})), "단위": ""},
        # Supervisor
        {"지표 그룹": "Supervisor 개입",    "세부 항목": "슈퍼바이저 메시지 수",  "값": sup.get("supervisor_messages", 0), "단위": "건"},
        {"지표 그룹": "Supervisor 개입",    "세부 항목": "중재/결정 메시지 수",   "값": sup.get("mediation_decisions", 0), "단위": "건"},
        {"지표 그룹": "Supervisor 개입",    "세부 항목": "전체 메시지 수",        "값": sup.get("total_messages", 0),      "단위": "건"},
        # Success Rate
        {"지표 그룹": "Success Rate",       "세부 항목": "커버된 기능 수",        "값": sr.get("covered", 0),              "단위": "건"},
        {"지표 그룹": "Success Rate",       "세부 항목": "전체 필수 기능 수",     "값": sr.get("total_features", 0),       "단위": "건"},
        {"지표 그룹": "Success Rate",       "세부 항목": "미커버 기능",           "값": str(sr.get("uncovered_features", [])), "단위": ""},
        # Planning Score
        {"지표 그룹": "Planning Score",     "세부 항목": "평가된 배정 건수",
         "값": ps.get("num_assignments_evaluated", 0) if ps_val >= 0 else "N/A", "단위": "건"},
        {"지표 그룹": "Planning Score",     "세부 항목": "최저 유사도 Top5",
         "값": (", ".join(f"{x['task_id']} ({x['similarity']:.4f})" for x in ps.get("top_5_lowest", []))
                or (ps.get("error", "데이터 없음") if ps_val < 0 else "데이터 없음")),
         "단위": ""},
        # Buffer Ratio
        {"지표 그룹": "Buffer Ratio",       "세부 항목": "총 예상 작업일",        "값": buf.get("total_estimated_days", 0), "단위": "일"},
        {"지표 그룹": "Buffer Ratio",       "세부 항목": "총 버퍼 일수",          "값": buf.get("total_buffer_days", 0),   "단위": "일"},
        {"지표 그룹": "Buffer Ratio",       "세부 항목": "L1 태스크 수",          "값": buf.get("l1_task_count", 0),       "단위": "건"},
        # Convergence
        {"지표 그룹": "Convergence",        "세부 항목": "수렴 추세 (trend)",     "값": conv.get("convergence_trend", 0.0), "단위": ""},
        {"지표 그룹": "Convergence",        "세부 항목": "전반부 평균 변화량",    "값": conv.get("first_half_avg_delta", 0.0), "단위": ""},
        {"지표 그룹": "Convergence",        "세부 항목": "후반부 평균 변화량",    "값": conv.get("second_half_avg_delta", 0.0), "단위": ""},
        {"지표 그룹": "Convergence",        "세부 항목": "버퍼 제안 횟수",        "값": conv.get("proposal_count", 0),     "단위": "회"},
    ]

    df_detail = pd.DataFrame(detail_rows)
    st.dataframe(df_detail, use_container_width=True, hide_index=True)

    # ── 실행 이력 표 ──────────────────────────────────────────
    history_path = os.path.join(
        os.environ.get("OUTPUT_DIR", "./generated"),
        "metrics_history.csv",
    )
    if os.path.isfile(history_path):
        st.markdown("#### 실행 이력 비교 (최근 10회)")
        try:
            df_hist = pd.read_csv(history_path, encoding="utf-8")
            display_cols = [
                "timestamp", "project_name", "model_backend", "total_tasks", "debate_rounds",
                "ragas_faithfulness", "success_rate", "planning_score",
                "buffer_ratio_pct", "interaction_turns",
                "supervisor_intervention_ratio", "convergence_is_converging",
                "judge_ragas_faithfulness", "judge_success_rate", "judge_planning_score",
                "judge_buffer_ratio_pct", "judge_interaction_turns",
                "judge_supervisor_intervention_ratio", "judge_convergence",
            ]
            # 존재하는 컬럼만 선택
            display_cols = [c for c in display_cols if c in df_hist.columns]
            df_show = df_hist[display_cols].tail(10).reset_index(drop=True)
            df_show.index += 1  # 1-based row index

            # 판정 컬럼 색상 적용
            judge_cols = [c for c in display_cols if c.startswith("judge_")]

            def _color_judge_cell(val):
                if "✅" in str(val):
                    return "background-color: #d1fae5; color: #065f46; font-weight:600;"
                elif "⚠️" in str(val):
                    return "background-color: #fef3c7; color: #92400e; font-weight:600;"
                elif "❌" in str(val):
                    return "background-color: #fee2e2; color: #991b1b; font-weight:600;"
                return ""

            styled_hist = df_show.style.applymap(_color_judge_cell, subset=judge_cols) if judge_cols else df_show.style
            st.dataframe(styled_hist, use_container_width=True)
        except Exception as e:
            st.warning(f"이력 파일 읽기 오류: {e}")

    with st.expander("📋 원본 JSON 상세"):
        st.json(metrics_data)


def _render_gantt_chart(tasks: List, member_map: dict, key_suffix: str = ""):
    """
    Primavera P6 스타일 간트 차트.
    - L1/L2 bar: 자식 범위로 roll-up, 두꺼운 배경 바
    - L3 bar: 담당자 색상, 높이 차등
    - 의존성: FS 꺾임 커넥터 (horizontal → vertical → horizontal+arrow)
    - 중요도 표시: High 태스크에 강조 테두리
    """
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    from agents.supervisor_agent import _rollup_parent_dates

    BASE_DATE = datetime(2024, 1, 1)

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        level_filter = st.radio(
            "표시 레벨",
            ["전체", "L1만", "L1 + L2"],
            index=0,
            horizontal=True,
            key=f"gantt_level_filter{key_suffix}",
        )
    with col_ctrl2:
        show_deps = st.checkbox("의존성 표시", value=True, key=f"gantt_deps{key_suffix}")

    def _sort_key(task):
        parts = task.task_id.split("-")
        try:
            return (int(parts[1]) if len(parts) > 1 else 0,
                    int(parts[2]) if len(parts) > 2 else 0,
                    int(parts[3]) if len(parts) > 3 else 0)
        except Exception:
            return (99, 99, 99)

    # ── Pass 1: roll-up 적용 ────────────────────────────────────────────────
    tasks_rolled = _rollup_parent_dates(list(tasks))

    # ── Pass 2: 필터 + 행 구성 ──────────────────────────────────────────────
    # Primavera 스타일 담당자 색상 팔레트 (채도 높고 구분 명확)
    ASSIGNEE_COLORS = [
        "#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED",
        "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#6366F1",
    ]
    assignee_color_map: dict = {}
    color_idx = 0

    # Primavera P6 스타일: L1=진한 회색 전체 폭 바, L2=중간 바, L3=얇은 실선 바
    BAR_HEIGHT = {"L1": 0.7, "L2": 0.55, "L3": 0.4}
    BAR_OPACITY = {"L1": 0.25, "L2": 0.45, "L3": 0.85}
    BAR_COLOR = {"L1": "#1E293B", "L2": "#475569"}

    rows = []
    label_order = []
    importance_map: dict = {}  # label → importance

    for t in sorted(tasks_rolled, key=_sort_key):
        if t.start_week is None or t.end_week is None:
            continue
        level_val = t.level.value if hasattr(t.level, "value") else str(t.level)
        if level_filter == "L1만" and level_val != "L1":
            continue
        if level_filter == "L1 + L2" and level_val == "L3":
            continue

        sw = max(1, int(t.start_week))
        ew = max(sw + 1, int(t.end_week))
        start_dt = BASE_DATE + timedelta(weeks=sw - 1)
        end_dt   = BASE_DATE + timedelta(weeks=ew - 1)

        assignee_names = [member_map.get(str(mid), str(mid)) for mid in (t.assigned_to or []) if mid]
        assignee = assignee_names[0] if assignee_names else ""

        if level_val == "L3" and assignee:
            if assignee not in assignee_color_map:
                assignee_color_map[assignee] = ASSIGNEE_COLORS[color_idx % len(ASSIGNEE_COLORS)]
                color_idx += 1

        # Primavera 스타일 레이블: L1 CAPS, L2 들여쓰기, L3 더 들여쓰기
        if level_val == "L1":
            label = f"■ {t.task_id}  {t.title[:30].upper()}{'…' if len(t.title) > 30 else ''}"
        elif level_val == "L2":
            label = f"  ▶ [{t.task_id}] {t.title[:28]}{'…' if len(t.title) > 28 else ''}"
        else:
            label = f"      [{t.task_id}] {t.title[:25]}{'…' if len(t.title) > 25 else ''}"

        imp = getattr(t, "importance", "Medium") or "Medium"
        rows.append({
            "label": label,
            "start": start_dt,
            "end": end_dt,
            "assignee": assignee,
            "level": level_val,
            "task_id": t.task_id,
            "역할": t.assigned_role or "",
            "총 일수": t.total_days,
            "버퍼": t.buffer_days,
            "dependencies": t.dependencies or [],
            "importance": imp,
        })
        label_order.append(label)
        importance_map[label] = imp

    if not rows:
        st.info("간트 차트를 그릴 일정 데이터가 없습니다. (start_week/end_week 미설정)")
        return

    # ── Pass 3: Figure 구성 ─────────────────────────────────────────────────
    fig = go.Figure()

    # y축 카테고리 순서 설정 (autorange reversed)
    y_categories = list(reversed(label_order))

    legend_added = set()
    for row in rows:
        lv = row["level"]
        imp = row["importance"]

        if lv == "L3":
            bar_color = assignee_color_map.get(row["assignee"], "#94A3B8")
            legend_name = row["assignee"] or "미배정"
        else:
            bar_color = BAR_COLOR.get(lv, "#475569")
            legend_name = f"{lv} (요약)"

        show_in_legend = legend_name not in legend_added
        if show_in_legend:
            legend_added.add(legend_name)

        # High 중요도: 테두리 강조
        if imp == "High" and lv == "L3":
            border_color = "rgba(220,38,38,0.9)"
            border_width = 2.0
        elif imp == "Low":
            border_color = "rgba(0,0,0,0.08)"
            border_width = 0.5
        else:
            border_color = "rgba(0,0,0,0.15)"
            border_width = 0.8

        hover_text = (
            f"<b>{row['label'].strip()}</b><br>"
            f"기간: {row['start'].strftime('%Y/%m/%d')} ~ {row['end'].strftime('%Y/%m/%d')}<br>"
            f"담당: {row['assignee'] or '-'} ({row['역할']})<br>"
            f"총 일수: {row['총 일수']}일 (예상 {row['총 일수'] - row['버퍼']}일 + 버퍼 {row['버퍼']}일)<br>"
            f"중요도: {imp}"
        )

        fig.add_trace(go.Bar(
            x=[(row["end"] - row["start"]).total_seconds() * 1000],
            y=[row["label"]],
            base=[row["start"]],
            orientation="h",
            name=legend_name,
            showlegend=show_in_legend,
            legendgroup=legend_name,
            width=BAR_HEIGHT.get(lv, 0.4),
            marker=dict(
                color=bar_color,
                opacity=BAR_OPACITY.get(lv, 0.85),
                line=dict(color=border_color, width=border_width),
            ),
            hovertemplate=hover_text + "<extra></extra>",
        ))

        # L3 High 중요도: 오른쪽 끝에 작은 마름모 마커 추가
        if imp == "High" and lv == "L3":
            fig.add_trace(go.Scatter(
                x=[row["end"]],
                y=[row["label"]],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="#DC2626", line=dict(color="white", width=1)),
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── Pass 4: FS 의존성 꺾임 커넥터 (Primavera P6 스타일) ─────────────────
    if show_deps:
        task_id_to_row = {r["task_id"]: r for r in rows}
        label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}
        visible_ids = {r["task_id"] for r in rows}
        ARROW_COLOR = "rgba(220, 38, 38, 0.75)"
        ARROW_WIDTH = 1.8

        for row in rows:
            for dep_id in row["dependencies"]:
                if dep_id not in visible_ids:
                    continue
                dep_row = task_id_to_row.get(dep_id)
                if not dep_row:
                    continue

                x_start = dep_row["end"]   # 선행 태스크 끝
                x_end   = row["start"]     # 후행 태스크 시작
                y_src   = dep_row["label"]
                y_dst   = row["label"]

                # 같은 행: 단순 수평 화살표
                if y_src == y_dst:
                    fig.add_annotation(
                        x=x_end, y=y_dst,
                        ax=x_start, ay=y_src,
                        xref="x", yref="y", axref="x", ayref="y",
                        text="", showarrow=True,
                        arrowhead=3, arrowsize=1.0, arrowwidth=ARROW_WIDTH,
                        arrowcolor=ARROW_COLOR,
                    )
                    continue

                # 꺾임 커넥터: x_start → mid_x → x_end (elbow connector)
                # mid_x: 선행 끝 + 1일 오프셋 (너무 붙으면 겹침)
                offset = timedelta(days=1)
                mid_x = x_start + offset

                # Segment 1: 선행 끝 → mid_x (수평, 선행 y레벨)
                fig.add_shape(
                    type="line",
                    x0=x_start, y0=y_src, x1=mid_x, y1=y_src,
                    xref="x", yref="y",
                    line=dict(color=ARROW_COLOR, width=ARROW_WIDTH),
                )
                # Segment 2: mid_x 에서 후행 y레벨로 수직 연결
                fig.add_shape(
                    type="line",
                    x0=mid_x, y0=y_src, x1=mid_x, y1=y_dst,
                    xref="x", yref="y",
                    line=dict(color=ARROW_COLOR, width=ARROW_WIDTH),
                )
                # Segment 3: mid_x → x_end (수평, 후행 y레벨) + 화살표
                fig.add_annotation(
                    x=x_end, y=y_dst,
                    ax=mid_x, ay=y_dst,
                    xref="x", yref="y", axref="x", ayref="y",
                    text="", showarrow=True,
                    arrowhead=3, arrowsize=1.0, arrowwidth=ARROW_WIDTH,
                    arrowcolor=ARROW_COLOR,
                )

    # ── Pass 5: 레이아웃 (Primavera P6 스타일) ──────────────────────────────
    # 주차 그리드 라인 (수직)
    total_weeks = max((r["end"] for r in rows), default=BASE_DATE) - BASE_DATE
    total_weeks_n = int(total_weeks.days / 7) + 2

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            type="date",
            tickformat="%m/%d",
            dtick="W1",           # 주 단위 눈금
            tickangle=-45,
            title_text="",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.25)",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#CBD5E1",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=y_categories,
            tickfont=dict(size=10, family="monospace"),
            showgrid=True,
            gridcolor="rgba(241,245,249,1)",
            gridwidth=1,
            zeroline=False,
            showline=False,
        ),
        margin=dict(l=20, r=20, t=48, b=20),
        height=max(480, len(rows) * 38 + 140),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="white",
        font=dict(family="Pretendard, 'Apple SD Gothic Neo', sans-serif", size=11),
        title=dict(
            text="📅 프로젝트 간트 차트 (Finish-to-Start 의존성)",
            font=dict(size=14, color="#1E293B"),
            x=0, xanchor="left",
            pad=dict(l=10, t=8),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def _render_wbs_result(wbs_output, is_draft=False, key_suffix=""):
    """WBS 결과 렌더링"""
    from schemas.wbs_schema import WBSLevel
    import html as _html
    prd_name = wbs_output.project_name
    tasks = wbs_output.tasks

    title_prefix = "🕒 [DRAFT]" if is_draft else "📊"
    st.markdown(
        f'<div class="section-title">{title_prefix} {_html.escape(prd_name)} — 계층적 WBS 상세</div>',
        unsafe_allow_html=True
    )

    # ─── 메트릭 카드 ────────────────────────────────────────
    # L3 (leaf) 태스크 기준 집계 — 버퍼는 L3에만 적용되므로 정확한 합계
    _l3_tasks = [t for t in tasks if (t.level.value if hasattr(t.level, 'value') else str(t.level)) == "L3"]
    _l1_tasks = [t for t in tasks if (t.level.value if hasattr(t.level, 'value') else str(t.level)) == "L1"]

    # 총 기간: L3 합산 기준 (roll-up 후 L1도 동일하나, L3가 최신값)
    total_weeks = wbs_output.total_weeks
    if total_weeks < 1.0 and _l3_tasks:
        total_weeks = round(sum(t.total_days for t in _l3_tasks) / 5, 1)
    elif total_weeks < 1.0 and _l1_tasks:
        total_weeks = round(sum(t.total_days for t in _l1_tasks) / 5, 1)

    # 버퍼·예상: L3 집계 (토론에서 직접 업데이트된 값)
    total_buffer = round(sum(t.buffer_days for t in _l3_tasks), 1) if _l3_tasks else round(sum(t.buffer_days for t in tasks), 1)
    total_base   = round(sum(t.estimated_days for t in _l3_tasks), 1) if _l3_tasks else round(sum(t.estimated_days for t in tasks), 1)
    total_tasks = len(tasks)
    buffer_pct = round(total_buffer / (total_base + 0.001) * 100, 1) if total_base > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, f"{total_weeks}주", "총 프로젝트 기간"),
        (col2, str(total_tasks), "총 태스크 수"),
        (col3, f"{total_buffer:.1f}일", "총 버퍼 일수"),
        (col4, f"{buffer_pct}%", "버퍼 비율"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ─── R&R / 간트 / 다운로드 / 요약: 최종 결과에서만 렌더링 ───
    if is_draft:
        # 초안에서는 interactive widget 없이 테이블만 표시
        return

    # ─── R&R 배분 현황 패널 ──────────────────────────────
    import html as html_lib
    import streamlit.components.v1 as components

    team_members = st.session_state.get("team_members", [])
    member_map = {m.member_id: m.name for m in team_members}
    calling_context = st.session_state.get("calling_context", {})  # {agent_role: member_id}
    # 역방향 맵: member_id -> 어떤 agent_role 담당했는지
    member_agent_role_map = {v: k for k, v in calling_context.items()}

    # 역할별 색상
    role_colors = {
        "planner": "#0891b2", "frontend": "#d97706", "backend": "#7c3aed",
        "designer": "#db2777", "qa": "#dc2626", "pm": "#4f46e5",
    }
    role_labels = {
        "planner": "📅 플래너", "frontend": "💻 FE 개발자", "backend": "⚙️ BE 개발자",
        "designer": "🎨 디자이너", "qa": "🔍 QA 엔지니어", "pm": "🎯 PM",
    }

    with st.expander("👥 R&R 배분 현황 (팀원별 담당 태스크)", expanded=True):
        if not team_members:
            st.info("팀원 정보가 없습니다.")
        else:
            # 팀원별 배정 태스크 수집
            member_tasks: dict = {m.member_id: [] for m in team_members}
            for t in tasks:
                for mid in (t.assigned_to or []):
                    if mid in member_tasks:
                        member_tasks[mid].append(t)

            # calling_context 에이전트 역할 배지
            agent_badge_html = ""
            if calling_context:
                badges = []
                for role, mid in calling_context.items():
                    name = member_map.get(mid, mid)
                    color = role_colors.get(role.lower(), "#64748b")
                    label = role_labels.get(role.lower(), f"🤖 {role}")
                    badges.append(
                        f'<span style="display:inline-flex;align-items:center;gap:5px;'
                        f'background:{color}18;border:1px solid {color}40;border-radius:99px;'
                        f'padding:3px 10px;margin:3px;font-size:0.8rem;color:{color};font-weight:600;">'
                        f'{html_lib.escape(label)} → {html_lib.escape(name)}</span>'
                    )
                agent_badge_html = (
                    f'<div style="margin-bottom:1rem;">'
                    f'<div style="font-size:0.78rem;color:#64748b;text-transform:uppercase;'
                    f'letter-spacing:0.06em;margin-bottom:6px;">에이전트 역할 배정 (마지막 라운드)</div>'
                    f'{"".join(badges)}</div>'
                )

            # 팀원 카드 HTML 생성
            cards_html = []
            for m in team_members:
                assigned = member_tasks.get(m.member_id, [])
                agent_role = member_agent_role_map.get(m.member_id, "")
                color = role_colors.get(agent_role.lower(), "#475569")
                label = role_labels.get(agent_role.lower(), "")
                total_d = sum(t.total_days for t in assigned)
                task_items = "".join(
                    f'<div style="font-size:0.8rem;padding:3px 0;border-bottom:1px solid #f1f5f9;'
                    f'display:flex;justify-content:space-between;">'
                    f'<span style="color:#374151;">{html_lib.escape(t.task_id)} {html_lib.escape(t.title[:28])}{"…" if len(t.title)>28 else ""}</span>'
                    f'<span style="color:#6b7280;font-size:0.75rem;">{t.total_days}d</span>'
                    f'</div>'
                    for t in assigned[:8]
                ) or '<div style="color:#94a3b8;font-size:0.8rem;padding:4px 0;">배정된 태스크 없음</div>'
                overflow = f'<div style="font-size:0.75rem;color:#9ca3af;padding-top:3px;">+{len(assigned)-8}개 더...</div>' if len(assigned) > 8 else ""
                agent_badge = (
                    f'<span style="font-size:0.72rem;background:{color}18;color:{color};'
                    f'border:1px solid {color}40;border-radius:99px;padding:1px 8px;font-weight:600;">'
                    f'🤖 회의 역할: {html_lib.escape(label)}</span>'
                ) if label else ""

                workload_pct = min(100, int(total_d / max(total_weeks * 5, 1) * 100)) if total_d else 0
                card = (
                    f'<div style="background:white;border:1px solid #e8ecf4;border-radius:14px;'
                    f'padding:1rem;border-top:4px solid {color};">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">'
                    f'<div>'
                    f'<div style="font-weight:700;color:#1e293b;font-size:0.95rem;">{html_lib.escape(m.name)}</div>'
                    f'<div style="color:#64748b;font-size:0.8rem;">{html_lib.escape(getattr(m.role, "value", str(m.role)) if m.role else "팀원")}</div>'
                    f'</div>'
                    f'<div style="text-align:right;">{agent_badge}'
                    f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;">'
                    f'{len(assigned)}개 태스크 · {total_d:.0f}일</div></div></div>'
                    f'<div style="background:#f1f5f9;border-radius:99px;height:5px;margin-bottom:10px;">'
                    f'<div style="background:{color};width:{workload_pct}%;height:5px;border-radius:99px;"></div></div>'
                    f'{task_items}{overflow}'
                    f'</div>'
                )
                cards_html.append(card)

            n = len(cards_html)
            cols_per_row = min(3, n)
            rows = [cards_html[i:i+cols_per_row] for i in range(0, n, cols_per_row)]
            grid_rows = "".join(
                f'<div style="display:grid;grid-template-columns:repeat({len(row)},1fr);gap:1rem;margin-bottom:1rem;">'
                + "".join(row) + "</div>"
                for row in rows
            )
            full_html = (
                f'<div style="font-family:sans-serif;">'
                f'{agent_badge_html}'
                f'{grid_rows}'
                f'</div>'
            )
            components.html(full_html, height=max(300, 220 * ((n // cols_per_row) + 1) + (120 if calling_context else 0)), scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 WBS 계층 테이블</div>', unsafe_allow_html=True)

    # ─── WBS 테이블 ──────────────────────────────
    # member_map, team_members는 위 R&R 섹션에서 이미 선언됨

    table_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; font-family: 'Inter', -apple-system, sans-serif; background: #f8fafc; }

        .wbs-container {
            padding: 12px;
            background: #f8fafc;
        }
        /* 검색/필터 바 */
        .wbs-filter {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 8px 14px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: #64748b;
        }
        .level-pill {
            padding: 3px 12px;
            border-radius: 99px;
            font-size: 11px;
            font-weight: 700;
            cursor: default;
            border: 1px solid;
        }
        .pill-l1 { background: #fffbeb; color: #92400e; border-color: #fcd34d; }
        .pill-l2 { background: #eff6ff; color: #1d4ed8; border-color: #93c5fd; }
        .pill-l3 { background: #f0fdf4; color: #15803d; border-color: #86efac; }

        .wbs-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: white;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
        }
        .wbs-table thead {
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .wbs-table th {
            background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
            color: #c7d2fe;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 10.5px;
            padding: 13px 14px;
            border-bottom: none;
            text-align: left;
            letter-spacing: 0.07em;
            white-space: nowrap;
        }
        .wbs-table th:first-child { border-radius: 0; }
        .wbs-table td {
            padding: 11px 14px;
            border-bottom: 1px solid #f1f5f9;
            color: #334155;
            vertical-align: middle;
            transition: background 0.15s ease;
        }
        .wbs-table tr:last-child td { border-bottom: none; }

        /* ── L1: Phase ── */
        .lv-1 td {
            background: linear-gradient(135deg, #fffdf0 0%, #fefce8 100%);
            border-left: 5px solid #f59e0b;
            font-weight: 800;
            font-size: 0.875rem;
            color: #1c1917;
            letter-spacing: 0.01em;
        }
        .lv-1:hover td { background: linear-gradient(135deg, #fef9c3, #fefce8); }

        /* ── L2: 기능그룹 ── */
        .lv-2 td {
            background: #ffffff;
            border-left: 5px solid #3b82f6;
            font-weight: 600;
            color: #1e3a5f;
        }
        .lv-2 td.title-cell { padding-left: 2rem; }
        .lv-2:hover td { background: #f0f9ff; }

        /* ── L3: 세부 태스크 ── */
        .lv-3 td {
            background: #fafbfc;
            border-left: 5px solid #cbd5e1;
            font-size: 0.82rem;
            color: #475569;
        }
        .lv-3 td.title-cell {
            padding-left: 3.5rem;
            color: #374151;
            font-weight: 400;
        }
        .lv-3:hover td { background: #f1f5f9; border-left-color: #94a3b8; }

        /* ── 배지 ── */
        .rr-badge {
            display: inline-flex;
            align-items: center;
            padding: 2px 9px;
            border-radius: 99px;
            background: #f1f5f9;
            color: #475569;
            font-size: 10.5px;
            font-weight: 600;
            margin-top: 4px;
            border: 1px solid #e2e8f0;
            white-space: nowrap;
        }
        .week-badge {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            font-size: 10.5px;
            color: #6366f1;
            font-weight: 600;
            background: #eef2ff;
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid #e0e7ff;
            white-space: nowrap;
        }
        .id-cell {
            font-family: 'Courier New', monospace;
            font-size: 11px;
            font-weight: 700;
            color: #6366f1;
            letter-spacing: 0.02em;
        }
        .days-total {
            font-weight: 800;
            color: #4338ca;
            font-size: 0.9rem;
        }
        /* 행 구분을 위한 L1 그룹 separator */
        .lv-1 td:first-child::before {
            content: '';
        }
    </style>
    """

    # 통계 요약 바
    l1_count = sum(1 for t in tasks if getattr(t.level,'value',str(t.level))=='L1')
    l2_count = sum(1 for t in tasks if getattr(t.level,'value',str(t.level))=='L2')
    l3_count = sum(1 for t in tasks if getattr(t.level,'value',str(t.level))=='L3')

    filter_bar = f"""
    <div class="wbs-filter">
        <span style="font-weight:700;color:#1e293b;font-size:13px;">📊 WBS 구조 요약</span>
        <span style="margin:0 4px;color:#e2e8f0;">|</span>
        <span class="level-pill pill-l1">■ L1 Phase: {l1_count}개</span>
        <span class="level-pill pill-l2">▶ L2 기능그룹: {l2_count}개</span>
        <span class="level-pill pill-l3">◆ L3 세부태스크: {l3_count}개</span>
        <span style="margin-left:auto;color:#94a3b8;font-size:11px;">총 {len(tasks)}개 항목</span>
    </div>
    """

    table_header = f"""
    <div class="wbs-container">
    {filter_bar}
    <table class="wbs-table">
        <thead>
            <tr>
                <th style="width:8%;">ID</th>
                <th style="width:34%;">단계 / 태스크명</th>
                <th style="width:16%;">담당 (R&amp;R)</th>
                <th style="width:8%;">중요도</th>
                <th style="width:9%;">주차</th>
                <th style="width:8%;">예상</th>
                <th style="width:7%;">버퍼</th>
                <th style="width:10%;">합계</th>
            </tr>
        </thead>
        <tbody>
    """

    def _get_sort_key(task):
        parts = task.task_id.split("-")
        try:
            p1 = int(parts[1]) if len(parts) > 1 else 0
            p2 = int(parts[2]) if len(parts) > 2 else 0
            p3 = int(parts[3]) if len(parts) > 3 else 0
            return (p1, p2, p3)
        except Exception:
            return (99, 99, 99)

    rows_html = ""
    for t in sorted(tasks, key=_get_sort_key):
        level_val = getattr(t.level, 'value', str(t.level))
        if level_val == "L1":
            row_class = "lv-1"
            prefix = ""
        elif level_val == "L2":
            row_class = "lv-2"
            prefix = "└─ "
        else:
            row_class = "lv-3"
            prefix = "&nbsp;&nbsp;&nbsp;&nbsp;◆ "

        assignee_ids = t.assigned_to if isinstance(t.assigned_to, list) else [t.assigned_to]
        assignee_names = [member_map.get(str(mid), str(mid)) for mid in assignee_ids if mid]
        if level_val in ("L1", "L2"):
            name_str = ""
        else:
            name_str = html_lib.escape(", ".join(assignee_names) or "미배정")

        role_str = html_lib.escape(str(t.assigned_role)) if t.assigned_role else "담당자"
        req_role_str = html_lib.escape(str(getattr(t, 'required_role', '')))
        role_badge = f"<span class='rr-badge'>👤 {role_str}</span>" if role_str else ""
        if req_role_str and req_role_str != role_str:
            role_badge += (
                f"<div style='font-size:10px;color:#94a3b8;font-weight:500;margin-top:4px;"
                f"border-top:1px dashed #e2e8f0;padding-top:3px;'>📋 {req_role_str}</div>"
            )

        task_id_safe = html_lib.escape(str(t.task_id))
        title_safe = html_lib.escape(str(t.title))

        # 중요도 배지
        importance_val = getattr(t, 'importance', 'Medium') or 'Medium'
        _imp_styles = {
            "High":   ("🔴", "#dc2626", "#fef2f2", "#fca5a5"),
            "Medium": ("🟡", "#d97706", "#fffbeb", "#fcd34d"),
            "Low":    ("🟢", "#16a34a", "#f0fdf4", "#86efac"),
        }
        imp_icon, imp_color, imp_bg, imp_border = _imp_styles.get(importance_val, _imp_styles["Medium"])
        importance_badge = (
            f'<span style="display:inline-flex;align-items:center;gap:3px;font-size:10.5px;'
            f'font-weight:700;color:{imp_color};background:{imp_bg};border:1px solid {imp_border};'
            f'border-radius:99px;padding:2px 8px;white-space:nowrap;">'
            f'{imp_icon} {html_lib.escape(importance_val)}</span>'
        )

        # 주차 배지
        sw = getattr(t, 'start_week', None)
        ew = getattr(t, 'end_week', None)
        week_badge = ""
        if sw and ew:
            week_badge = f'<span class="week-badge">W{sw}→W{ew}</span>'

        # 이름 + 역할 셀
        assignee_cell = f"<div style='font-size:12px;font-weight:600;color:#1e293b;margin-bottom:2px;'>{name_str}</div>" if name_str else ""

        rows_html += f"""
        <tr class="{row_class}">
            <td><span class="id-cell">{task_id_safe}</span></td>
            <td class="title-cell">{prefix}{title_safe}</td>
            <td>{assignee_cell}{role_badge}</td>
            <td style="text-align:center;">{importance_badge}</td>
            <td style="text-align:center;">{week_badge}</td>
            <td style="text-align:center;font-size:12px;">{t.estimated_days}d</td>
            <td style="text-align:center;font-size:12px;color:{'#dc2626' if t.buffer_days > 0 else '#94a3b8'};">{t.buffer_days}d</td>
            <td style="text-align:center;"><span class="days-total">{t.total_days}d</span></td>
        </tr>
        """

    full_table_html = table_css + table_header + rows_html + "</tbody></table></div>"
    table_height = min(800, max(300, 100 + len(tasks) * 50))
    components.html(full_table_html, height=table_height, scrolling=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ─── 간트 차트 ───────────────────────────────
    st.markdown('<div class="section-title">📅 프로젝트 간트 차트</div>', unsafe_allow_html=True)
    _render_gantt_chart(tasks, member_map, key_suffix=key_suffix)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ─── 다운로드 버튼 ────────────────────────────
    st.markdown("### 💾 산출물 다운로드")
    col_d1, col_d2 = st.columns(2)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
    wbs_md_path = os.path.join(output_dir, "wbs_output.md")
    log_md_path = os.path.join(output_dir, "debate_log.md")

    with col_d1:
        if os.path.exists(wbs_md_path):
            with open(wbs_md_path, encoding="utf-8") as f:
                st.download_button(
                    "📋 WBS 다운로드 (Markdown)",
                    f.read(),
                    file_name=f"{prd_name}_WBS.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_wbs{key_suffix}"
                )
    with col_d2:
        if os.path.exists(log_md_path):
            with open(log_md_path, encoding="utf-8") as f:
                st.download_button(
                    "💬 토론 로그 다운로드",
                    f.read(),
                    file_name=f"{prd_name}_토론로그.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_log{key_suffix}"
                )

    # ─── 요약 ─────────────────────────────────────
    if wbs_output.summary:
        with st.expander("📝 생성 요약"):
            st.markdown(wbs_output.summary)


if __name__ == "__main__":
    main()
