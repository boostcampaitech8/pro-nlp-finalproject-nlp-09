
# app.py

import html
import textwrap

import re

import json

from datetime import date

import time

import streamlit as st

import requests

from urllib.parse import urlparse

from google import genai

from google.genai.types import HttpOptions

import plotly.graph_objects as go

import streamlit.components.v1 as components



# 프로젝트 ID와 지역 설정 (본인의 환경에 맞게 수정)

PROJECT_ID = "project-5b75bb04-485d-454e-af7"

LOCATION = "asia-northeast3"

MODEL_NAME = "gemini-2.5-flash"

GENAI_CLIENT = genai.Client(

    vertexai=True,

    project=PROJECT_ID,

    location=LOCATION,

    http_options=HttpOptions(api_version="v1"),

)



def compress_report_text(report_text: str) -> str:

    if not report_text:

        return ""

    lines = [ln.strip() for ln in report_text.splitlines() if ln.strip()]

    keywords = (

        "종합", "결론", "요약", "전망", "리스크", "상승", "하락",

        "확률", "신뢰도", "예측", "뉴스", "감성", "지표", "가격", "%"

    )

    picked = []

    for ln in lines:

        is_heading = ln.startswith("#") or ln.endswith(":")

        has_number = any(ch.isdigit() for ch in ln) or "%" in ln

        has_kw = any(k in ln for k in keywords)

        if is_heading or has_number or has_kw:

            picked.append(ln)

    compact = "\n".join(picked)

    max_chars = 300

    return compact[-max_chars:]



def compress_question_text(question: str) -> str:

    if not question:

        return ""

    q = " ".join(question.split())

    max_len = 120

    return q[:max_len]

def get_gemini_3_response(report_text, user_question):

    # 1. 리포트 핵심 압축 사용 (과도한 토큰 사용 방지)

    full_context = compress_report_text(report_text)

    short_question = compress_question_text(user_question)



    system_instruction = (

        "너는 금융 데이터 전문 분석가다. 제공된 리포트 수치로만 답한다. "

        "답변은 3~5줄로 짧게, 결론→근거(수치 1~2개)→리스크 순서."

    )



    # 3. 답변 생성

    def _call_model(context_text: str, question_text: str, max_tokens: int, system_text: str):

        # 기본 재시도 (429 완화)

        delays = [1.5, 3.0, 6.0]

        last_err = None

        for i in range(len(delays) + 1):

            try:

                return GENAI_CLIENT.models.generate_content(

                    model=MODEL_NAME,

                    contents=[system_text, f"[시장 리포트]\n{context_text}\n\n질문: {question_text}"],

                    config={

                        "temperature": 0.2,

                        "max_output_tokens": max_tokens,

                    },

                )

            except Exception as e:

                last_err = e

                if i < len(delays):

                    time.sleep(delays[i])

        raise last_err



    response = _call_model(full_context, short_question, 512, system_instruction)



    # 응답 추출 (빈 응답/차단 케이스 방어)

    def _extract_text(resp):

        try:

            if getattr(resp, "text", None):

                return resp.text

        except Exception:

            pass

        try:

            candidates = getattr(resp, "candidates", []) or []

            for cand in candidates:

                content = getattr(cand, "content", None)

                parts = getattr(content, "parts", None) if content else None

                if parts:

                    texts = [getattr(p, "text", None) for p in parts if getattr(p, "text", None)]

                    if texts:

                        return "\n".join(texts)

        except Exception:

            pass

        return None



    text = _extract_text(response)

    if text:

        return text



    # 2차 시도: 컨텍스트 더 축소 + 출력 토큰 줄이기

    retry_context = full_context[-300:] if full_context else ""

    retry_question = short_question[:80]

    retry_system = (

        system_instruction

        + " 길이 제한이 있으니 핵심만 3줄로 요약해서 답해."

    )

    response_retry = _call_model(retry_context, retry_question, 256, retry_system)

    text_retry = _extract_text(response_retry)

    if text_retry:

        return text_retry



    # 빈 응답 진단 메시지 구성

    # 빈 응답 진단 메시지

    finish_reason = None

    try:

        cand0 = (getattr(response_retry, "candidates", []) or [None])[0]

        if cand0:

            finish_reason = getattr(cand0, "finish_reason", None)

    except Exception:

        pass

    return f"응답이 비어있어요. finish_reason={finish_reason}"

def _parse_gcs_url(u: str):

    if u.startswith("gs://"):

        parts = u.replace("gs://", "").split("/", 1)

        return parts[0], parts[1] if len(parts) > 1 else ""

    parsed = urlparse(u)

    if parsed.netloc in ("storage.googleapis.com", "storage.cloud.google.com"):

        path = parsed.path.lstrip("/")

        parts = path.split("/", 1)

        return parts[0], parts[1] if len(parts) > 1 else ""

    return None, None



def _parse_sections(report_text: str):

    if not report_text:

        return []

    lines = report_text.splitlines()

    sections = []

    current_title = "보고서 본문"

    current_lines = []

    header_pattern = re.compile(

        r"^\s*(?:[-*]\s*)?(?:#{1,6}\s*)?(?:\*\*)?\s*(?:\d+\.|\d+\)|[IVX]+\.)\s*(.+?)\s*$"

    )

    alt_headers = (

        "종합 의견",

        "뉴스 분석",

        "뉴스 빅데이터",

        "시장 심리",

        "인사이트",

        "Insight",

    )

    for ln in lines:

        m = header_pattern.match(ln)

        if m or any(h in ln for h in alt_headers):

            if current_lines:

                sections.append((current_title, "\n".join(current_lines).strip()))

                current_lines = []

            current_title = (m.group(1).strip() if m else ln.strip())

            continue

        current_lines.append(ln)

    if current_lines:

        sections.append((current_title, "\n".join(current_lines).strip()))

    return sections



def _infer_section_key(title: str, body: str):

    text = f"{title}\n{body}".lower()

    if any(k in text for k in ["퀀트", "quant", "기술적", "모델", "예측"]):

        return "quant"

    if any(k in text for k in ["뉴스", "news", "심리", "빅데이터", "인사이트", "insight"]):

        return "news"

    if any(k in text for k in ["종합", "결론", "의견", "summary", "outlook"]):

        return "summary"

    return None



def _extract_header_lines(report_text: str):

    if not report_text:

        return []

    lines = report_text.splitlines()

    header_pattern = re.compile(

        r"^\s*(?:[-*]\s*)?(?:#{1,6}\s*)?(?:\*\*)?\s*(?:\d+\.|\d+\)|[IVX]+\.)\s*.+$"

    )

    alt_headers = (

        "종합 의견",

        "뉴스 분석",

        "뉴스 빅데이터",

        "시장 심리",

        "인사이트",

        "Insight",

    )

    headers = []

    for ln in lines:

        if header_pattern.match(ln) or any(h in ln for h in alt_headers):

            headers.append(ln.strip())

    return headers



def _find_header_line(report_text: str, key: str):

    if not report_text:

        return ""

    lines = report_text.splitlines()

    fixed_headers = {

        "quant": "1. 📈 [Quant] 퀀트 기반 기술적 분석",

        "news": "2. 📰 [Insight] 뉴스 빅데이터 기반 시장 심리 분석",

        "summary": "3. 종합 의견",

    }

    if key in fixed_headers:

        target = fixed_headers[key]

        if target in report_text:

            return target

    key_map = {

        "quant": ["퀀트", "Quant", "기술적", "모델", "예측"],

        "news": ["뉴스", "News", "심리", "빅데이터", "Insight", "인사이트"],

        "summary": ["종합 의견", "종합", "결론", "의견", "Summary", "Outlook"],

    }

    targets = key_map.get(key, [])

    for ln in lines:

        if any(t in ln for t in targets):

            return ln.strip()

    return ""



def _extract_summary_row(report_text: str):

    if not report_text:

        return None

    lines = [ln.rstrip() for ln in report_text.splitlines() if ln.strip()]

    header_idx = None

    for i, ln in enumerate(lines):

        if "|" in ln and all(k in ln for k in ["어제", "시계열", "머신러닝", "시장 심리", "종합 의견"]):

            header_idx = i

            break

        if all(k in ln for k in ["어제", "시계열", "머신러닝", "시장 심리", "종합 의견"]):

            header_idx = i

            break

    if header_idx is not None:

        for j in range(header_idx + 1, min(header_idx + 4, len(lines))):

            row = lines[j]

            if re.match(r"^\s*[\-|:\s|]+\s*$", row):

                continue

            if "|" in row:

                cells = [c.strip() for c in row.strip("|").split("|")]

                cells = [c for c in cells if c]

                if len(cells) >= 5:

                    return {

                        "close": cells[0],

                        "ts_forecast": cells[1],

                        "ml_dir": cells[2],

                        "sentiment": cells[3].replace(" ", ""),

                        "opinion": cells[4].replace(" ", ""),

                    }

            else:

                parts = re.split(r"\s+", row)

                if len(parts) >= 5:

                    return {

                        "close": parts[0],

                        "ts_forecast": parts[1],

                        "ml_dir": parts[2],

                        "sentiment": parts[3],

                        "opinion": parts[4],

                    }

    m = re.search(

        r"어제 종가.*?\n\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([A-Za-z가-힣]+)\s+([가-힣]+)\s+(BUY|SELL|HOLD)",

        report_text,

        re.S,

    )

    if m:

        return {

            "close": m.group(1),

            "ts_forecast": m.group(2),

            "ml_dir": m.group(3),

            "sentiment": m.group(4),

            "opinion": m.group(5),

        }

    return None



def _extract_opinion(report_text: str):

    row = _extract_summary_row(report_text)

    if row and row.get("opinion"):

        return row["opinion"].upper()

    if not report_text:

        return "HOLD"

    text_upper = report_text.upper()

    if "SELL" in text_upper or "매도" in report_text:

        return "SELL"

    if "BUY" in text_upper or "매수" in report_text:

        return "BUY"

    if "HOLD" in text_upper or "중립" in report_text or "유지" in report_text:

        return "HOLD"

    return "HOLD"



def _extract_ts_forecast(report_text: str):

    row = _extract_summary_row(report_text)

    if row and row.get("ts_forecast"):

        return row["ts_forecast"]

    m = re.search(r"시계열 분석 예측값[^0-9\-]*([\-]?\d+\.?\d*)", report_text)

    if m:

        return m.group(1)

    return None



def _extract_ml_direction(report_text: str):

    row = _extract_summary_row(report_text)

    if row and row.get("ml_dir"):

        return row["ml_dir"]

    m = re.search(r"머신러닝 (?:방향 )?예측[^가-힣a-zA-Z]*([가-힣a-zA-Z]+)", report_text)

    if m:

        token = m.group(1)

        if "상승" in token or "UP" in token.upper():

            return "Up"

        if "하락" in token or "DOWN" in token.upper():

            return "Down"

    return None



def _extract_sentiment_score(report_text: str):

    row = _extract_summary_row(report_text)

    if row and row.get("sentiment"):

        label = row["sentiment"]

    else:

        m = re.search(r"시장 심리[^가-힣]*([가-힣]+)", report_text)

        label = m.group(1) if m else ""

    if "부정" in label:

        return -55

    if "긍정" in label:

        return 55

    if "중립" in label or "유지" in label:

        return 0

    return 0



def _extract_sentiment_label(report_text: str):

    row = _extract_summary_row(report_text)

    label = row.get("sentiment") if row else ""

    if not label:

        m = re.search(r"시장 심리[^가-힣]*([가-힣]+)", report_text)

        label = m.group(1) if m else ""

    if "부정" in label:

        return "부정적"

    if "긍정" in label:

        return "긍정적"

    if "중립" in label or "유지" in label:

        return "중립적"

    return "N/A"



def _parse_price_table(section_text: str):

    if not section_text:

        return None

    rows = []

    for ln in section_text.splitlines():

        m = re.match(

            r"^\s*(\d{4}[-./]\d{1,2}[-./]\d{1,2})\s+([\-]?\d+\.?\d*)\s+([\-]?\d+\.?\d*)\s+([\-]?\d+\.?\d*)",

            ln,

        )

        if m:

            rows.append(

                {

                    "date": m.group(1),

                    "d0": float(m.group(2)),

                    "d1": float(m.group(3)),

                    "d3": float(m.group(4)),

                }

            )

    return rows or None



def _parse_components(report_text: str):

    if not report_text:

        return None

    items = {

        "추세": None,

        "연간": None,

        "주기": None,

        "주간": None,

        "잔차": None,

    }

    for key in list(items.keys()):

        m = re.search(rf"{key}\s*[:=]\s*([\-]?\d+\.?\d*)", report_text)

        if m:

            try:

                items[key] = float(m.group(1))

            except ValueError:

                items[key] = None

    filtered = {k: v for k, v in items.items() if v is not None}

    if len(filtered) < 3:

        return None

    return filtered



def _extract_news_counts(report_text: str):

    if not report_text:

        return {"positive": 0, "negative": 0}

    lines = report_text.splitlines()

    pos = 0

    neg = 0

    mode = None

    for ln in lines:

        if "A-1. 긍정적인 뉴스" in ln:

            mode = "pos"

            continue

        if "A-2. 부정적인 뉴스" in ln:

            mode = "neg"

            continue

        if re.match(r"^\s*[A-C]\.\s", ln):

            mode = None

        if mode and re.match(r"^\s*\d+\s+", ln):

            if mode == "pos":

                pos += 1

            elif mode == "neg":

                neg += 1

    return {"positive": pos, "negative": neg}



def _highlight_keywords(text: str):

    if not text:

        return ""

    neg_keywords = [

        "하방 압력",

        "부정적",

        "급락",

        "하락",

        "위험",

        "리스크",

        "불확실",

        "약세",

    ]

    pos_keywords = [

        "상방 압력",

        "긍정적",

        "급등",

        "상승",

        "강세",

        "호재",

        "개선",

        "기대",

    ]

    escaped = html.escape(text)

    if pos_keywords:

        pos_pattern = re.compile("|".join(map(re.escape, pos_keywords)))

        escaped = pos_pattern.sub(r"<span class='highlight-pos'>\g<0></span>", escaped)

    if neg_keywords:

        neg_pattern = re.compile("|".join(map(re.escape, neg_keywords)))

        escaped = neg_pattern.sub(r"<span class='highlight-neg'>\g<0></span>", escaped)

    return escaped



st.set_page_config(page_title="Finance Agent", layout="wide")



st.markdown(

    """

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Source+Serif+4:wght@400;600;700&display=swap');

    :root {

        --paper: #ffffff;

        --ink: #0f172a;

        --muted: #64748b;

        --accent: #0ea5e9;

        --edge: #e2e8f0;

    }

    [data-testid="stAppViewContainer"] {

        background: #ffffff;

        font-family: "Inter", sans-serif;

        color: var(--ink);

    }

    .report-shell {

        background: linear-gradient(180deg, #f7f4ee 0%, #fbf9f4 100%);

        border: 1px solid var(--edge);

        border-radius: 16px;

        padding: 24px 28px;

        box-shadow: 0 10px 30px rgba(0,0,0,0.06);

        position: relative;

        overflow: hidden;

    }

    .report-shell:before {

        content: "";

        position: absolute;

        top: -60px;

        right: -80px;

        width: 220px;

        height: 220px;

        background: radial-gradient(circle at center, rgba(14,165,233,0.10), rgba(14,165,233,0.0) 70%);

        pointer-events: none;

    }

    .report-title {

        font-family: "Source Serif 4", serif;

        font-size: 24px;

        color: var(--ink);

        margin: 0 0 6px 0;

        letter-spacing: 0.2px;

    }

    .report-sub {

        font-family: "Inter", sans-serif;

        font-size: 12px;

        color: var(--muted);

        margin-bottom: 16px;

    }

    .report-body {

        font-family: "Source Serif 4", serif;

        font-size: 14px;

        line-height: 1.7;

        color: var(--ink);

        white-space: pre-wrap;

        margin: 0;

    }

    .summary-row {

        display: grid;

        grid-template-columns: 1.1fr 1fr 1fr;

        gap: 12px;

        margin-bottom: 16px;

    }

    .summary-card {

        background: #ffffff;

        border: 1px solid var(--edge);

        border-radius: 14px;

        padding: 14px 16px;

        box-shadow: 0 10px 24px rgba(0,0,0,0.06);

    }

    .summary-title {

        font-size: 12px;

        color: var(--muted);

        text-transform: uppercase;

        letter-spacing: 0.08em;

        margin-bottom: 8px;

        font-family: "Inter", sans-serif;

    }

    .opinion-badge {

        font-family: "Inter", sans-serif;

        font-size: 22px;

        padding: 6px 12px;

        border-radius: 999px;

        display: inline-flex;

        align-items: center;

        gap: 8px;

        font-weight: 600;

    }

    .opinion-buy {

        background: rgba(26, 188, 156, 0.12);

        color: #0f766e;

        border: 1px solid rgba(26, 188, 156, 0.45);

    }

    .opinion-sell {

        background: rgba(220, 38, 38, 0.12);

        color: #b91c1c;

        border: 1px solid rgba(220, 38, 38, 0.45);

    }

    .opinion-hold {

        background: rgba(245, 158, 11, 0.15);

        color: #92400e;

        border: 1px solid rgba(245, 158, 11, 0.5);

    }

    .summary-metric {

        font-size: 18px;

        color: var(--ink);

        font-weight: 600;

        margin-bottom: 4px;

    }

    .summary-sub {

        font-size: 12px;

        color: var(--muted);

    }

    .section-anchor {

        position: relative;

        top: -80px;

        height: 1px;

    }

    .section-title {

        font-family: "Inter", sans-serif;

        font-size: 18px;

        margin: 18px 0 6px 0;

        color: var(--ink);

    }

    .callout-box {

        background: #f1f5f9;

        border: 1px solid #e2e8f0;

        border-radius: 12px;

        padding: 10px 12px;

        font-size: 13px;

        color: #0f172a;

        margin: 10px 0 14px 0;

        font-family: "Inter", sans-serif;

    }

    .highlight-neg {

        background: rgba(239, 68, 68, 0.12);

        color: #7f1d1d;

        font-weight: 600;

        padding: 0 4px;

        border-radius: 6px;

    }

    .highlight-pos {

        background: rgba(34, 197, 94, 0.12);

        color: #14532d;

        font-weight: 600;

        padding: 0 4px;

        border-radius: 6px;

    }

    .chat-shell {

        background: #ffffff;

        border: 1px solid #ece8e1;

        border-radius: 16px;

        padding: 14px 14px 6px 14px;

        height: 560px;

        overflow-y: auto;

        box-shadow: 0 14px 32px rgba(0,0,0,0.10), 0 2px 8px rgba(0,0,0,0.06);

    }

    .chat-msg {

        display: flex;

        gap: 10px;

        margin-bottom: 12px;

    }

    .chat-role {

        width: 28px;

        height: 28px;

        border-radius: 8px;

        display: inline-flex;

        align-items: center;

        justify-content: center;

        font-size: 14px;

        flex: 0 0 28px;

    }

    .chat-role.user {

        background: #ffe4e1;

    }

    .chat-role.assistant {

        background: #ffe8c2;

    }

    .chat-bubble {

        background: #f6f6f6;

        border-radius: 12px;

        padding: 10px 12px;

        font-family: "Inter", sans-serif;

        font-size: 13px;

        line-height: 1.6;

        color: #1b1b1b;

        white-space: pre-wrap;

        flex: 1;

    }

    </style>

    """,

    unsafe_allow_html=True,

)



today = date.today()

commodity_map = {"옥수수": "corn", "밀": "wheat", "대두": "soybean"}

report_prefix_map = {

    "corn": "corn_report_",

    "wheat": "wheat_report_",

    "soybean": "soybean_report_",

}

default_commodity_label = "옥수수"

default_commodity = commodity_map[default_commodity_label]

default_url = (

    f"https://storage.cloud.google.com/team-blue-raw-data/reports/"

    f"{default_commodity}/{today:%Y}/{today:%m}/"

    f"{report_prefix_map[default_commodity]}{today:%Y-%m-%d}.txt"

)



with st.sidebar:

    st.header("📌 Report Controls")

    commodity_label = st.radio(

        "종목 선택",

        list(commodity_map.keys()),

        index=0,

        key="report_commodity_label",

        horizontal=True,

    )

    commodity = commodity_map[commodity_label]

    report_date = st.date_input("Report date", value=today)

    if "report_url" not in st.session_state:

        st.session_state.report_url = default_url

    if (

        report_date != st.session_state.get("last_report_date")

        or commodity != st.session_state.get("last_report_commodity")

    ):

        st.session_state.report_url = (

            f"https://storage.cloud.google.com/team-blue-raw-data/reports/"

            f"{commodity}/{report_date:%Y}/{report_date:%m}/"

            f"{report_prefix_map[commodity]}{report_date:%Y-%m-%d}.txt"

        )

    url = st.session_state.report_url

    use_public = False

    use_service_account = True

    auto_load = False

    load_clicked = st.button("Load Report")

    if st.session_state.get("report_text"):

        st.markdown("### 📚 Navigate")

        if st.button("퀀트 분석"):

            st.session_state.scroll_target = "section-quant"

        if st.button("뉴스 분석"):

            st.session_state.scroll_target = "section-news"

        if st.button("종합 의견"):

            st.session_state.scroll_target = "section-summary"



if "report_text" not in st.session_state:

    st.session_state.report_text = ""

if "report_status" not in st.session_state:

    st.session_state.report_status = ""

if "report_compact" not in st.session_state:

    st.session_state.report_compact = ""

if st.session_state.get("last_report_date") and st.session_state.last_report_date != today:

    st.session_state.last_report_date = None

    st.session_state.last_loaded_date = None

if st.session_state.get("last_report_commodity") and st.session_state.last_report_commodity != commodity:

    st.session_state.last_report_commodity = None

    st.session_state.last_loaded_date = None

page_title = f"✅ Daily Market Brief · {commodity} · {report_date:%Y-%m-%d}"
st.title(page_title)



should_load = False

if auto_load and report_date != st.session_state.get("last_loaded_date"):

    should_load = True

if load_clicked:

    should_load = True



if should_load:

    try:

        if use_service_account:

            try:

                from google.cloud import storage

            except Exception as e:

                raise RuntimeError("google-cloud-storage 미설치") from e

            bucket_name, blob_name = _parse_gcs_url(url)

            if not bucket_name or not blob_name:

                raise ValueError("GCS URL 형식이 올바르지 않습니다. gs://bucket/path 또는 storage.googleapis.com URL을 사용하세요.")

            client = storage.Client()  # GOOGLE_APPLICATION_CREDENTIALS 사용

            blob = client.bucket(bucket_name).blob(blob_name)

            text = blob.download_as_text()

            st.session_state.report_text = text

            st.session_state.report_status = f"✅ Loaded via service account: {len(text):,} chars"

        else:

            fetch_url = url

            if use_public and "storage.cloud.google.com/" in url:

                fetch_url = url.replace("https://storage.cloud.google.com/", "https://storage.googleapis.com/")

            resp = requests.get(fetch_url, timeout=15)

            resp.raise_for_status()

            st.session_state.report_text = resp.text

            if resp.text.lstrip().lower().startswith("<!doctype html") or "<html" in resp.text[:200].lower():

                st.session_state.report_status = "⚠️ HTML 응답 감지됨: 인증 페이지일 가능성이 큽니다. 공개/서명 URL로 다시 시도해 주세요."

            else:

                st.session_state.report_status = f"✅ Loaded: {len(resp.text):,} chars"

        st.session_state.last_loaded_date = report_date

        st.session_state.last_report_date = report_date

        st.session_state.last_report_commodity = commodity

        st.session_state.report_url = url

        st.session_state.report_compact = compress_report_text(st.session_state.report_text)

    except Exception as e:

        st.session_state.report_text = ""

        st.session_state.report_status = f"❌ Load failed: {e}"

        st.caption("스토리지 권한이 필요한 URL일 수 있습니다. 공개 링크 or 서명된 URL로 테스트해 주세요.")



show_chat = st.toggle("💬 Open Chatbot", value=False)



if show_chat:

    left_col, right_col = st.columns([3, 1], gap="large")

else:

    left_col, right_col = st.columns([1, 0.0001], gap="large")



with left_col:

    # 보고서 상태 배너는 표시하지 않음

    if st.session_state.report_text:

        report_text = st.session_state.report_text

        sections = _parse_sections(report_text)

        opinion = _extract_opinion(report_text)

        ml_dir = _extract_ml_direction(report_text) or "N/A"

        sentiment_label = _extract_sentiment_label(report_text)

        badge_class = "opinion-hold"

        if opinion == "BUY":

            badge_class = "opinion-buy"

        elif opinion == "SELL":

            badge_class = "opinion-sell"

        ml_badge_class = "opinion-hold"

        if str(ml_dir).lower() == "up":

            ml_badge_class = "opinion-buy"

        elif str(ml_dir).lower() == "down":

            ml_badge_class = "opinion-sell"

        sentiment_badge_class = "opinion-hold"

        if "긍정" in sentiment_label:

            sentiment_badge_class = "opinion-buy"

        elif "부정" in sentiment_label:

            sentiment_badge_class = "opinion-sell"



        st.markdown(
            textwrap.dedent(
                f"""
                <div class="summary-row">
                  <div class="summary-card">
                    <div class="summary-title">종합 의견</div>
                    <div class="opinion-badge {badge_class}">{opinion}</div>
                  </div>
                  <div class="summary-card">
                    <div class="summary-title">머신러닝 예측</div>
                    <div class="summary-metric">
                      <span class="opinion-badge {ml_badge_class}">{ml_dir}</span>
                    </div>
                  </div>
                  <div class="summary-card">
                    <div class="summary-title">시장 심리</div>
                    <div class="summary-metric">
                      <span class="opinion-badge {sentiment_badge_class}">{sentiment_label}</span>
                    </div>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )



        anchors = []

        for key in ["quant", "news", "summary"]:

            line = _find_header_line(report_text, key)

            anchors.append({"key": key, "line": line})



        safe_text = html.escape(report_text)

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="report-shell">
                <div class="report-title">Daily Market Brief · {commodity_label}</div>
                <div class="report-sub">Date: {report_date:%Y-%m-%d} · Source: Loaded report text</div>
                  <pre class="report-body">{safe_text}</pre>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )



        if anchors:

            anchors_json = json.dumps(anchors, ensure_ascii=False)

            components.html(

                f"""

                <script>

                (function() {{

                  const anchors = {anchors_json};

                  const doc = parent.document;

                  const pre = doc.querySelector("pre.report-body");

                  if (!pre) return;

                  const textNode = pre.firstChild;

                  if (!textNode || textNode.nodeType !== Node.TEXT_NODE) return;

                  anchors.forEach(a => {{

                    if (!a.line) {{

                      const span = doc.createElement("span");

                      span.id = `section-${{a.key}}`;

                      span.className = "section-anchor";

                      pre.appendChild(span);

                      return;

                    }}

                    const idx = textNode.data.indexOf(a.line);

                    if (idx === -1) {{

                      const span = doc.createElement("span");

                      span.id = `section-${{a.key}}`;

                      span.className = "section-anchor";

                      pre.appendChild(span);

                      return;

                    }}

                    const startNode = textNode.splitText(idx);

                    startNode.splitText(a.line.length);

                    const span = doc.createElement("span");

                    span.id = `section-${{a.key}}`;

                    span.className = "section-anchor";

                    span.textContent = a.line;

                    pre.replaceChild(span, startNode);

                  }});

                }})();

                </script>

                """,

                height=0,

            )



        if st.session_state.get("scroll_target"):

            target = st.session_state.get("scroll_target")

            components.html(

                f"""

                <script>

                const targetId = "{target}";

                setTimeout(() => {{

                  const el = parent.document.getElementById(targetId);

                  if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});

                }}, 150);

                </script>

                """,

                height=0,

            )

            st.session_state.scroll_target = None



with right_col:

    if show_chat:

        st.subheader("💬 Report Q&A")

        if "chat_messages" not in st.session_state:

            st.session_state.chat_messages = []

        if st.session_state.get("clear_chat_input"):

            st.session_state.chat_input = ""

            st.session_state.clear_chat_input = False

        chat_blocks = []

        for msg in st.session_state.chat_messages:

            role = msg.get("role", "assistant")

            raw_content = msg.get("content", "")

            if raw_content is None or str(raw_content).strip() == "":

                continue

            content = html.escape(str(raw_content))

            icon = "🙋" if role == "user" else "🤖"

            chat_blocks.append(

                f"<div class='chat-msg'><div class='chat-role {role}'>{icon}</div>"

                f"<div class='chat-bubble'>{content}</div></div>"

            )

        chat_html = f"<div class='chat-shell'>{''.join(chat_blocks)}</div>"

        st.markdown(chat_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        user_input = st.text_input("보고서에 대해 질문하세요", key="chat_input")

        send_clicked = st.button("Send")

        if send_clicked and user_input.strip():

            question = user_input.strip()

            st.session_state.chat_messages.append({"role": "user", "content": question})

            st.session_state.clear_chat_input = True

            if not st.session_state.report_text:

                st.session_state.chat_messages.append(

                    {"role": "assistant", "content": "먼저 보고서를 로드해 주세요."}

                )

            else:

                with st.spinner("금융 에이전트가 보고서를 분석하고 있습니다..."):

                    try:

                        answer = get_gemini_3_response(st.session_state.report_text, question)

                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

                    except Exception as e:

                        err_msg = f"Gemini 호출 실패: {e}"

                        st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})

            st.rerun()
