# ==========================================
# MULTILINGUAL HARMFUL CONTENT ANALYSER
# Streamlit Dashboard — Enhanced Version
# ==========================================

import os
import re
import pickle
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="SafeNet — Multilingual Content Analyser",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS — Dark Forensic Theme
# ==========================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e17;
    color: #e2e8f0;
  }

  .stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #0f1623 50%, #0a0e17 100%);
  }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .metric-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }

  .verdict-harmful {
    background: linear-gradient(135deg, #2d0a0a, #450f0f);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 800;
    color: #fca5a5;
    letter-spacing: 2px;
    animation: pulse-red 2s infinite;
  }

  .verdict-safe {
    background: linear-gradient(135deg, #0a2d1a, #0f450f);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 800;
    color: #86efac;
    letter-spacing: 2px;
  }

  .tag {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #2563eb;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #93c5fd;
    margin: 2px;
  }

  .tag-red {
    background: #2d0a0a;
    border-color: #ef4444;
    color: #fca5a5;
  }

  .tag-orange {
    background: #2d1a0a;
    border-color: #f97316;
    color: #fdba74;
  }

  .tag-green {
    background: #0a2d1a;
    border-color: #22c55e;
    color: #86efac;
  }

  .score-bar-bg {
    background: #1e293b;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin: 6px 0;
  }

  .score-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
  }

  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #64748b;
    text-transform: uppercase;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }

  .stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
  }

  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important;
  }

  [data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b !important;
  }

  .stDataFrame { border-radius: 10px; overflow: hidden; }

  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50% { box-shadow: 0 0 20px 8px rgba(239,68,68,0.15); }
  }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MULTILINGUAL ANALYSIS LOGIC
# ==========================================

HARMFUL_TANGLISH = [
    "mokka","thevdiya","punda","otha","naaye","koothi","velaiya po",
    "di poda","da poda","sunni","oombu","vaaya moodu","paithiyam","loosu",
    "bakwaas","chutiya","saala","kamina","harami","gadha","ullu",
    "bewakoof","chup kar","nikal","maro","jalao","bhago","maar","khatam karo"
]

TONE_KEYWORDS = {
    "threatening": ["kill","attack","destroy","burn","shoot","beat","violence",
                    "bomb","stab","hurt","harm","murder","maro","jalao","maar","adi","thakku"],
    "hateful":     ["hate","ban","deport","enemies","invaders","terrorist","throw out",
                    "not real","inferior","filth","naxal","traitor","anti-national"],
    "doxxing":     ["home address","phone number","personal information","share details",
                    "post his","post her","find out where","school name","college name","doxx","expose"],
    "discriminatory": ["caste","reservation","minority","religion","community","upper caste",
                       "lower caste","dalit","muslim","christian","northeast","migrant","scheduled caste"],
    "cyberbullying": ["hack","private photos","rumors","reputation destroyed","publicly humiliate",
                      "disgraced","silenced","expelled","blacklisted","fake reviews"],
    "neutral": []
}

INTENT_MAP = {
    "threatening":     "Physical Harm / Violence",
    "hateful":         "Hate Speech / Incitement",
    "doxxing":         "Doxxing / Privacy Violation",
    "discriminatory":  "Discrimination / Bias",
    "cyberbullying":   "Cyberbullying / Harassment",
    "neutral":         "No Harmful Intent"
}

TONE_COLORS = {
    "threatening":    "#ef4444",
    "hateful":        "#f97316",
    "doxxing":        "#a855f7",
    "discriminatory": "#eab308",
    "cyberbullying":  "#06b6d4",
    "neutral":        "#22c55e"
}

STOP_WORDS = set([
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn",
    "na", "nee", "avan", "aval", "avanga", "inga", "anga",
    "enna", "epdi", "endha", "yaar", "oru", "alla", "illai",
    "da", "di", "pa", "ma", "la", "nu", "ku", "le"
])

def detect_language(text):
    text_lower = text.lower()
    tanglish = ["poda","ponga","da","di","macha","pa","kku","nnu","lla","nga","oru","naan","en"]
    hindi    = ["hai","nahi","karo","aur","mera","tera","yaar","bhai","dost","accha"]
    ts = sum(1 for m in tanglish if f" {m} " in f" {text_lower} ")
    hs = sum(1 for m in hindi   if f" {m} " in f" {text_lower} ")
    if ts >= 2: return "🌐 Tanglish"
    if hs >= 2: return "🌐 Hindi-English"
    if any(w in text_lower for w in HARMFUL_TANGLISH): return "🌐 Regional"
    return "🇬🇧 English"

def detect_tone(text):
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower)
              for t, kws in TONE_KEYWORDS.items() if t != "neutral"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def extract_keywords(text):
    text_lower = text.lower()
    all_kws = [kw for kws in TONE_KEYWORDS.values() for kw in kws] + HARMFUL_TANGLISH
    return [kw for kw in set(all_kws) if kw in text_lower][:6] or ["—"]

def analyse_message(text, model):
    cleaned   = clean_text(text)
    language  = detect_language(text)
    tone      = detect_tone(text)
    intent    = INTENT_MAP[tone]
    keywords  = extract_keywords(text)

    proba     = model.predict_proba([cleaned])[0]
    classes   = list(model.classes_)
    harmful_i = classes.index('harmful')
    score     = proba[harmful_i]
    verdict   = "harmful" if score >= 0.5 else "non-harmful"

    return {
        "verdict":   verdict,
        "score":     score,
        "language":  language,
        "tone":      tone,
        "intent":    intent,
        "keywords":  keywords,
        "cleaned":   cleaned
    }

# ==========================================
# LOAD MODEL & DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "processed_output.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

model = load_model()
df    = load_data()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ **SafeNet**")
    st.markdown("<p style='color:#64748b;font-size:0.8rem;'>Multilingual Harmful Content Detection</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🔍 Live Analyser",
        "📊 Dashboard",
        "🗺️  Heatmap",
        "📄 Dataset"
    ])

    st.divider()
    if df is not None:
        total   = len(df)
        harmful = len(df[df['label'] == 'harmful'])
        safe    = total - harmful
        st.markdown(f"**Total Records:** `{total}`")
        st.markdown(f"**Harmful:** `{harmful}` ({harmful/total*100:.1f}%)")
        st.markdown(f"**Safe:** `{safe}` ({safe/total*100:.1f}%)")
    st.divider()
    st.markdown("<p style='color:#374151;font-size:0.7rem;text-align:center;'>Model: Logistic Regression + TF-IDF<br>Languages: EN · Tamil · Tanglish · Hindi</p>", unsafe_allow_html=True)

# ==========================================
# PAGE: LIVE ANALYSER
# ==========================================
if page == "🔍 Live Analyser":
    st.markdown("# 🛡️ SafeNet — Live Analyser")
    st.markdown("<p style='color:#64748b;'>Analyse messages in English, Tamil, Tanglish, or Hindi for harmful content, tone & intent.</p>", unsafe_allow_html=True)
    st.divider()

    if model is None:
        st.error("⚠️ Model not found. Please run `model.py` first to train and save the model.")
    else:
        user_input = st.text_area(
            "Enter a message to analyse",
            placeholder="Type in English, Tamil, Tanglish, or Hindi...\nExample: Da poda, I will hack your account and destroy you!",
            height=120
        )

        col_btn, col_clear = st.columns([1, 5])
        with col_btn:
            analyse = st.button("⚡ Analyse", use_container_width=True)

        if analyse and user_input.strip():
            result = analyse_message(user_input, model)

            st.divider()

            # Verdict banner
            if result['verdict'] == 'harmful':
                st.markdown(f"""<div class="verdict-harmful">⚠️ HARMFUL CONTENT DETECTED</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="verdict-safe">✅ CONTENT IS SAFE</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">ANALYSIS RESULTS</div>', unsafe_allow_html=True)

                # Toxicity score bar
                score_pct = result['score'] * 100
                color = "#ef4444" if score_pct >= 70 else "#f97316" if score_pct >= 40 else "#22c55e"
                st.markdown(f"""
                <div style="margin-bottom:16px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.85rem;color:#94a3b8;">Toxicity Score</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:{color};">{score_pct:.1f}%</span>
                  </div>
                  <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{score_pct}%;background:{color};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Language
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">DETECTED LANGUAGE</div>
                  <div style="font-size:1.1rem;font-weight:700;">{result['language']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Tone
                tone_color = TONE_COLORS.get(result['tone'], "#94a3b8")
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">DETECTED TONE</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{tone_color};">
                    {result['tone'].upper()}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="section-header">INTENT & KEYWORDS</div>', unsafe_allow_html=True)

                # Intent
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">CLASSIFIED INTENT</div>
                  <div style="font-size:1rem;font-weight:700;">{result['intent']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Keywords
                kw_html = " ".join([
                    f'<span class="tag {"tag-red" if result["verdict"]=="harmful" else "tag-green"}">{kw}</span>'
                    for kw in result['keywords']
                ])
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:8px;">EXTRACTED KEYWORDS</div>
                  <div>{kw_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Processed text
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">PROCESSED TEXT</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#94a3b8;line-height:1.6;">
                    {result['cleaned'] if result['cleaned'] else '—'}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        elif analyse and not user_input.strip():
            st.warning("Please enter a message to analyse.")

# ==========================================
# PAGE: DASHBOARD
# ==========================================
elif page == "📊 Dashboard":
    st.markdown("# 📊 Dataset Dashboard")

    if df is None:
        st.error("⚠️ processed_output.csv not found. Please run model.py first.")
    else:
        # Top KPIs
        total   = len(df)
        harmful = len(df[df['label'] == 'harmful'])
        safe    = total - harmful
        avg_tox = df['toxicity_score'].mean() if 'toxicity_score' in df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Total Records",  f"{total}",          "#60a5fa"),
            (c2, "Harmful",        f"{harmful}",         "#ef4444"),
            (c3, "Non-Harmful",    f"{safe}",            "#22c55e"),
            (c4, "Avg Toxicity",   f"{avg_tox:.2f}",    "#f97316"),
        ]:
            col.markdown(f"""
            <div class="metric-card" style="text-align:center;">
              <div style="font-size:0.72rem;color:#64748b;letter-spacing:2px;">{label}</div>
              <div style="font-size:2rem;font-weight:800;color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📊 Toxicity Score Distribution")
            if 'toxicity_score' in df.columns:
                chart_df = df[['toxicity_score']].copy()
                st.bar_chart(chart_df, color="#ef4444", height=280)

        with col_b:
            st.markdown("#### 🔥 Top 10 Regions by Avg Toxicity")
            if 'location' in df.columns and 'toxicity_score' in df.columns:
                top = df.groupby('location')['toxicity_score'].mean().sort_values(ascending=False).head(10)
                top_df = top.reset_index()
                top_df.columns = ['Location', 'Avg Toxicity']
                st.bar_chart(top_df.set_index('Location'), color="#f97316", height=280)

        st.divider()

        # Tone distribution
        if 'tone' in df.columns:
            st.markdown("#### 🎭 Tone Distribution")
            tone_counts = df['tone'].value_counts().reset_index()
            tone_counts.columns = ['Tone', 'Count']
            st.bar_chart(tone_counts.set_index('Tone'), color="#a855f7", height=220)

# ==========================================
# PAGE: HEATMAP
# ==========================================
elif page == "🗺️  Heatmap":
    st.markdown("# 🗺️ Geo-Toxicity Heatmap")
    st.markdown("<p style='color:#64748b;'>Geographic distribution of harmful content across India.</p>", unsafe_allow_html=True)

    if df is None:
        st.error("⚠️ processed_output.csv not found. Run model.py first.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_only = st.selectbox("Filter by label", ["All", "Harmful Only", "Non-Harmful Only"])
        with col_f2:
            min_tox = st.slider("Min Toxicity Score", 0.0, 1.0, 0.0, 0.05) if 'toxicity_score' in df.columns else 0.0

        map_df = df.copy()
        if show_only == "Harmful Only":
            map_df = map_df[map_df['label'] == 'harmful']
        elif show_only == "Non-Harmful Only":
            map_df = map_df[map_df['label'] == 'non-harmful']
        if 'toxicity_score' in map_df.columns:
            map_df = map_df[map_df['toxicity_score'] >= min_tox]

        map_df['lat'] = map_df['latitude']
        map_df['lon'] = map_df['longitude']

        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5,
                       tiles="CartoDB dark_matter")

        heat_data = map_df[['lat', 'lon', 'toxicity_score']].dropna().values.tolist()
        if heat_data:
            HeatMap(heat_data, radius=18, blur=15, min_opacity=0.4,
                    gradient={0.2: 'blue', 0.5: 'orange', 1.0: 'red'}).add_to(m)

        for _, row in map_df[map_df['label'] == 'harmful'].iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color='#ef4444',
                fill=True,
                fill_color='#ef4444',
                fill_opacity=0.7,
                tooltip=f"{row.get('location','?')} — {row.get('toxicity_score',0):.2f}"
            ).add_to(m)

        st_folium(m, width=None, height=520)

        st.markdown(f"**Showing `{len(map_df)}` records on map.**")

# ==========================================
# PAGE: DATASET
# ==========================================
elif page == "📄 Dataset":
    st.markdown("# 📄 Dataset Viewer")

    if df is None:
        st.error("⚠️ processed_output.csv not found. Run model.py first.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            label_filter = st.multiselect("Filter by Label", ["harmful", "non-harmful"],
                                          default=["harmful", "non-harmful"])
        with col2:
            if 'tone' in df.columns:
                tone_filter = st.multiselect("Filter by Tone", df['tone'].unique().tolist(),
                                             default=df['tone'].unique().tolist())
            else:
                tone_filter = []
        with col3:
            search = st.text_input("Search text", "")

        filtered = df[df['label'].isin(label_filter)]
        if tone_filter and 'tone' in df.columns:
            filtered = filtered[filtered['tone'].isin(tone_filter)]
        if search:
            filtered = filtered[filtered['text'].str.contains(search, case=False, na=False)]

        st.markdown(f"**Showing `{len(filtered)}` of `{len(df)}` records**")

        cols_to_show = [c for c in ['tweet_id','text','location','label','tone','intent','toxicity_score','language'] if c in filtered.columns]
        st.dataframe(
            filtered[cols_to_show].style.map(
                lambda v: 'background-color: #2d0a0a; color: #fca5a5;' if v == 'harmful'
                else ('background-color: #0a2d1a; color: #86efac;' if v == 'non-harmful' else ''),
                subset=['label'] if 'label' in cols_to_show else []
            ),
            use_container_width=True,
            height=500
        )

        # Download
        csv = filtered.to_csv(index=False)
        st.download_button("⬇️ Download Filtered CSV", csv, "filtered_data.csv", "text/csv")