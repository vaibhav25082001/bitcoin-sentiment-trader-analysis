"""
Bitcoin Market Sentiment × Trader Performance Analysis
=======================================================
Full Streamlit app with 5 AI-powered features:
  1. Ask-the-Data Chatbot
  2. AI Insight Generator (per chart)
  3. Sentiment Predictor (ML – Random Forest)
  4. Anomaly Explainer
  5. Trade Strategy Advisor

Supports 3 AI providers — use whichever API key you have:
  - Anthropic Claude  (claude-sonnet-4)
  - OpenAI ChatGPT    (gpt-4o)
  - Google Gemini     (gemini-1.5-pro)

Requirements:
  pip install streamlit anthropic openai google-generativeai scikit-learn plotly pandas numpy scipy

Usage:
  Place both CSVs in the same folder as this file, then run:
    streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import anthropic
import openai
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Sentiment × Trader Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:800; color:#f7931a; }
    .sub-title   { font-size:1.0rem; color:#aaa; margin-bottom:1.5rem; }
    .metric-card { background:#1e1e1e; border-radius:10px; padding:1rem;
                   border-left:4px solid #f7931a; margin-bottom:0.5rem; }
    .insight-box { background:#1a2636; border-radius:10px; padding:1rem;
                   border-left:4px solid #00c9ff; margin-top:0.5rem; }
    .chat-user   { background:#2a2a2a; border-radius:8px; padding:0.6rem 1rem;
                   margin:4px 0; }
    .chat-ai     { background:#1a2636; border-radius:8px; padding:0.6rem 1rem;
                   margin:4px 0; border-left:3px solid #00c9ff; }
    .anomaly-box { background:#2a1a1a; border-radius:10px; padding:1rem;
                   border-left:4px solid #ff4b4b; margin-top:0.5rem; }
    .strategy-box{ background:#1a2a1a; border-radius:10px; padding:1rem;
                   border-left:4px solid #00ff88; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MULTI-PROVIDER AI CALLER
# ─────────────────────────────────────────────────────────────

def call_ai(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """
    Universal AI caller — routes to whichever provider the user selected.
    Falls back gracefully with a readable error if the key is missing/wrong.
    """
    provider = st.session_state.get("ai_provider", "Anthropic Claude")
    api_key  = st.session_state.get("ai_api_key", "").strip()

    if not api_key:
        return "⚠️ No API key entered. Paste your key in the sidebar to activate AI features."

    try:
        # ── Anthropic Claude ──────────────────────────────────
        if provider == "Anthropic Claude":
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text

        # ── OpenAI ChatGPT ────────────────────────────────────
        elif provider == "OpenAI ChatGPT":
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content

        # ── Google Gemini ─────────────────────────────────────
        elif provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                system_instruction=system_prompt,
            )
            resp = model.generate_content(user_prompt)
            return resp.text

        else:
            return "⚠️ Unknown provider selected."

    except Exception as e:
        return f"⚠️ AI error ({provider}): {e}"


# Keep backward-compat alias used in the chatbot tab
def call_claude(system_prompt, user_prompt, max_tokens=800):
    return call_ai(system_prompt, user_prompt, max_tokens)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(trader_path: str, fg_path: str):
    # ── Fear & Greed ──────────────────────────────────────────
    fg = pd.read_csv(fg_path)
    fg["date"] = pd.to_datetime(fg["date"], dayfirst=False)
    fg = fg.rename(columns={"value": "fg_value", "classification": "sentiment"})

    # ── Trader data ───────────────────────────────────────────
    df = pd.read_csv(trader_path)
    df["date"] = pd.to_datetime(df["Timestamp IST"], format="mixed", dayfirst=True).dt.date
    df["date"] = pd.to_datetime(df["date"])
    df["Closed PnL"] = pd.to_numeric(df["Closed PnL"], errors="coerce").fillna(0)
    df["Fee"]        = pd.to_numeric(df["Fee"],        errors="coerce").fillna(0)
    df["Size USD"]   = pd.to_numeric(df["Size USD"],   errors="coerce").fillna(0)

    # ── Merge ─────────────────────────────────────────────────
    merged = df.merge(fg[["date", "fg_value", "sentiment"]], on="date", how="left")
    merged = merged.dropna(subset=["sentiment"])

    # ── Closing trades only ───────────────────────────────────
    closing = merged[merged["Closed PnL"] != 0].copy()
    closing["win"] = (closing["Closed PnL"] > 0).astype(int)

    return fg, df, merged, closing

# ─────────────────────────────────────────────────────────────
# SIDEBAR — file pickers
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📁 Data Sources")

default_trader = "historical_data.csv"
default_fg     = "fear_greed_index.csv"

trader_file = st.sidebar.text_input("Trader CSV path", value=default_trader)
fg_file     = st.sidebar.text_input("Fear & Greed CSV path", value=default_fg)

# Try to resolve paths — support relative or absolute
def resolve(path):
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(alt):
        return alt
    return None

trader_path = resolve(trader_file)
fg_path     = resolve(fg_file)

if not trader_path or not fg_path:
    st.error("❌ CSV files not found. Place both CSVs in the same folder as app.py and check the paths above.")
    st.stop()

fg, df, merged, closing = load_data(trader_path, fg_path)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — API key
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 AI Settings")

provider = st.sidebar.selectbox(
    "AI Provider",
    ["Anthropic Claude", "OpenAI ChatGPT", "Google Gemini"],
    help="Use whichever API key you already have",
)
st.session_state["ai_provider"] = provider

KEY_LABELS = {
    "Anthropic Claude": "Anthropic API Key",
    "OpenAI ChatGPT":   "OpenAI API Key",
    "Google Gemini":    "Google Gemini API Key",
}
KEY_LINKS = {
    "Anthropic Claude": "console.anthropic.com",
    "OpenAI ChatGPT":   "platform.openai.com/api-keys",
    "Google Gemini":    "aistudio.google.com/app/apikey",
}

api_key_input = st.sidebar.text_input(
    KEY_LABELS[provider],
    type="password",
    help=f"Get yours at {KEY_LINKS[provider]}",
)
st.session_state["ai_api_key"] = api_key_input

if api_key_input:
    st.sidebar.success(f"✅ {provider} key set")
else:
    st.sidebar.info(f"🔑 Get a free key at {KEY_LINKS[provider]}")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📊 BTC Sentiment × Trader Performance</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hyperliquid Trader Data + Fear & Greed Index — AI-Enhanced Analysis</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_COLORS = {
    "Extreme Fear": "#ff4b4b",
    "Fear":         "#ff9900",
    "Neutral":      "#aaaaaa",
    "Greed":        "#00cc88",
    "Extreme Greed":"#00ff88",
}

m1, m2, m3, m4, m5 = st.columns(5)
stats = closing.groupby("sentiment").agg(
    mean_pnl=("Closed PnL", "mean"),
    win_rate =("win", "mean"),
    trades   =("Closed PnL", "count"),
).reset_index()

for i, col in enumerate([m1, m2, m3, m4, m5]):
    sent = SENTIMENT_ORDER[i]
    row  = stats[stats["sentiment"] == sent]
    if not row.empty:
        col.metric(
            label=sent,
            value=f"${row['mean_pnl'].values[0]:.1f}",
            delta=f"WR {row['win_rate'].values[0]*100:.1f}%",
        )

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Dashboard",
    "💬 Ask the Data",
    "🔍 AI Insights",
    "🧠 Sentiment Predictor",
    "⚠️ Anomaly Explainer",
    "🎯 Strategy Advisor",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Fear & Greed Index — Distribution & Timeline")
    c1, c2 = st.columns(2)

    with c1:
        cnt = fg["sentiment"].value_counts().reset_index()
        cnt.columns = ["sentiment", "count"]
        fig = px.pie(cnt, names="sentiment", values="count",
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                     title="Sentiment Distribution (All History)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(fg.sort_values("date"), x="date", y="fg_value",
                      color_discrete_sequence=["#f7931a"],
                      title="Fear & Greed Index Over Time")
        fig.add_hrect(y0=0, y1=25, fillcolor="#ff4b4b", opacity=0.08)
        fig.add_hrect(y0=25, y1=45, fillcolor="#ff9900", opacity=0.08)
        fig.add_hrect(y0=55, y1=75, fillcolor="#00cc88", opacity=0.08)
        fig.add_hrect(y0=75, y1=100, fillcolor="#00ff88", opacity=0.08)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean PnL & Win Rate by Sentiment")
    c3, c4 = st.columns(2)

    with c3:
        fig = px.bar(stats, x="sentiment", y="mean_pnl",
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                     category_orders={"sentiment": SENTIMENT_ORDER},
                     title="Mean PnL per Closing Trade by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        stats["win_pct"] = stats["win_rate"] * 100
        fig = px.bar(stats, x="sentiment", y="win_pct",
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                     category_orders={"sentiment": SENTIMENT_ORDER},
                     title="Win Rate (%) by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trade Volume by Sentiment")
    c5, c6 = st.columns(2)

    with c5:
        vol = merged.groupby("sentiment")["Size USD"].sum().reset_index()
        fig = px.bar(vol, x="sentiment", y="Size USD",
                     color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                     category_orders={"sentiment": SENTIMENT_ORDER},
                     title="Total Volume (USD) by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        top_coins = closing.groupby("Coin")["Closed PnL"].mean().nlargest(10).reset_index()
        fig = px.bar(top_coins, x="Coin", y="Closed PnL",
                     title="Top 10 Coins by Mean Closed PnL",
                     color="Closed PnL",
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily PnL Timeline with Sentiment Background")
    daily = closing.groupby(["date", "sentiment"]).agg(
        total_pnl=("Closed PnL", "sum"),
        trade_count=("Closed PnL", "count"),
    ).reset_index().sort_values("date")

    daily["cum_pnl"] = daily["total_pnl"].cumsum()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Daily PnL", "Cumulative PnL"),
                        vertical_spacing=0.08)

    for sent in SENTIMENT_ORDER:
        sub = daily[daily["sentiment"] == sent]
        fig.add_trace(go.Bar(x=sub["date"], y=sub["total_pnl"],
                             name=sent, marker_color=SENTIMENT_COLORS[sent],
                             showlegend=True), row=1, col=1)

    fig.add_trace(go.Scatter(x=daily["date"], y=daily["cum_pnl"],
                             mode="lines", name="Cumulative PnL",
                             line=dict(color="#f7931a", width=2),
                             showlegend=True), row=2, col=1)
    fig.update_layout(height=500, barmode="stack")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ASK-THE-DATA CHATBOT
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("💬 Ask the Data")
    st.markdown("Ask any question about the dataset — the AI will answer using real numbers from your data.")

    # Build a compact data summary for the AI context
    @st.cache_data
    def build_data_summary():
        s = stats.copy()
        s["win_pct"] = (s["win_rate"] * 100).round(1)
        s["mean_pnl"] = s["mean_pnl"].round(2)

        trade_vol = merged.groupby("sentiment")["Size USD"].sum().round(0).to_dict()
        daily_corr = closing.groupby("date").agg(
            pnl=("Closed PnL", "sum"),
            fg =("fg_value",   "first"),
        ).corr().loc["fg", "pnl"]

        top_coins = closing.groupby("Coin")["Closed PnL"].mean().nlargest(5).round(2).to_dict()
        trader_pnl = closing.groupby("Account")["Closed PnL"].sum().round(0).to_dict()

        return f"""
DATASET SUMMARY (use these exact numbers when answering):
- Total trades: {len(merged):,}
- Closing trades with PnL: {len(closing):,}
- Date range: {merged['date'].min().date()} to {merged['date'].max().date()}
- Unique traders: {merged['Account'].nunique()}
- Unique coins: {merged['Coin'].nunique()}

SENTIMENT PERFORMANCE TABLE:
{s[['sentiment','mean_pnl','win_pct','trades']].to_string(index=False)}

VOLUME (USD) BY SENTIMENT: {json.dumps(trade_vol, default=str)}
CORRELATION (F&G value vs daily PnL): {daily_corr:.4f}
TOP 5 COINS BY MEAN PnL: {json.dumps(top_coins)}
TRADER TOTAL PnL (first 10): {json.dumps(dict(list(trader_pnl.items())[:10]))}
"""

    data_summary = build_data_summary()

    SYSTEM_CHAT = f"""You are a financial data analyst assistant. You have access to a dataset of
Hyperliquid trader activity merged with Bitcoin Fear & Greed Index data.
Always answer using the real numbers provided. Be concise and specific.
When relevant, mention if a finding is counter-intuitive or strategically interesting.

{data_summary}"""

    # Chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.markdown("**Suggested questions:**")
    q_cols = st.columns(3)
    suggestions = [
        "Which sentiment gives the best mean PnL?",
        "Is there a negative correlation between F&G and PnL?",
        "Which coin has the highest mean profit?",
        "What is the win rate during Extreme Fear?",
        "During which sentiment do traders trade the most?",
        "Who is the most profitable trader overall?",
    ]
    for i, q in enumerate(suggestions):
        if q_cols[i % 3].button(q, key=f"sugg_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})

    # Chat input
    user_q = st.chat_input("Type your question about the data…")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})

    # Generate AI reply for the last unanswered message
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("Thinking…"):
            # Build a simple conversation string for providers that don't support multi-turn natively via call_ai
            history_msgs = st.session_state.chat_history[-8:]
            conversation_text = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in history_msgs[:-1]  # everything except the latest question
            )
            last_q = history_msgs[-1]["content"]
            full_prompt = (
                f"{conversation_text}\n\nUser: {last_q}" if conversation_text
                else last_q
            )
            ai_reply = call_ai(SYSTEM_CHAT, full_prompt, max_tokens=600)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑‍💻 <b>You:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 <b>AI:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True)

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 3 — AI INSIGHT GENERATOR
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🔍 AI Insight Generator")
    st.markdown("Select a chart topic and get a written AI analysis of the underlying data.")

    insight_topic = st.selectbox("Choose analysis topic:", [
        "Fear/Greed Distribution → Trading Behavior",
        "Mean PnL per Sentiment Class",
        "Win Rate per Sentiment Class",
        "Trading Volume by Sentiment",
        "Cumulative PnL Over Time",
        "Top Coins by Mean PnL",
        "Overall Dataset Summary & Key Takeaways",
    ])

    TOPIC_DATA = {
        "Fear/Greed Distribution → Trading Behavior": lambda: (
            merged["sentiment"].value_counts().to_dict(),
            "sentiment distribution and trade counts"
        ),
        "Mean PnL per Sentiment Class": lambda: (
            stats.set_index("sentiment")["mean_pnl"].round(2).to_dict(),
            "mean PnL by sentiment"
        ),
        "Win Rate per Sentiment Class": lambda: (
            (stats.set_index("sentiment")["win_rate"] * 100).round(1).to_dict(),
            "win rate % by sentiment"
        ),
        "Trading Volume by Sentiment": lambda: (
            merged.groupby("sentiment")["Size USD"].sum().round(0).to_dict(),
            "total USD volume by sentiment"
        ),
        "Cumulative PnL Over Time": lambda: (
            {
                "total_pnl": closing["Closed PnL"].sum().round(2),
                "best_day_pnl": closing.groupby("date")["Closed PnL"].sum().max().round(2),
                "worst_day_pnl": closing.groupby("date")["Closed PnL"].sum().min().round(2),
            },
            "cumulative and daily PnL stats"
        ),
        "Top Coins by Mean PnL": lambda: (
            closing.groupby("Coin")["Closed PnL"].mean().nlargest(10).round(2).to_dict(),
            "top 10 coins by mean closed PnL"
        ),
        "Overall Dataset Summary & Key Takeaways": lambda: (
            {
                "total_trades": len(merged),
                "closing_trades": len(closing),
                "traders": merged["Account"].nunique(),
                "coins": merged["Coin"].nunique(),
                "sentiment_mean_pnl": stats.set_index("sentiment")["mean_pnl"].round(2).to_dict(),
                "sentiment_win_rate": (stats.set_index("sentiment")["win_rate"] * 100).round(1).to_dict(),
            },
            "full dataset summary"
        ),
    }

    if st.button("✨ Generate AI Insight", type="primary"):
        data_fn = TOPIC_DATA[insight_topic]
        data_dict, data_desc = data_fn()

        prompt = f"""Analyze this {data_desc} from a Bitcoin trader dataset merged with the Fear & Greed Index.

Data:
{json.dumps(data_dict, indent=2, default=str)}

Write a structured insight with:
1. **Key Finding** — the single most important observation
2. **Why It Matters** — trading implication
3. **Counter-intuitive or Surprising Elements** — anything that defies conventional wisdom
4. **Actionable Recommendation** — one specific trading rule based on this data

Keep it concise, data-driven, and practical. Use exact numbers where possible."""

        with st.spinner("Generating insight…"):
            insight = call_claude(
                "You are a senior quantitative trading analyst. Provide sharp, data-driven insights.",
                prompt,
                max_tokens=700,
            )

        st.markdown('<div class="insight-box">' + insight.replace("\n", "<br>") + '</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — SENTIMENT PREDICTOR (ML)
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🧠 Sentiment Predictor — Will the next trade be profitable?")
    st.markdown("A Random Forest model trained on historical data predicts win probability based on market conditions.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    @st.cache_resource
    def train_model():
        feat = closing[["fg_value", "Size USD", "Fee", "Side", "Coin"]].copy()
        feat["Side_enc"] = (feat["Side"] == "BUY").astype(int)

        # Encode coin — top 10 by count, rest = "Other"
        top10 = feat["Coin"].value_counts().nlargest(10).index.tolist()
        feat["Coin_grp"] = feat["Coin"].apply(lambda x: x if x in top10 else "Other")
        le = LabelEncoder()
        feat["Coin_enc"] = le.fit_transform(feat["Coin_grp"])

        X = feat[["fg_value", "Size USD", "Fee", "Side_enc", "Coin_enc"]].values
        y = closing["win"].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        acc = accuracy_score(y_te, clf.predict(X_te))
        fi  = pd.Series(clf.feature_importances_,
                        index=["FG Value", "Size USD", "Fee", "Side", "Coin"]).sort_values(ascending=False)
        return clf, acc, fi, le, top10

    with st.spinner("Training model (first run only)…"):
        clf, acc, fi, le, top10 = train_model()

    st.success(f"✅ Model trained — Test Accuracy: **{acc*100:.1f}%**")

    # Feature importance chart
    fig = px.bar(fi.reset_index(), x="index", y=0,
                 title="Feature Importance (what drives win/loss prediction)",
                 labels={"index": "Feature", "0": "Importance"},
                 color="0", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Predict a Trade")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pred_fg    = st.slider("Fear & Greed Index Value", 0, 100, 50)
        pred_side  = st.selectbox("Trade Side", ["BUY", "SELL"])
    with pc2:
        pred_size  = st.number_input("Trade Size (USD)", min_value=10.0, max_value=500000.0, value=5000.0, step=100.0)
        pred_fee   = st.number_input("Expected Fee (USD)", min_value=0.0, max_value=1000.0, value=2.5, step=0.1)
    with pc3:
        coin_choices = top10 + ["Other"]
        pred_coin  = st.selectbox("Coin", coin_choices)

    if st.button("🎯 Predict Win Probability", type="primary"):
        side_enc = 1 if pred_side == "BUY" else 0
        coin_enc = le.transform([pred_coin])[0] if pred_coin in le.classes_ else le.transform(["Other"])[0]
        X_pred   = np.array([[pred_fg, pred_size, pred_fee, side_enc, coin_enc]])
        prob     = clf.predict_proba(X_pred)[0][1]

        col_r, col_g = st.columns(2)
        col_r.metric("Win Probability", f"{prob*100:.1f}%")
        col_g.metric("Loss Probability", f"{(1-prob)*100:.1f}%")

        bar_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Win Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00c9ff"},
                "steps": [
                    {"range": [0, 40],  "color": "#ff4b4b"},
                    {"range": [40, 60], "color": "#ff9900"},
                    {"range": [60, 100],"color": "#00cc88"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": prob*100}
            }
        ))
        bar_fig.update_layout(height=300)
        st.plotly_chart(bar_fig, use_container_width=True)

        # AI explanation of the prediction
        with st.spinner("Getting AI explanation…"):
            sentiment_label = fg[
                (fg["fg_value"] - pred_fg).abs() == (fg["fg_value"] - pred_fg).abs().min()
            ]["sentiment"].values[0]

            explanation = call_claude(
                "You are a trading AI. Explain trade predictions concisely.",
                f"""A trader is about to make this trade:
- Coin: {pred_coin}
- Side: {pred_side}
- Size: ${pred_size:,.0f} USD
- Fee: ${pred_fee:.2f}
- Current F&G Index: {pred_fg} ({sentiment_label})
- ML Model Win Probability: {prob*100:.1f}%

In 3-4 sentences: explain WHY the model predicted this probability, what role the sentiment plays, and give one specific piece of advice."""
            )
        st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — ANOMALY EXPLAINER
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("⚠️ Anomaly Explainer")
    st.markdown("Detect unusual trading days (statistical outliers in PnL or volume) and get AI explanations.")

    daily_stats = closing.groupby("date").agg(
        total_pnl   =("Closed PnL", "sum"),
        trade_count =("Closed PnL", "count"),
        mean_pnl    =("Closed PnL", "mean"),
        fg_value    =("fg_value", "first"),
        sentiment   =("sentiment", "first"),
    ).reset_index().sort_values("date")

    # Z-score anomaly detection
    from scipy import stats as scipy_stats
    daily_stats["pnl_z"] = scipy_stats.zscore(daily_stats["total_pnl"])
    daily_stats["vol_z"] = scipy_stats.zscore(daily_stats["trade_count"])

    threshold = st.slider("Z-score Threshold (higher = fewer anomalies)", 1.5, 3.5, 2.0, 0.1)
    anomalies = daily_stats[
        (daily_stats["pnl_z"].abs() > threshold) |
        (daily_stats["vol_z"].abs() > threshold)
    ].copy()

    st.markdown(f"**{len(anomalies)} anomalous days detected** (z > {threshold})")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stats["date"], y=daily_stats["total_pnl"],
        mode="lines", name="Daily PnL", line=dict(color="#f7931a", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["date"], y=anomalies["total_pnl"],
        mode="markers", name="Anomaly",
        marker=dict(color="#ff4b4b", size=10, symbol="x")
    ))
    fig.update_layout(title="Daily PnL with Anomalies Flagged", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Show top anomalies table
    top_anomalies = anomalies.nlargest(10, "pnl_z")[
        ["date", "total_pnl", "trade_count", "fg_value", "sentiment", "pnl_z", "vol_z"]
    ].reset_index(drop=True)
    top_anomalies["date"] = top_anomalies["date"].dt.strftime("%Y-%m-%d")
    top_anomalies = top_anomalies.round(2)
    st.dataframe(top_anomalies, use_container_width=True)

    # Explain selected anomaly
    st.markdown("### 🤖 Explain a Specific Anomaly")
    if not anomalies.empty:
        dates_list = anomalies["date"].dt.strftime("%Y-%m-%d").tolist()
        selected_date = st.selectbox("Select anomalous date to explain:", dates_list)

        if st.button("🔍 Explain This Anomaly", type="primary"):
            row = anomalies[anomalies["date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]

            # Coin breakdown for that day
            day_coins = closing[closing["date"].dt.strftime("%Y-%m-%d") == selected_date]
            coin_breakdown = day_coins.groupby("Coin")["Closed PnL"].sum().nlargest(5).round(2).to_dict()
            side_split     = day_coins["Side"].value_counts().to_dict()

            with st.spinner("Analyzing anomaly…"):
                explanation = call_claude(
                    "You are a quantitative risk analyst. Explain market anomalies clearly.",
                    f"""Analyze this anomalous trading day:

Date: {selected_date}
Total PnL: ${row['total_pnl']:,.2f} (Z-score: {row['pnl_z']:.2f})
Trade Count: {int(row['trade_count'])} (Z-score: {row['vol_z']:.2f})
Fear & Greed Index: {row['fg_value']} ({row['sentiment']})
Mean PnL per trade: ${row['mean_pnl']:.2f}
Top coins by PnL: {json.dumps(coin_breakdown)}
Buy/Sell split: {json.dumps(side_split)}

Explain:
1. **What made this day unusual** (high/low PnL, high/low volume, or both)
2. **Role of sentiment** ({row['sentiment']}) on this day
3. **Likely cause** — was this a market event, sentiment extreme, or position squeeze?
4. **What traders should learn** from this day
Keep it under 200 words, cite the specific numbers."""
                )
            st.markdown(f'<div class="anomaly-box">{explanation.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6 — TRADE STRATEGY ADVISOR
# ═══════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🎯 Trade Strategy Advisor")
    st.markdown("Enter today's Fear & Greed score and your trade parameters — get a personalized AI strategy recommendation backed by historical data.")

    sa1, sa2 = st.columns(2)
    with sa1:
        today_fg      = st.slider("Today's Fear & Greed Score", 0, 100, 50)
        trade_coin    = st.selectbox("Which coin?", ["BTC", "ETH", "SOL", "HYPE", "Other"])
        trade_capital = st.number_input("Available Capital (USD)", 100.0, 1000000.0, 10000.0, 500.0)
    with sa2:
        risk_tolerance = st.select_slider("Risk Tolerance", ["Low", "Medium", "High"])
        holding_period = st.selectbox("Holding Period", ["Scalp (< 1hr)", "Intraday", "Swing (days)", "Position (weeks+)"])
        existing_bias  = st.radio("Your current market bias", ["Bullish", "Bearish", "Neutral"])

    # Compute historical stats for this F&G range
    fg_range    = 15
    hist_subset = closing[
        (closing["fg_value"] >= today_fg - fg_range) &
        (closing["fg_value"] <= today_fg + fg_range)
    ]
    hist_mean_pnl = hist_subset["Closed PnL"].mean() if len(hist_subset) > 0 else 0
    hist_win_rate = hist_subset["win"].mean() * 100 if len(hist_subset) > 0 else 0
    hist_trades   = len(hist_subset)

    # Determine sentiment label
    if today_fg <= 25:
        today_sentiment = "Extreme Fear"
    elif today_fg <= 45:
        today_sentiment = "Fear"
    elif today_fg <= 55:
        today_sentiment = "Neutral"
    elif today_fg <= 75:
        today_sentiment = "Greed"
    else:
        today_sentiment = "Extreme Greed"

    # Show historical context metrics
    m_a, m_b, m_c = st.columns(3)
    m_a.metric("Sentiment Zone", today_sentiment)
    m_b.metric("Hist. Mean PnL (±15 pts)", f"${hist_mean_pnl:.1f}")
    m_c.metric("Hist. Win Rate (±15 pts)", f"{hist_win_rate:.1f}%")

    if st.button("📊 Generate Strategy Recommendation", type="primary"):
        # Full sentiment stats to give AI context
        sent_stats_str = stats[["sentiment", "mean_pnl", "win_rate", "trades"]].to_string(index=False)

        with st.spinner("Building strategy…"):
            strategy = call_claude(
                """You are a senior crypto trading strategist. You give concise, actionable advice
based on quantitative historical data. Always back recommendations with specific numbers.""",
                f"""Generate a trading strategy recommendation for the following situation:

TRADER PROFILE:
- Available Capital: ${trade_capital:,.0f} USD
- Target Coin: {trade_coin}
- Risk Tolerance: {risk_tolerance}
- Holding Period: {holding_period}
- Current Market Bias: {existing_bias}

CURRENT MARKET CONDITIONS:
- Fear & Greed Index: {today_fg} → {today_sentiment}
- Historical performance at similar F&G levels (±15 pts):
  - Trades in dataset: {hist_trades:,}
  - Mean PnL per trade: ${hist_mean_pnl:.2f}
  - Win rate: {hist_win_rate:.1f}%

FULL HISTORICAL SENTIMENT PERFORMANCE TABLE:
{sent_stats_str}

KEY INSIGHT FROM DATA:
- Fear produces highest mean PnL ($139) and win rate (86.7%)
- Greed has most trades (19,785) but lowest quality outcomes
- F&G vs daily PnL correlation: −0.22 (higher sentiment = lower returns)

Provide:
## 1. Position Recommendation
(Long/Short/Stay Out + reason)

## 2. Position Sizing
(How much of the ${trade_capital:,.0f} to deploy, with reasoning)

## 3. Entry Strategy
(Specific conditions based on current F&G = {today_fg})

## 4. Risk Management
(Stop loss %, take profit target, max trades per day)

## 5. Contrarian Signal Check
(Does the data suggest going against the current sentiment? Why/why not?)

Keep each section to 2–3 lines. Be specific and cite historical numbers.""",
                max_tokens=900,
            )
        st.markdown(f'<div class="strategy-box">{strategy.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📚 Historical Strategy Backtester")
    st.markdown("How would a simple contrarian rule have performed? (Buy during Fear, stay out during Greed)")

    contrarian = closing[closing["sentiment"].isin(["Fear", "Extreme Fear"])].copy()
    normal     = closing[closing["sentiment"].isin(["Greed", "Extreme Greed"])].copy()

    bc1, bc2 = st.columns(2)
    with bc1:
        st.metric("Contrarian (Fear only) Mean PnL", f"${contrarian['Closed PnL'].mean():.2f}")
        st.metric("Contrarian Win Rate", f"{contrarian['win'].mean()*100:.1f}%")
        st.metric("Contrarian Trades", f"{len(contrarian):,}")

    with bc2:
        st.metric("Momentum (Greed only) Mean PnL", f"${normal['Closed PnL'].mean():.2f}")
        st.metric("Momentum Win Rate", f"{normal['win'].mean()*100:.1f}%")
        st.metric("Momentum Trades", f"{len(normal):,}")

    # Contrarian vs momentum cumulative PnL chart
    contrarian_daily = contrarian.groupby("date")["Closed PnL"].sum().cumsum().reset_index()
    contrarian_daily.columns = ["date", "cum_pnl"]
    contrarian_daily["Strategy"] = "Contrarian (Fear)"

    momentum_daily = normal.groupby("date")["Closed PnL"].sum().cumsum().reset_index()
    momentum_daily.columns = ["date", "cum_pnl"]
    momentum_daily["Strategy"] = "Momentum (Greed)"

    combined = pd.concat([contrarian_daily, momentum_daily])
    fig = px.line(combined, x="date", y="cum_pnl", color="Strategy",
                  color_discrete_map={"Contrarian (Fear)": "#ff9900", "Momentum (Greed)": "#00c9ff"},
                  title="Cumulative PnL: Contrarian vs Momentum Strategy")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Data: Hyperliquid × CNN Fear & Greed Index</small></center>",
    unsafe_allow_html=True
)
