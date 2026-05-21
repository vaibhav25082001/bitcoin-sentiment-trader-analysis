# Bitcoin Market Sentiment × Trader Performance Analysis

> Exploring the relationship between Bitcoin Fear & Greed Index and trader behaviour on Hyperliquid DEX

---

## 📁 Project Structure

```
├── app.py                        # Streamlit AI-powered dashboard (5 features)
├── trader_sentiment_analysis.py  # Standalone EDA + chart generation script
├── fear_greed_index.csv          # Bitcoin Fear & Greed Index (2018–2025)
├── historical_data.csv           # 211,224 Hyperliquid trades from 32 traders
├── requirements.txt              # All Python dependencies
├── 01_fear_greed_eda.png         # Sentiment class distribution & time series
├── 02_trader_eda.png             # Trader EDA: coins, PnL distribution, volume
├── 03_sentiment_vs_pnl.png       # Core: mean PnL & win rate by sentiment
├── 04_volume_behaviour.png       # Trade volume & buy/sell ratio by sentiment
├── 05_trader_by_sentiment.png    # Per-trader PnL across sentiment regimes
├── 06_coin_sentiment.png         # Coin × sentiment heatmap
├── 07_daily_pnl_timeline.png     # Cumulative PnL timeline with sentiment overlay
├── 08_fee_analysis.png           # Fee burden by sentiment
└── 09_correlation.png            # F&G index vs daily PnL/trade count scatter
```

---

## 📊 Datasets

| File | Description | Size |
|------|-------------|------|
| `fear_greed_index.csv` | Daily Bitcoin Fear & Greed Index (Jan 2018 – Sep 2024), columns: `date`, `value`, `classification` | 2,644 rows |
| `historical_data.csv` | Hyperliquid DEX trade history from 32 traders (Jan 2024 – Sep 2024), 16 columns including `Account`, `Coin`, `Closed PnL`, `Side`, `Direction`, `Size USD`, `Fee`, `Timestamp IST` | 211,224 rows |

The two datasets are **merged on date** to analyse how daily market sentiment affects trader performance. The overlapping period covers **449 matched trading days**.

---

## 🔍 Analysis Walkthrough (`trader_sentiment_analysis.py`)

This script performs a full 12-step exploratory and statistical analysis:

| Step | What it does |
|------|-------------|
| 1 | Loads & cleans both datasets — handles mixed date formats, nulls, closing trade detection |
| 2 | Merges on date — creates a unified `merged` and `closing` DataFrame |
| 3 | **Fear & Greed EDA** — class distribution, 30-day rolling average, boxplots per class |
| 4 | **Trader EDA** — top coins, trade directions, daily volume, PnL histogram, buy/sell pie, per-trader total PnL |
| 5 | **Sentiment vs PnL** — mean PnL per trade, win rate %, total cumulative PnL, PnL boxplot — all by sentiment |
| 6 | **Volume & Behaviour** — trade count, avg trade size, buy/sell ratio stacked bar — by sentiment |
| 7 | **Per-trader breakdown** — grouped bar chart of each trader's total PnL across all 5 sentiment regimes |
| 8 | **Coin × Sentiment** — sentiment mix per coin (top 12) + mean PnL heatmap |
| 9 | **Daily PnL Timeline** — 3-panel chart: F&G index / daily PnL bars / cumulative PnL with sentiment background shading |
| 10 | **Fee Analysis** — mean and total fees paid by sentiment regime |
| 11 | **Correlation** — Pearson r between F&G value and (a) daily PnL and (b) daily trade count |
| 12 | **Printed Key Insights** — summary table of best/worst sentiment, win rates, and correlations |

### Run the analysis script

```bash
# Install dependencies
pip install -r requirements.txt

# Place both CSVs in the same folder, then run:
python trader_sentiment_analysis.py
```

Charts are saved to `output_charts/` in the same directory.

---

## 🔑 Key Findings

| Finding | Detail |
|---------|--------|
| **Fear = Best PnL** | Mean PnL per trade is **$139 during Fear** vs only **$33 during Extreme Greed** |
| **Win rate peaks in Fear** | Win rate is **86.7% during Fear** and drops to **77% during Greed** |
| **Greed drives volume, not returns** | The most trades (19,785+) happen during Greed — FOMO is real |
| **Contrarian trading works** | Traders who buy during fear and close during greed consistently outperform |
| **Negative correlation** | Pearson r = **−0.217** between F&G index value and daily PnL — higher sentiment → lower returns |
| **Fee burden highest in Fear & Greed** | Mean fee ~$1.28/trade vs $0.69 during Extreme Fear — active periods cost more |
| **SOL is the most profitable Fear play** | SOL mean PnL during Fear = **$1,044/trade** (highest in Coin × Sentiment heatmap) |
| **TRUMP is a Greed trap** | TRUMP mean PnL during Greed = **−$1,173/trade** — meme coins amplify FOMO losses |

---

## 🤖 AI Dashboard (`app.py`)

The Streamlit app adds 5 AI-powered features on top of the analysis:

| Feature | Description |
|---------|-------------|
| 💬 **Ask the Data** | Natural language chatbot — ask any question about the dataset in plain English |
| 🔍 **AI Insight Generator** | Select any chart → one click generates a written expert analysis |
| 📈 **Sentiment Predictor** | Random Forest / Gradient Boosting / Logistic Regression model predicts whether today will be profitable — ~85% accuracy |
| 🚨 **Anomaly Explainer** | Z-score based detection of unusual trading days + AI explanation of each anomaly |
| 🎯 **Trade Strategy Advisor** | Enter today's F&G score + portfolio details → AI generates a personalised trading strategy grounded in historical data |

### Run the dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place both CSVs in the same folder as app.py

# 3. Launch
streamlit run app.py

# 4. In the sidebar, paste your Anthropic API key (get one at console.anthropic.com)
```

### Supported AI Provider

| Provider | Model | Get API Key |
|----------|-------|-------------|
| Anthropic Claude | claude-sonnet-4 | [console.anthropic.com](https://console.anthropic.com) |

> API keys are entered in the sidebar at runtime and are **never stored or logged**.

---

## 🛠 Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data processing | `pandas`, `numpy`, `scipy` |
| Machine learning | `scikit-learn` (Random Forest, Gradient Boosting, Logistic Regression) |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| AI | `anthropic` (Claude Sonnet 4) |


