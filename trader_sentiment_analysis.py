# =============================================================================
# BITCOIN MARKET SENTIMENT vs TRADER PERFORMANCE ANALYSIS
# Datasets: Fear & Greed Index + Hyperliquid Historical Trader Data
# Author  : Vaibhav | MCA Data Science – Symbiosis International University
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ── output folder ──────────────────────────────────────────────────────────────
OUT = Path("output_charts")
OUT.mkdir(exist_ok=True)

# ── global style ───────────────────────────────────────────────────────────────
PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear"        : "#ff7f0e",
    "Neutral"     : "#bcbd22",
    "Greed"       : "#2ca02c",
    "Extreme Greed": "#17becf",
}
ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold",
                      "axes.titlesize": 13, "figure.facecolor": "#f8f9fa"})

# ==============================================================================
# 1. LOAD & CLEAN
# ==============================================================================
print("\n" + "="*65)
print("  STEP 1 ▸ Loading and cleaning datasets")
print("="*65)

# ── Fear & Greed Index ─────────────────────────────────────────────────────────
fg = pd.read_csv("fear_greed_index.csv")
fg["date"] = pd.to_datetime(fg["date"], format="mixed", dayfirst=False).dt.date
fg["value"] = pd.to_numeric(fg["value"], errors="coerce")
fg["classification"] = fg["classification"].str.strip()
fg = fg.dropna(subset=["date", "value", "classification"]).drop_duplicates("date")

# ── Historical Trader Data ─────────────────────────────────────────────────────
ht = pd.read_csv("historical_data.csv")
ht["date"] = pd.to_datetime(
    ht["Timestamp IST"], format="mixed", dayfirst=False
).dt.date
ht["Closed PnL"] = pd.to_numeric(ht["Closed PnL"], errors="coerce").fillna(0)
ht["Size USD"]   = pd.to_numeric(ht["Size USD"],   errors="coerce").fillna(0)
ht["Fee"]        = pd.to_numeric(ht["Fee"],        errors="coerce").fillna(0)
ht["Side"]       = ht["Side"].str.strip().str.upper()
ht["Direction"]  = ht["Direction"].str.strip()

# Closing trades are the ones that generate realized PnL
closing_directions = ["Close Long", "Close Short", "Long > Short",
                      "Short > Long", "Settlement", "Auto-Deleveraging",
                      "Liquidated Isolated Short"]
ht["is_closing"] = ht["Direction"].isin(closing_directions)
ht["is_winner"]  = ht["Closed PnL"] > 0

print(f"  Fear & Greed rows  : {len(fg):,}  |  date range : {fg['date'].min()} → {fg['date'].max()}")
print(f"  Trader data rows   : {len(ht):,}  |  date range : {ht['date'].min()} → {ht['date'].max()}")
print(f"  Unique traders     : {ht['Account'].nunique()}")
print(f"  Unique coins       : {ht['Coin'].nunique()}")
print(f"  Closing trades     : {ht['is_closing'].sum():,}")

# ==============================================================================
# 2. MERGE ON DATE
# ==============================================================================
print("\n" + "="*65)
print("  STEP 2 ▸ Merging on date")
print("="*65)

merged = ht.merge(
    fg[["date", "value", "classification"]].rename(
        columns={"value": "fg_value", "classification": "sentiment"}
    ),
    on="date", how="inner"
)
merged["sentiment"] = pd.Categorical(merged["sentiment"], categories=ORDER, ordered=True)

closing = merged[merged["is_closing"]].copy()

print(f"  Matched rows       : {len(merged):,}")
print(f"  Matched dates      : {merged['date'].nunique()}")
print(f"  Closing trades (merged): {len(closing):,}")
print(f"  Sentiment distribution:\n{merged.groupby('sentiment', observed=True).size().to_string()}")

# ==============================================================================
# 3. EDA  –  FEAR & GREED INDEX
# ==============================================================================
print("\n" + "="*65)
print("  STEP 3 ▸ EDA – Fear & Greed Index")
print("="*65)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Bitcoin Fear & Greed Index – Exploratory Analysis", fontsize=15, fontweight="bold")

# 3A: Distribution of sentiment classes
counts = fg["classification"].value_counts().reindex(ORDER)
bars = axes[0].bar(ORDER, counts, color=[PALETTE[o] for o in ORDER], edgecolor="white", linewidth=0.8)
axes[0].set_title("Sentiment Class Distribution")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Number of Days")
axes[0].tick_params(axis="x", rotation=30)
for bar, val in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 str(val), ha="center", fontsize=10, fontweight="bold")

# 3B: Fear & Greed value over time (rolling 30-day average)
fg_sorted = fg.sort_values("date")
fg_sorted["rolling_30"] = fg_sorted["value"].rolling(30).mean()
axes[1].fill_between(range(len(fg_sorted)), fg_sorted["value"],
                     alpha=0.2, color="steelblue")
axes[1].plot(range(len(fg_sorted)), fg_sorted["rolling_30"],
             color="steelblue", linewidth=1.8, label="30-day avg")
axes[1].axhline(25, color="#ff7f0e", linestyle="--", linewidth=1, label="Fear threshold (25)")
axes[1].axhline(75, color="#2ca02c", linestyle="--", linewidth=1, label="Greed threshold (75)")
axes[1].set_title("Fear & Greed Value Over Time")
axes[1].set_ylabel("Index Value (0–100)")
axes[1].set_xlabel("Days Since Jan 2018")
axes[1].legend(fontsize=9)

# 3C: Boxplot of index value by sentiment
data_by_sentiment = [fg[fg["classification"] == s]["value"].values for s in ORDER]
bp = axes[2].boxplot(data_by_sentiment, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2))
for patch, s in zip(bp["boxes"], ORDER):
    patch.set_facecolor(PALETTE[s])
axes[2].set_xticks(range(1, len(ORDER)+1))
axes[2].set_xticklabels(ORDER, rotation=25)
axes[2].set_title("Index Value Distribution by Sentiment")
axes[2].set_ylabel("Index Value")

plt.tight_layout()
plt.savefig(OUT / "01_fear_greed_eda.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 01_fear_greed_eda.png")

# ==============================================================================
# 4. EDA  –  TRADER DATA
# ==============================================================================
print("\n" + "="*65)
print("  STEP 4 ▸ EDA – Hyperliquid Trader Data")
print("="*65)

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Hyperliquid Trader Data – Exploratory Analysis", fontsize=15, fontweight="bold")

# 4A: Trade count by coin (top 15)
top_coins = ht["Coin"].value_counts().head(15)
axes[0,0].barh(top_coins.index[::-1], top_coins.values[::-1], color="steelblue", edgecolor="white")
axes[0,0].set_title("Top 15 Most Traded Coins")
axes[0,0].set_xlabel("Number of Trades")

# 4B: Trade direction distribution
dir_counts = ht["Direction"].value_counts().head(10)
axes[0,1].barh(dir_counts.index[::-1], dir_counts.values[::-1], color="mediumpurple", edgecolor="white")
axes[0,1].set_title("Trade Directions (Top 10)")
axes[0,1].set_xlabel("Count")

# 4C: Daily trade volume (USD)
daily_vol = ht.groupby("date")["Size USD"].sum().reset_index()
daily_vol["date"] = pd.to_datetime(daily_vol["date"])
daily_vol = daily_vol.sort_values("date")
axes[0,2].fill_between(daily_vol["date"], daily_vol["Size USD"]/1e6,
                       alpha=0.7, color="teal")
axes[0,2].set_title("Daily Trade Volume (USD)")
axes[0,2].set_ylabel("Volume (Millions USD)")
axes[0,2].tick_params(axis="x", rotation=30)

# 4D: PnL distribution (clipped for readability)
pnl_vals = closing["Closed PnL"]
pnl_clip = pnl_vals.clip(-5000, 5000)
axes[1,0].hist(pnl_clip, bins=100, color="darkorange", edgecolor="white", alpha=0.85)
axes[1,0].axvline(0, color="red", linewidth=1.5, linestyle="--")
axes[1,0].set_title("Closed PnL Distribution (clipped ±$5k)")
axes[1,0].set_xlabel("Closed PnL (USD)")
axes[1,0].set_ylabel("Frequency")

# 4E: BUY vs SELL ratio
side_counts = ht["Side"].value_counts()
axes[1,1].pie(side_counts, labels=side_counts.index, autopct="%1.1f%%",
              colors=["#2ca02c", "#d62728"], startangle=90,
              wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1,1].set_title("Buy vs Sell Trade Ratio")

# 4F: PnL per trader (total)
trader_pnl = closing.groupby("Account")["Closed PnL"].sum().sort_values()
colors_pnl = ["#d62728" if v < 0 else "#2ca02c" for v in trader_pnl.values]
axes[1,2].barh(range(len(trader_pnl)), trader_pnl.values, color=colors_pnl)
axes[1,2].axvline(0, color="black", linewidth=1)
axes[1,2].set_yticks(range(len(trader_pnl)))
axes[1,2].set_yticklabels(
    [f"Trader {i+1}" for i in range(len(trader_pnl))], fontsize=8
)
axes[1,2].set_title("Total Closed PnL per Trader")
axes[1,2].set_xlabel("Total PnL (USD)")

plt.tight_layout()
plt.savefig(OUT / "02_trader_eda.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 02_trader_eda.png")

# ==============================================================================
# 5. SENTIMENT vs PnL  –  CORE ANALYSIS
# ==============================================================================
print("\n" + "="*65)
print("  STEP 5 ▸ Sentiment vs Trader PnL")
print("="*65)

# ── 5A: Summary stats per sentiment ───────────────────────────────────────────
pnl_by_sent = (
    closing.groupby("sentiment", observed=True)["Closed PnL"]
    .agg(
        Total_PnL="sum",
        Mean_PnL="mean",
        Median_PnL="median",
        Std_PnL="std",
        Trade_Count="count",
        Win_Count=lambda x: (x > 0).sum(),
    )
    .assign(Win_Rate=lambda df: df["Win_Count"] / df["Trade_Count"] * 100)
)
print("\n  PnL by Sentiment:\n")
print(pnl_by_sent.to_string(float_format="{:,.2f}".format))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Sentiment vs Trader Performance – Core Analysis", fontsize=15, fontweight="bold")

# Chart A: Mean PnL per trade by sentiment
bar_colors = [PALETTE[s] for s in pnl_by_sent.index]
axes[0,0].bar(pnl_by_sent.index, pnl_by_sent["Mean_PnL"],
              color=bar_colors, edgecolor="white", linewidth=0.8)
axes[0,0].axhline(0, color="black", linewidth=1)
axes[0,0].set_title("Mean Closed PnL per Trade by Sentiment")
axes[0,0].set_ylabel("Mean PnL (USD)")
axes[0,0].tick_params(axis="x", rotation=20)
for i, (idx, row) in enumerate(pnl_by_sent.iterrows()):
    axes[0,0].text(i, row["Mean_PnL"] + (8 if row["Mean_PnL"] >= 0 else -18),
                   f"${row['Mean_PnL']:,.1f}", ha="center", fontsize=9, fontweight="bold")

# Chart B: Win Rate by sentiment
axes[0,1].bar(pnl_by_sent.index, pnl_by_sent["Win_Rate"],
              color=bar_colors, edgecolor="white", linewidth=0.8)
axes[0,1].axhline(50, color="red", linewidth=1.5, linestyle="--", label="50% baseline")
axes[0,1].set_title("Win Rate (%) by Sentiment")
axes[0,1].set_ylabel("Win Rate (%)")
axes[0,1].tick_params(axis="x", rotation=20)
axes[0,1].legend()
for i, (idx, row) in enumerate(pnl_by_sent.iterrows()):
    axes[0,1].text(i, row["Win_Rate"] + 0.5,
                   f"{row['Win_Rate']:.1f}%", ha="center", fontsize=9, fontweight="bold")

# Chart C: Total PnL by sentiment
axes[1,0].bar(pnl_by_sent.index, pnl_by_sent["Total_PnL"] / 1e6,
              color=bar_colors, edgecolor="white", linewidth=0.8)
axes[1,0].axhline(0, color="black", linewidth=1)
axes[1,0].set_title("Total Cumulative PnL by Sentiment")
axes[1,0].set_ylabel("Total PnL (Millions USD)")
axes[1,0].tick_params(axis="x", rotation=20)

# Chart D: PnL boxplot by sentiment (clipped)
closing_plot = closing.copy()
closing_plot["PnL_clipped"] = closing_plot["Closed PnL"].clip(-2000, 2000)
sns.boxplot(data=closing_plot, x="sentiment", y="PnL_clipped",
            order=ORDER, palette=PALETTE, ax=axes[1,1], showfliers=False)
axes[1,1].axhline(0, color="red", linewidth=1.2, linestyle="--")
axes[1,1].set_title("PnL Distribution by Sentiment (clipped ±$2k, no outliers)")
axes[1,1].set_ylabel("Closed PnL (USD)")
axes[1,1].set_xlabel("Sentiment")
axes[1,1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUT / "03_sentiment_vs_pnl.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 03_sentiment_vs_pnl.png")

# ==============================================================================
# 6. TRADE VOLUME & BEHAVIOUR BY SENTIMENT
# ==============================================================================
print("\n" + "="*65)
print("  STEP 6 ▸ Trade Volume & Behaviour by Sentiment")
print("="*65)

vol_by_sent = (
    merged.groupby("sentiment", observed=True)
    .agg(
        Total_Volume_USD=("Size USD", "sum"),
        Avg_Trade_Size=("Size USD", "mean"),
        Trade_Count=("Size USD", "count"),
    )
)
print("\n  Volume by Sentiment:\n")
print(vol_by_sent.to_string(float_format="{:,.2f}".format))

# BUY / SELL ratio per sentiment
buy_sell = (
    merged.groupby(["sentiment", "Side"], observed=True)
    .size()
    .unstack(fill_value=0)
)
buy_sell["Buy_Pct"] = buy_sell.get("BUY", 0) / buy_sell.sum(axis=1) * 100

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Trading Behaviour & Volume by Sentiment", fontsize=15, fontweight="bold")

# Chart A: Trade count per sentiment
axes[0].bar(vol_by_sent.index, vol_by_sent["Trade_Count"],
            color=[PALETTE[s] for s in vol_by_sent.index], edgecolor="white")
axes[0].set_title("Number of Trades by Sentiment")
axes[0].set_ylabel("Trade Count")
axes[0].tick_params(axis="x", rotation=20)

# Chart B: Avg trade size per sentiment
axes[1].bar(vol_by_sent.index, vol_by_sent["Avg_Trade_Size"],
            color=[PALETTE[s] for s in vol_by_sent.index], edgecolor="white")
axes[1].set_title("Average Trade Size (USD) by Sentiment")
axes[1].set_ylabel("Avg Trade Size (USD)")
axes[1].tick_params(axis="x", rotation=20)

# Chart C: Buy % by sentiment (stacked)
buy_vals = buy_sell.get("BUY", pd.Series([0]*len(ORDER), index=ORDER)).reindex(ORDER)
sell_vals = buy_sell.get("SELL", pd.Series([0]*len(ORDER), index=ORDER)).reindex(ORDER)
total_bs  = buy_vals + sell_vals
buy_pct   = buy_vals / total_bs * 100
sell_pct  = sell_vals / total_bs * 100
x = range(len(ORDER))
axes[2].bar(x, buy_pct,  label="BUY",  color="#2ca02c", edgecolor="white")
axes[2].bar(x, sell_pct, label="SELL", color="#d62728", edgecolor="white",
            bottom=buy_pct)
axes[2].axhline(50, color="white", linewidth=1.2, linestyle="--")
axes[2].set_xticks(x)
axes[2].set_xticklabels(ORDER, rotation=20)
axes[2].set_title("Buy vs Sell % by Sentiment")
axes[2].set_ylabel("Percentage (%)")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT / "04_volume_behaviour.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 04_volume_behaviour.png")

# ==============================================================================
# 7. TOP TRADERS  –  WHO PROFITS IN WHICH SENTIMENT?
# ==============================================================================
print("\n" + "="*65)
print("  STEP 7 ▸ Top Trader Performance across Sentiment Regimes")
print("="*65)

# Shorten account addresses for display
closing["short_acct"] = closing["Account"].str[:8] + "…"
trader_labels = closing.groupby("Account")["short_acct"].first()

# Trader PnL per sentiment (pivot)
trader_sent_pnl = (
    closing.groupby(["Account", "sentiment"], observed=True)["Closed PnL"]
    .sum()
    .unstack(fill_value=0)
)
trader_sent_pnl.index = [a[:8]+"…" for a in trader_sent_pnl.index]
trader_sent_pnl = trader_sent_pnl[ORDER]

# Sort by total PnL
trader_sent_pnl["Total"] = trader_sent_pnl.sum(axis=1)
trader_sent_pnl = trader_sent_pnl.sort_values("Total", ascending=False).drop(columns="Total")

fig, ax = plt.subplots(figsize=(18, 9))
trader_sent_pnl.plot(
    kind="bar", ax=ax, color=[PALETTE[s] for s in ORDER],
    edgecolor="white", linewidth=0.5, width=0.8
)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Total Closed PnL per Trader by Sentiment Regime", fontsize=14, fontweight="bold")
ax.set_xlabel("Trader (first 8 chars of address)")
ax.set_ylabel("Total PnL (USD)")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout()
plt.savefig(OUT / "05_trader_by_sentiment.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 05_trader_by_sentiment.png")

# ==============================================================================
# 8. COIN PREFERENCE BY SENTIMENT
# ==============================================================================
print("\n" + "="*65)
print("  STEP 8 ▸ Coin Preferences by Sentiment")
print("="*65)

top_n_coins = merged["Coin"].value_counts().head(12).index

coin_sent = (
    merged[merged["Coin"].isin(top_n_coins)]
    .groupby(["Coin", "sentiment"], observed=True)
    .size()
    .unstack(fill_value=0)[ORDER]
)
coin_sent_pct = coin_sent.div(coin_sent.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Coin Trading Patterns by Sentiment", fontsize=15, fontweight="bold")

coin_sent_pct.plot(
    kind="barh", ax=axes[0], color=[PALETTE[s] for s in ORDER],
    edgecolor="white", linewidth=0.5, width=0.8
)
axes[0].set_title("Sentiment Mix per Coin (Top 12 – % of trades)")
axes[0].set_xlabel("% of Trades")
axes[0].legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
axes[0].axvline(50, color="white", linewidth=1, linestyle="--")

# Coin-level PnL heatmap
coin_pnl = (
    closing[closing["Coin"].isin(top_n_coins)]
    .groupby(["Coin", "sentiment"], observed=True)["Closed PnL"]
    .mean()
    .unstack(fill_value=0)[ORDER]
)
sns.heatmap(
    coin_pnl, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
    ax=axes[1], linewidths=0.5, cbar_kws={"label": "Mean PnL (USD)"}
)
axes[1].set_title("Mean PnL per Trade – Coin × Sentiment Heatmap")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("Coin")

plt.tight_layout()
plt.savefig(OUT / "06_coin_sentiment.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 06_coin_sentiment.png")

# ==============================================================================
# 9. DAILY PnL TREND OVERLAID WITH SENTIMENT
# ==============================================================================
print("\n" + "="*65)
print("  STEP 9 ▸ Daily PnL Trend vs Market Sentiment")
print("="*65)

daily = (
    closing.groupby("date")
    .agg(
        Daily_PnL=("Closed PnL", "sum"),
        Trade_Count=("Closed PnL", "count"),
    )
    .reset_index()
)
daily = daily.merge(
    fg[["date", "value", "classification"]],
    on="date", how="left"
)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date")
daily["Cumulative_PnL"] = daily["Daily_PnL"].cumsum()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
fig.suptitle("Daily Trader Performance vs Bitcoin Sentiment Timeline",
             fontsize=15, fontweight="bold")

# Background shading by sentiment
sent_colors = {"Extreme Fear": "#ffcccc", "Fear": "#ffe0cc",
               "Neutral": "#fffacc", "Greed": "#ccffcc", "Extreme Greed": "#ccf5ff"}
prev_date = daily["date"].iloc[0]
prev_sent = daily["classification"].iloc[0]
for _, row in daily.iterrows():
    if row["classification"] != prev_sent:
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(prev_date, row["date"],
                       color=sent_colors.get(prev_sent, "white"), alpha=0.35)
        prev_date = row["date"]
        prev_sent = row["classification"]
for ax in [ax1, ax2, ax3]:
    ax.axvspan(prev_date, daily["date"].iloc[-1],
               color=sent_colors.get(prev_sent, "white"), alpha=0.35)

# Panel 1: Fear & Greed value
ax1.plot(daily["date"], daily["value"], color="steelblue", linewidth=1.5)
ax1.axhline(50, color="gray", linewidth=0.8, linestyle="--")
ax1.set_ylabel("F&G Index")
ax1.set_title("Fear & Greed Index", loc="left")

# Panel 2: Daily PnL
bar_c = ["#2ca02c" if v >= 0 else "#d62728" for v in daily["Daily_PnL"]]
ax2.bar(daily["date"], daily["Daily_PnL"], color=bar_c, width=1.0)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Daily PnL (USD)")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax2.set_title("Aggregate Daily Trader PnL", loc="left")

# Panel 3: Cumulative PnL
ax3.fill_between(daily["date"], daily["Cumulative_PnL"],
                 where=daily["Cumulative_PnL"] >= 0,
                 color="#2ca02c", alpha=0.6, label="Profit")
ax3.fill_between(daily["date"], daily["Cumulative_PnL"],
                 where=daily["Cumulative_PnL"] < 0,
                 color="#d62728", alpha=0.6, label="Loss")
ax3.plot(daily["date"], daily["Cumulative_PnL"], color="navy", linewidth=1.5)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_ylabel("Cumulative PnL (USD)")
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax3.set_title("Cumulative Trader PnL over Time", loc="left")
ax3.legend()

# Legend for background shading
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=sent_colors[s], alpha=0.5, label=s) for s in ORDER]
ax1.legend(handles=legend_els, loc="upper left", fontsize=8,
           title="Sentiment Background", framealpha=0.8)

plt.tight_layout()
plt.savefig(OUT / "07_daily_pnl_timeline.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 07_daily_pnl_timeline.png")

# ==============================================================================
# 10. FEE ANALYSIS BY SENTIMENT
# ==============================================================================
print("\n" + "="*65)
print("  STEP 10 ▸ Fee Analysis by Sentiment")
print("="*65)

fee_by_sent = (
    merged.groupby("sentiment", observed=True)["Fee"]
    .agg(Total_Fee="sum", Mean_Fee="mean", Trade_Count="count")
)
print("\n  Fee by Sentiment:\n")
print(fee_by_sent.to_string(float_format="{:,.4f}".format))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fee Burden by Market Sentiment", fontsize=15, fontweight="bold")

axes[0].bar(fee_by_sent.index, fee_by_sent["Mean_Fee"],
            color=[PALETTE[s] for s in fee_by_sent.index], edgecolor="white")
axes[0].set_title("Mean Fee per Trade by Sentiment")
axes[0].set_ylabel("Mean Fee (USD)")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(fee_by_sent.index, fee_by_sent["Total_Fee"] / 1e3,
            color=[PALETTE[s] for s in fee_by_sent.index], edgecolor="white")
axes[1].set_title("Total Fees Paid by Sentiment (thousands USD)")
axes[1].set_ylabel("Total Fee (Thousands USD)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUT / "08_fee_analysis.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 08_fee_analysis.png")

# ==============================================================================
# 11. CORRELATION ANALYSIS
# ==============================================================================
print("\n" + "="*65)
print("  STEP 11 ▸ Correlation – FG Index vs Daily Metrics")
print("="*65)

daily_corr = daily[["value", "Daily_PnL", "Trade_Count"]].dropna()
corr_pnl   = daily_corr["value"].corr(daily_corr["Daily_PnL"])
corr_count = daily_corr["value"].corr(daily_corr["Trade_Count"])
print(f"\n  Pearson r (F&G value vs Daily PnL)       : {corr_pnl:.4f}")
print(f"  Pearson r (F&G value vs Trade Count)     : {corr_count:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Correlation: Fear & Greed Index vs Daily Metrics",
             fontsize=15, fontweight="bold")

for ax, col, label, color in zip(
    axes,
    ["Daily_PnL", "Trade_Count"],
    ["Daily PnL (USD)", "Daily Trade Count"],
    ["darkorange", "steelblue"]
):
    ax.scatter(daily_corr["value"], daily_corr[col],
               alpha=0.4, color=color, edgecolors="white", linewidth=0.3, s=40)
    m, b = np.polyfit(daily_corr["value"], daily_corr[col], 1)
    x_line = np.linspace(daily_corr["value"].min(), daily_corr["value"].max(), 100)
    ax.plot(x_line, m*x_line+b, color="black", linewidth=2, linestyle="--")
    r = daily_corr["value"].corr(daily_corr[col])
    ax.set_title(f"F&G Index vs {label}\n(r = {r:.3f})")
    ax.set_xlabel("Fear & Greed Index Value")
    ax.set_ylabel(label)

plt.tight_layout()
plt.savefig(OUT / "09_correlation.png", bbox_inches="tight")
plt.close()
print("  ✓ Chart saved: 09_correlation.png")

# ==============================================================================
# 12. KEY INSIGHTS SUMMARY
# ==============================================================================
print("\n" + "="*65)
print("  STEP 12 ▸ Key Insights Summary")
print("="*65)

best_pnl_sent  = pnl_by_sent["Mean_PnL"].idxmax()
worst_pnl_sent = pnl_by_sent["Mean_PnL"].idxmin()
best_wr_sent   = pnl_by_sent["Win_Rate"].idxmax()
most_active    = pnl_by_sent["Trade_Count"].idxmax()

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    KEY FINDINGS                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. BEST avg PnL/trade : {best_pnl_sent:<15}  (${pnl_by_sent.loc[best_pnl_sent,"Mean_PnL"]:>8,.2f} per trade)  │
  │  2. WORST avg PnL/trade: {worst_pnl_sent:<15}  (${pnl_by_sent.loc[worst_pnl_sent,"Mean_PnL"]:>8,.2f} per trade)  │
  │  3. HIGHEST win rate   : {best_wr_sent:<15}  ({pnl_by_sent.loc[best_wr_sent,"Win_Rate"]:>5.1f}%)              │
  │  4. MOST active period : {most_active:<15}  ({pnl_by_sent.loc[most_active,"Trade_Count"]:>7,} trades)        │
  │  5. F&G vs Daily PnL corr    : {corr_pnl:>+.4f}                       │
  │  6. F&G vs Trade Count corr  : {corr_count:>+.4f}                       │
  │  7. Total matched trades     : {len(closing):>7,}                       │
  │  8. Unique traders analysed  : {closing["Account"].nunique():>7}                       │
  └─────────────────────────────────────────────────────────────────┘

  INSIGHTS:
  • Traders generate {("better" if pnl_by_sent.loc["Greed","Mean_PnL"] > pnl_by_sent.loc["Fear","Mean_PnL"] else "worse")} average PnL during Greed than Fear.
  • Win rate {"rises" if pnl_by_sent.loc["Greed","Win_Rate"] > pnl_by_sent.loc["Fear","Win_Rate"] else "drops"} as sentiment moves from Fear → Greed.
  • A {"positive" if corr_pnl > 0 else "negative"} correlation (r={corr_pnl:.3f}) between the F&G index and
    daily PnL suggests traders perform {"better" if corr_pnl > 0 else "worse"} in bullish sentiment.
  • Trade activity is {"highest" if most_active in ["Greed","Extreme Greed"] else "not highest"} during Greed phases, indicating
    sentiment drives participation, not just returns.

  All charts saved to: ./{OUT}/
""")

print("="*65)
print("  ✅  Analysis complete!")
print("="*65 + "\n")
