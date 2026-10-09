"""
app.py -- SEM & Landing-Page A/B Test Analyzer (Streamlit)

A portfolio project for a Digital Marketing Analyst context: paid-search KPIs,
a landing-page/ad-copy A/B test analyzer (two-proportion z-test), campaign
performance scanning, and a budget-reallocation recommender.

Charts follow Cole Nussbaumer Knaflic's "Storytelling with Data": muted
palette with a single accent for the point being made, direct labels instead
of busy legends, no gridline/chartjunk clutter, and a clear takeaway per view.

ALL DATA IS SYNTHETIC (see data_gen.py). Not real IXL Learning data.

Author: Leo Chan -- github.com/GoldenCrate
Run with:  streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import analytics as an
import data_gen

# --------------------------------------------------------------------------
# Knaflic-style palette: grey for context, one blue accent for the focus.
# --------------------------------------------------------------------------
GREY = "#B8B8B8"
GREY_DARK = "#7A7A7A"
ACCENT = "#1B6CA8"      # focus / "good"
WARN = "#C8553D"        # under-performer / "bad"
POSITIVE = "#2E7D5B"    # scale up / win
INK = "#2B2B2B"

st.set_page_config(page_title="SEM & Landing-Page A/B Test Analyzer",
                   page_icon="chart", layout="wide")


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = data_gen.generate_campaign_data()
    summary, visitors = data_gen.generate_ab_test_data()
    return df, summary, visitors


df, ab_summary, ab_visitors = load_data()


def fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%"


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("SEM & A/B Analyzer")
st.sidebar.caption("Digital Marketing Analyst portfolio demo")
view = st.sidebar.radio(
    "View",
    ["Overview KPIs", "A/B Test Analyzer", "Campaign Performance",
     "Spend Efficiency", "About"],
)
st.sidebar.markdown("---")
st.sidebar.warning("All data is SYNTHETIC -- generated for demo purposes only. "
                   "Not real IXL Learning data.")
st.sidebar.caption("Leo Chan - github.com/GoldenCrate")


def so_what(text: str):
    """Render a consistent 'So what / recommendation' callout."""
    st.markdown(
        f"<div style='background:#EAF2F8;border-left:5px solid {ACCENT};"
        f"padding:0.8rem 1rem;border-radius:4px;margin-top:0.5rem'>"
        f"<b>So what &rarr;</b> {text}</div>",
        unsafe_allow_html=True,
    )


# ==========================================================================
# VIEW 1 -- Overview KPIs
# ==========================================================================
if view == "Overview KPIs":
    st.title("Account KPIs")
    st.caption("Synthetic paid-search account, 60 days, 6 campaigns.")

    imp = df["impressions"].sum()
    clk = df["clicks"].sum()
    conv = df["conversions"].sum()
    spend = df["spend"].sum()
    rev = df["revenue"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Impressions", f"{imp:,.0f}")
    c2.metric("CTR", fmt_pct(an.ctr(clk, imp)))
    c3.metric("Conversion rate", fmt_pct(an.conversion_rate(conv, clk)))
    c4.metric("CPA", f"${an.cpa(spend, conv):,.2f}")
    c5.metric("ROAS", f"{an.roas(rev, spend):.2f}x")

    st.markdown(" ")
    c6, c7, c8 = st.columns(3)
    c6.metric("Total spend", fmt_money(spend))
    c7.metric("Total revenue", fmt_money(rev))
    c8.metric("Conversions", f"{conv:,.0f}")

    st.subheader("Daily conversions trend")
    daily = df.groupby("date", as_index=False)["conversions"].sum()
    line = (
        alt.Chart(daily)
        .mark_line(color=ACCENT, strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("conversions:Q", title="Conversions / day",
                    axis=alt.Axis(grid=False)),
            tooltip=["date:T", "conversions:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(line, width="stretch")

    portfolio_roas = an.roas(rev, spend)
    so_what(
        f"The account returns <b>{portfolio_roas:.2f}x</b> on spend at a "
        f"<b>${an.cpa(spend, conv):,.2f}</b> CPA. That is a healthy blended "
        f"position, but the blended number hides wide spread by campaign -- "
        f"see the Campaign Performance and Spend Efficiency views for where "
        f"to push and pull budget."
    )


# ==========================================================================
# VIEW 2 -- A/B Test Analyzer
# ==========================================================================
elif view == "A/B Test Analyzer":
    st.title("Landing-Page / Ad-Copy A/B Test")
    st.caption(f"{ab_summary['variant_a_name']}  vs.  {ab_summary['variant_b_name']}")

    with st.expander("Test parameters", expanded=True):
        colp1, colp2 = st.columns(2)
        alpha = colp1.selectbox("Significance level (alpha)", [0.10, 0.05, 0.01],
                                index=1)
        practical = colp2.slider(
            "Practical-impact bar (minimum relative lift to act on, %)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.5)

    r = an.two_proportion_ztest(
        ab_summary["conv_a"], ab_summary["n_a"],
        ab_summary["conv_b"], ab_summary["n_b"],
        alpha=alpha,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control A conv. rate", fmt_pct(r.rate_a))
    c2.metric("Variant B conv. rate", fmt_pct(r.rate_b),
              delta=f"{r.rel_lift_pct:+.1f}% rel")
    c3.metric("p-value", f"{r.p_value:.4f}")
    c4.metric("Winner", r.winner)

    st.subheader("Conversion rate by variant")
    bar_df = pd.DataFrame({
        "variant": ["A (control)", "B (variant)"],
        "rate": [r.rate_a, r.rate_b],
        "focus": ["context", "focus"],
    })
    bars = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X("variant:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rate:Q", title="Conversion rate",
                    axis=alt.Axis(format="%", grid=False)),
            color=alt.Color("focus:N",
                            scale=alt.Scale(domain=["context", "focus"],
                                            range=[GREY, ACCENT]),
                            legend=None),
            tooltip=[alt.Tooltip("rate:Q", format=".2%")],
        )
        .properties(height=300)
    )
    labels = bars.mark_text(dy=-8, color=INK, fontWeight="bold").encode(
        text=alt.Text("rate:Q", format=".2%"))
    st.altair_chart(bars + labels, width="stretch")

    st.subheader("95% confidence interval on absolute lift")
    st.write(
        f"Absolute lift: **{r.abs_lift * 100:.2f} pp** "
        f"(95% CI: {r.ci_low * 100:.2f} to {r.ci_high * 100:.2f} pp). "
        f"z = {r.z_score:.2f}."
    )

    st.subheader("Recommendation")
    st.info(an.interpret_ab(r, practical_threshold_pct=practical))
    so_what(
        "Separate the two questions every stakeholder conflates: is the lift "
        "<i>real</i> (statistical significance, the p-value and CI) and is it "
        "<i>big enough to matter</i> (practical significance, the lift vs. the "
        "business bar). Only ship when the answer to both is yes."
    )


# ==========================================================================
# VIEW 3 -- Campaign Performance
# ==========================================================================
elif view == "Campaign Performance":
    st.title("Campaign Performance")
    dim = st.radio("Group by", ["campaign", "audience"], horizontal=True)

    g = df.groupby(dim, as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
    )
    g["CTR"] = g.apply(lambda x: an.ctr(x["clicks"], x["impressions"]), axis=1)
    g["conv_rate"] = g.apply(lambda x: an.conversion_rate(x["conversions"], x["clicks"]), axis=1)
    g["CPA"] = g.apply(lambda x: an.cpa(x["spend"], x["conversions"]), axis=1)
    g["ROAS"] = g.apply(lambda x: an.roas(x["revenue"], x["spend"]), axis=1)

    g["ROAS_flag"] = an.flag_performance(g["ROAS"].tolist(), higher_is_better=True)
    g["CPA_flag"] = an.flag_performance(g["CPA"].tolist(), higher_is_better=False)

    metric = st.selectbox("Rank by", ["ROAS", "CPA", "CTR", "conv_rate"])
    higher_better = metric != "CPA"
    g_sorted = g.sort_values(metric, ascending=not higher_better)

    flag_col = {"ROAS": "ROAS_flag", "CPA": "CPA_flag"}.get(metric)
    if flag_col:
        color_enc = alt.Color(
            f"{flag_col}:N",
            scale=alt.Scale(domain=["Over", "On-par", "Under"],
                            range=[POSITIVE, GREY, WARN]),
            legend=alt.Legend(title="vs. account avg", orient="top"))
    else:
        color_enc = alt.value(ACCENT)

    axis_fmt = "%" if metric in ("CTR", "conv_rate") else (
        "$,.0f" if metric == "CPA" else ".2f")
    chart = (
        alt.Chart(g_sorted)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", title=metric,
                    axis=alt.Axis(format=axis_fmt, grid=False)),
            y=alt.Y(f"{dim}:N", sort="-x" if higher_better else "x", title=None),
            color=color_enc,
            tooltip=[dim, "spend", "conversions", "ROAS", "CPA"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

    show = g_sorted[[dim, "spend", "conversions", "CTR", "conv_rate", "CPA",
                     "ROAS", "ROAS_flag", "CPA_flag"]].copy()
    show["spend"] = show["spend"].map(fmt_money)
    show["CTR"] = show["CTR"].map(lambda v: fmt_pct(v))
    show["conv_rate"] = show["conv_rate"].map(lambda v: fmt_pct(v))
    show["CPA"] = show["CPA"].map(lambda v: f"${v:,.2f}")
    show["ROAS"] = show["ROAS"].map(lambda v: f"{v:.2f}x")
    st.dataframe(show, width="stretch", hide_index=True)

    over = g[g["ROAS_flag"] == "Over"][dim].tolist()
    under = g[g["ROAS_flag"] == "Under"][dim].tolist()
    over_txt = ", ".join(over) if over else "none stand out"
    under_txt = ", ".join(under) if under else "none stand out"
    so_what(
        f"On ROAS, over-performers (<b>{over_txt}</b>) are earning more per dollar "
        f"than the account average and can absorb more budget; under-performers "
        f"(<b>{under_txt}</b>) need a creative/landing-page test or a budget trim. "
        f"The Spend Efficiency view turns this into a concrete reallocation."
    )


# ==========================================================================
# VIEW 4 -- Spend Efficiency / Budget Reallocation
# ==========================================================================
elif view == "Spend Efficiency":
    st.title("Spend Efficiency & Budget Reallocation")

    move = st.slider("Share of below-average budget to reallocate", 0.0, 0.5,
                     0.20, 0.05)

    g = df.groupby("campaign", as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        conversions=("conversions", "sum"),
    )
    rows = an.recommend_reallocation(
        g["campaign"].tolist(), g["spend"].tolist(),
        g["revenue"].tolist(), g["conversions"].tolist(),
        move_fraction=move,
    )
    rec_df = pd.DataFrame([{
        "campaign": r.name,
        "ROAS": r.roas,
        "current": r.current_spend,
        "recommended": r.recommended_spend,
        "delta": r.delta,
        "action": r.action,
    } for r in rows]).sort_values("delta", ascending=False)

    gain = an.projected_revenue_gain(rows)
    c1, c2 = st.columns(2)
    c1.metric("Budget held constant", fmt_money(rec_df["current"].sum()))
    c2.metric("Projected revenue gain (same-ROAS)", f"+{fmt_money(gain)}")

    st.subheader("Current vs. recommended spend")
    melt = rec_df.melt(id_vars=["campaign", "action"],
                       value_vars=["current", "recommended"],
                       var_name="kind", value_name="dollars")
    chart = (
        alt.Chart(melt)
        .mark_bar()
        .encode(
            y=alt.Y("campaign:N", sort="-x", title=None),
            x=alt.X("dollars:Q", title="Spend ($)",
                    axis=alt.Axis(format="$,.0f", grid=False)),
            yOffset="kind:N",
            color=alt.Color("kind:N",
                            scale=alt.Scale(domain=["current", "recommended"],
                                            range=[GREY, ACCENT]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["campaign", "kind", alt.Tooltip("dollars:Q", format="$,.0f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")

    show = rec_df.copy()
    show["ROAS"] = show["ROAS"].map(lambda v: f"{v:.2f}x")
    show["current"] = show["current"].map(fmt_money)
    show["recommended"] = show["recommended"].map(fmt_money)
    show["delta"] = show["delta"].map(lambda v: f"{'+' if v >= 0 else ''}{v:,.0f}")
    st.dataframe(show, width="stretch", hide_index=True)

    scale_up = rec_df[rec_df["action"] == "Scale up"]["campaign"].tolist()
    cut = rec_df[rec_df["action"] == "Cut back"]["campaign"].tolist()
    so_what(
        f"Holding total budget flat, move money from low-ROAS campaigns "
        f"(<b>{', '.join(cut) if cut else 'none'}</b>) into high-ROAS ones "
        f"(<b>{', '.join(scale_up) if scale_up else 'none'}</b>). At current "
        f"margins this is worth about <b>+{fmt_money(gain)}</b> in revenue for "
        f"zero extra spend. Caveat: the estimate assumes ROAS holds as spend "
        f"scales -- validate with a phased budget test, not a one-shot move."
    )


# ==========================================================================
# VIEW 5 -- About
# ==========================================================================
else:
    st.title("About this project")
    st.markdown(
        """
**SEM & Landing-Page A/B Test Analyzer** -- a portfolio demo built for a
Digital Marketing Analyst context (A/B testing landing pages, ad copy and
audiences; campaign performance and budget decisions).

**What it shows**
- **Overview KPIs** -- impressions, CTR, conversion rate, CPA, ROAS.
- **A/B Test Analyzer** -- two-proportion z-test on conversion with lift,
  p-value and a 95% CI, plus a plain-English call that separates
  *statistical* from *practical* significance.
- **Campaign Performance** -- CTR/CPA/ROAS by campaign or audience, flagging
  over- and under-performers vs. the account average.
- **Spend Efficiency** -- a budget-reallocation recommender that holds total
  spend constant and moves money toward higher-ROAS campaigns.

**How it is built**
- `analytics.py` -- framework-free Python/NumPy math (unit-tested with
  `test_analytics.py` via pytest).
- `data_gen.py` -- synthetic data with a fixed seed and a designed A/B effect.
- `app.py` -- this Streamlit UI; charts follow Knaflic's *Storytelling with
  Data* (muted context, single accent, direct labels, minimal chartjunk).

**Data disclaimer:** every number here is **synthetic**, generated locally for
demonstration. It is not real IXL Learning data and is not drawn from any
proprietary source.

Author: **Leo Chan** -- github.com/GoldenCrate
        """
    )
