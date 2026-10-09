# SEM & Landing-Page A/B Test Analyzer

A working Streamlit app for a **Digital Marketing Analyst** workflow: paid-search
(SEM) performance, a landing-page / ad-copy **A/B test analyzer**, campaign
scanning, and a **budget-reallocation** recommender.

Built as a portfolio project by **Leo Chan — [github.com/GoldenCrate](https://github.com/GoldenCrate)**.

> **Data disclaimer:** every number in this app is **SYNTHETIC**, generated
> locally by `data_gen.py` with a fixed random seed. It is **not** real IXL
> Learning data and is not drawn from any proprietary or production source.

---

## What it does

Five views, each ending in a plain-English **"So what → recommendation"**:

1. **Overview KPIs** — impressions, CTR, conversion rate, CPA, ROAS, plus a
   daily conversions trend.
2. **A/B Test Analyzer** — a **two-proportion z-test** on conversion rate for a
   landing-page/ad-copy variant test, reporting absolute & relative **lift**, a
   **p-value**, a **95% confidence interval** on the lift, and a winner call
   that explicitly separates **statistical** significance from **practical**
   significance (an adjustable business-impact bar).
3. **Campaign Performance** — CTR / CPA / ROAS by campaign or audience, with
   automatic **over- / under-performer flags** vs. the account average.
4. **Spend Efficiency** — a **budget-reallocation** recommender that holds total
   spend constant and shifts money from low-ROAS to high-ROAS campaigns, with a
   same-ROAS projected revenue gain (and the caveat that this assumes ROAS holds
   as spend scales).
5. **About** — how the project is built.

Charts follow Cole Nussbaumer Knaflic's *Storytelling with Data*: a muted grey
baseline for context, a single accent color for the point being made, direct
value labels instead of busy legends, and minimal gridline/axis clutter.

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Multi-view Streamlit UI |
| `analytics.py` | Framework-free KPI math + two-proportion z-test + flagging + reallocation (no Streamlit, no scipy) |
| `data_gen.py` | Synthetic SEM + A/B experiment data (fixed seed, designed effect) |
| `test_analytics.py` | pytest suite for the analytics module |
| `requirements.txt` | Dependencies |

The analytics layer is deliberately framework-free so the statistics are unit-
tested independently of the UI. The z-test uses `math.erf` for the normal CDF
and a rational approximation for the inverse CDF, so there is **no scipy
dependency**.

## Run locally

```bash
pip install -r requirements.txt

# (optional) preview the synthetic data
python data_gen.py

# run the tests
pytest -q

# launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## The A/B method, briefly

- **H0:** variant B's conversion rate equals control A's.
- Test statistic uses the **pooled** proportion standard error; the **95% CI on
  the absolute lift** uses the **unpooled** SE (standard practice).
- A result is only a "ship it" when it is **both** statistically significant
  (p < α and CI excludes zero) **and** clears the practical-impact bar. The app
  calls out the common failure mode of a tiny-but-significant lift at large
  sample sizes, and of a large-but-underpowered lift that is not yet significant.

---

Author: **Leo Chan** — github.com/GoldenCrate · Synthetic data only.
