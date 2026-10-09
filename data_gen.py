"""
data_gen.py -- SYNTHETIC data generator for the SEM & Landing-Page
A/B Test Analyzer.

Everything produced here is FABRICATED for portfolio/demo purposes. It is
NOT real IXL Learning data and is not derived from any proprietary source.
A fixed random seed makes every run reproducible, and the A/B experiment has
a DESIGNED true effect so the statistical test has something real to find.

Author: Leo Chan -- github.com/GoldenCrate
"""

from __future__ import annotations

import numpy as np

SEED = 42

# The landing-page A/B test has a designed true effect: the new variant ("B")
# genuinely converts better than the control ("A").
TRUE_CONV_RATE_A = 0.0420   # control landing page
TRUE_CONV_RATE_B = 0.0495   # new variant (designed ~18% relative lift)


def generate_campaign_data(seed: int = SEED):
    """Daily SEM campaign performance across several campaigns and audiences.

    Returns a pandas DataFrame with one row per campaign per day.
    Columns: date, campaign, audience, impressions, clicks, conversions,
             spend, revenue.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    campaigns = {
        # name: (base daily impressions, base CTR, base conv rate, avg order value, cpc)
        "Brand - Search":          (18000, 0.085, 0.090, 48.0, 0.85),
        "Math Practice - Search":  (42000, 0.055, 0.052, 42.0, 1.65),
        "Parents - Display":       (95000, 0.006, 0.018, 39.0, 0.95),
        "Teachers - Search":       (26000, 0.048, 0.061, 55.0, 1.40),
        "Competitor - Search":     (15000, 0.038, 0.034, 41.0, 2.80),
        "Back-to-School - Social": (70000, 0.014, 0.029, 37.0, 1.10),
    }
    audiences = {
        "Brand - Search":          "Returning parents",
        "Math Practice - Search":  "In-market parents",
        "Parents - Display":       "Lookalike - parents",
        "Teachers - Search":       "K-12 educators",
        "Competitor - Search":     "Competitor shoppers",
        "Back-to-School - Social": "Seasonal prospects",
    }

    dates = pd.date_range("2025-08-01", periods=60, freq="D")
    rows = []
    for d in dates:
        # Mild weekly seasonality: weekends dip for search intent.
        dow_factor = 0.82 if d.dayofweek >= 5 else 1.0
        for name, (imp_base, ctr_base, conv_base, aov, cpc_base) in campaigns.items():
            impressions = int(rng.normal(imp_base, imp_base * 0.10) * dow_factor)
            impressions = max(impressions, 100)

            ctr_day = max(0.0005, rng.normal(ctr_base, ctr_base * 0.12))
            clicks = int(impressions * ctr_day)

            conv_day = max(0.0, rng.normal(conv_base, conv_base * 0.15))
            conversions = int(clicks * conv_day)

            cpc_day = max(0.10, rng.normal(cpc_base, cpc_base * 0.10))
            spend = round(clicks * cpc_day, 2)

            order_value = max(5.0, rng.normal(aov, aov * 0.08))
            revenue = round(conversions * order_value, 2)

            rows.append({
                "date": d,
                "campaign": name,
                "audience": audiences[name],
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": spend,
                "revenue": revenue,
            })

    return pd.DataFrame(rows)


def generate_ab_test_data(seed: int = SEED):
    """Landing-page / ad-copy A/B experiment with a designed true effect.

    Simulates visitors randomized to control (A) or variant (B) and whether
    each converted, using the module-level TRUE_CONV_RATE_* probabilities.

    Returns:
        summary  -- dict with per-variant visitors and conversions
        visitors -- pandas DataFrame, one row per visitor (variant, converted)
    """
    import pandas as pd

    rng = np.random.default_rng(seed + 1)

    n_a = 12_000
    n_b = 12_000

    conv_a_flags = rng.random(n_a) < TRUE_CONV_RATE_A
    conv_b_flags = rng.random(n_b) < TRUE_CONV_RATE_B

    visitors = pd.DataFrame({
        "variant": (["A"] * n_a) + (["B"] * n_b),
        "converted": np.concatenate([conv_a_flags, conv_b_flags]).astype(int),
    })

    summary = {
        "variant_a_name": "Control: current landing page",
        "variant_b_name": "Variant B: benefit-led headline + shorter form",
        "n_a": n_a,
        "n_b": n_b,
        "conv_a": int(conv_a_flags.sum()),
        "conv_b": int(conv_b_flags.sum()),
    }
    return summary, visitors


if __name__ == "__main__":
    df = generate_campaign_data()
    summary, visitors = generate_ab_test_data()
    print("SYNTHETIC DATA -- generated for portfolio/demo use only.\n")
    print(f"Campaign rows: {len(df)} across {df['campaign'].nunique()} campaigns, "
          f"{df['date'].nunique()} days")
    print(f"Total spend:   ${df['spend'].sum():,.0f}")
    print(f"Total revenue: ${df['revenue'].sum():,.0f}")
    print(f"Portfolio ROAS: {df['revenue'].sum() / df['spend'].sum():.2f}x\n")
    print("A/B experiment:")
    print(f"  A: {summary['conv_a']:,}/{summary['n_a']:,} = "
          f"{summary['conv_a'] / summary['n_a'] * 100:.2f}%")
    print(f"  B: {summary['conv_b']:,}/{summary['n_b']:,} = "
          f"{summary['conv_b'] / summary['n_b'] * 100:.2f}%")
