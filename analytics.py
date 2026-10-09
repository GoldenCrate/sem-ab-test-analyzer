"""
analytics.py -- Framework-free marketing analytics for the
SEM & Landing-Page A/B Test Analyzer.

Pure Python + NumPy only (no pandas, no scipy, no Streamlit here) so the
math is independently testable with pytest. Streamlit and data generation
import these functions; they do not live here.

Author: Leo Chan -- github.com/GoldenCrate
Data note: this project operates on SYNTHETIC data only (see data_gen.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Core KPI helpers
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is 0 (common with no spend/clicks)."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def ctr(clicks: float, impressions: float) -> float:
    """Click-through rate = clicks / impressions (0-1)."""
    return safe_div(clicks, impressions)


def conversion_rate(conversions: float, clicks: float) -> float:
    """Conversion rate = conversions / clicks (0-1)."""
    return safe_div(conversions, clicks)


def cpa(spend: float, conversions: float) -> float:
    """Cost per acquisition = spend / conversions (dollars)."""
    return safe_div(spend, conversions)


def cpc(spend: float, clicks: float) -> float:
    """Cost per click = spend / clicks (dollars)."""
    return safe_div(spend, clicks)


def roas(revenue: float, spend: float) -> float:
    """Return on ad spend = revenue / spend (ratio; 4.0 means $4 per $1)."""
    return safe_div(revenue, spend)


# ---------------------------------------------------------------------------
# A/B test: two-proportion z-test on conversion rate
# ---------------------------------------------------------------------------

def _norm_cdf(z: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_sf(z: float) -> float:
    """Standard normal survival function (1 - CDF)."""
    return 1.0 - _norm_cdf(z)


@dataclass
class ABResult:
    """Result of a two-proportion z-test comparing variant B against control A."""
    conv_a: int
    n_a: int
    conv_b: int
    n_b: int
    rate_a: float
    rate_b: float
    abs_lift: float          # rate_b - rate_a (percentage points, as a proportion)
    rel_lift: float          # (rate_b - rate_a) / rate_a
    z_score: float
    p_value: float           # two-sided
    ci_low: float            # 95% CI lower bound on absolute lift
    ci_high: float           # 95% CI upper bound on absolute lift
    significant: bool        # p < alpha
    winner: str              # "B", "A", or "No clear winner"

    @property
    def rel_lift_pct(self) -> float:
        return self.rel_lift * 100.0


def two_proportion_ztest(
    conv_a: int,
    n_a: int,
    conv_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> ABResult:
    """Two-proportion z-test on conversion.

    A = control, B = variant. Tests H0: rate_b == rate_a (two-sided).
    Returns an ABResult with lift, z-score, two-sided p-value, and a
    (1 - alpha) confidence interval on the ABSOLUTE lift (rate_b - rate_a).

    The pooled proportion is used for the test statistic; the unpooled
    standard error is used for the confidence interval (standard practice).
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")
    if not (0 <= conv_a <= n_a) or not (0 <= conv_b <= n_b):
        raise ValueError("Conversions must be between 0 and the sample size.")

    rate_a = conv_a / n_a
    rate_b = conv_b / n_b
    abs_lift = rate_b - rate_a
    rel_lift = safe_div(abs_lift, rate_a)

    # Pooled proportion for the test statistic.
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))

    if se_pooled == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = abs_lift / se_pooled
        p_value = 2.0 * _norm_sf(abs(z))

    # Unpooled SE for the confidence interval on the absolute lift.
    se_unpooled = math.sqrt(
        rate_a * (1.0 - rate_a) / n_a + rate_b * (1.0 - rate_b) / n_b
    )
    z_crit = _z_critical(alpha)
    ci_low = abs_lift - z_crit * se_unpooled
    ci_high = abs_lift + z_crit * se_unpooled

    significant = p_value < alpha
    if not significant:
        winner = "No clear winner"
    elif abs_lift > 0:
        winner = "B"
    else:
        winner = "A"

    return ABResult(
        conv_a=conv_a,
        n_a=n_a,
        conv_b=conv_b,
        n_b=n_b,
        rate_a=rate_a,
        rate_b=rate_b,
        abs_lift=abs_lift,
        rel_lift=rel_lift,
        z_score=z,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        winner=winner,
    )


def _z_critical(alpha: float) -> float:
    """Two-sided critical z value for common alphas (avoids a scipy dependency)."""
    table = {
        0.10: 1.6448536269514722,
        0.05: 1.959963984540054,
        0.01: 2.5758293035489004,
    }
    if alpha in table:
        return table[alpha]
    # Fallback: invert the normal CDF numerically for arbitrary alpha.
    return _inv_norm_cdf(1.0 - alpha / 2.0)


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def interpret_ab(result: ABResult, practical_threshold_pct: float = 5.0) -> str:
    """Plain-English winner call separating statistical and practical significance.

    practical_threshold_pct is the minimum RELATIVE lift (in %) that the
    business considers worth acting on, regardless of statistical significance.
    """
    rel = result.rel_lift_pct
    direction = "higher" if result.abs_lift > 0 else "lower"
    stat_sig = result.significant
    practically_meaningful = abs(rel) >= practical_threshold_pct

    header = (
        f"Variant B converted at {result.rate_b * 100:.2f}% vs. control A at "
        f"{result.rate_a * 100:.2f}% -- a {abs(rel):.1f}% relative lift "
        f"({direction} for B)."
    )

    if stat_sig and practically_meaningful:
        verdict = (
            f"Statistically significant (p = {result.p_value:.4f}) AND above the "
            f"{practical_threshold_pct:.0f}% practical-impact bar. "
            f"Recommendation: ship Variant B."
        )
    elif stat_sig and not practically_meaningful:
        verdict = (
            f"Statistically significant (p = {result.p_value:.4f}) but the lift is "
            f"below the {practical_threshold_pct:.0f}% practical bar. The effect is "
            f"real but small -- ship only if the change is cheap, otherwise keep testing."
        )
    elif (not stat_sig) and practically_meaningful:
        verdict = (
            f"The observed lift looks large but is NOT statistically significant "
            f"(p = {result.p_value:.4f}); the 95% CI on absolute lift "
            f"[{result.ci_low * 100:.2f}, {result.ci_high * 100:.2f}] pp still "
            f"includes zero. Likely under-powered -- collect more data before calling it."
        )
    else:
        verdict = (
            f"Not statistically significant (p = {result.p_value:.4f}) and not "
            f"practically meaningful. Treat A and B as a tie -- keep the simpler/"
            f"cheaper variant and move on."
        )
    return f"{header}\n\n{verdict}"


# ---------------------------------------------------------------------------
# Campaign performance: flag under / over performers
# ---------------------------------------------------------------------------

def flag_performance(values: list[float], higher_is_better: bool,
                     z_threshold: float = 1.0) -> list[str]:
    """Flag each value relative to the group mean using a z-score band.

    Returns a parallel list of "Over", "Under", or "On-par" labels.
    For metrics where lower is better (e.g. CPA), the direction is inverted.
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    if n == 1:
        return ["On-par"]
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)

    labels: list[str] = []
    for v in values:
        if std == 0:
            labels.append("On-par")
            continue
        z = (v - mean) / std
        if not higher_is_better:
            z = -z  # lower raw value -> better
        if z >= z_threshold:
            labels.append("Over")
        elif z <= -z_threshold:
            labels.append("Under")
        else:
            labels.append("On-par")
    return labels


# ---------------------------------------------------------------------------
# Spend efficiency / budget reallocation
# ---------------------------------------------------------------------------

@dataclass
class ReallocationRow:
    name: str
    current_spend: float
    roas: float
    cpa: float
    recommended_spend: float
    delta: float             # recommended - current
    action: str              # "Scale up", "Cut back", "Hold"


def recommend_reallocation(
    names: list[str],
    spend: list[float],
    revenue: list[float],
    conversions: list[float],
    move_fraction: float = 0.20,
) -> list[ReallocationRow]:
    """Budget-reallocation proposal that holds total spend constant.

    Logic: compute ROAS per channel. Take `move_fraction` of the budget away
    from below-average-ROAS channels and redistribute it to above-average-ROAS
    channels in proportion to how far above average they are. This mimics the
    real decision a marketing analyst makes: move money toward what pays back.
    """
    n = len(names)
    if not (n == len(spend) == len(revenue) == len(conversions)):
        raise ValueError("All input lists must be the same length.")
    if n == 0:
        return []

    roas_vals = [roas(revenue[i], spend[i]) for i in range(n)]
    total_spend = sum(spend)
    avg_roas = safe_div(sum(revenue), total_spend)  # spend-weighted portfolio ROAS

    # Pool to move = move_fraction of spend currently on below-average channels.
    pool = 0.0
    freed = [0.0] * n
    for i in range(n):
        if roas_vals[i] < avg_roas:
            freed[i] = spend[i] * move_fraction
            pool += freed[i]

    # Distribute the pool among above-average channels, weighted by excess ROAS.
    excess = [max(0.0, roas_vals[i] - avg_roas) for i in range(n)]
    total_excess = sum(excess)

    recommended = list(spend)
    for i in range(n):
        recommended[i] -= freed[i]
    if total_excess > 0 and pool > 0:
        for i in range(n):
            recommended[i] += pool * (excess[i] / total_excess)

    rows: list[ReallocationRow] = []
    for i in range(n):
        delta = recommended[i] - spend[i]
        if delta > 1e-6:
            action = "Scale up"
        elif delta < -1e-6:
            action = "Cut back"
        else:
            action = "Hold"
        rows.append(ReallocationRow(
            name=names[i],
            current_spend=spend[i],
            roas=roas_vals[i],
            cpa=cpa(spend[i], conversions[i]),
            recommended_spend=recommended[i],
            delta=delta,
            action=action,
        ))
    return rows


def projected_revenue_gain(rows: list[ReallocationRow]) -> float:
    """Estimate incremental revenue if each channel keeps its current ROAS.

    Projected gain = sum(delta_spend_i * roas_i). This is a first-order
    estimate that assumes ROAS holds at the margin (a documented caveat).
    """
    return sum(r.delta * r.roas for r in rows)
