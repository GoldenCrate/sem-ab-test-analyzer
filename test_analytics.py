"""
test_analytics.py -- pytest suite for analytics.py.

Covers KPI helpers, the two-proportion z-test (statistic, p-value, CI,
winner call), performance flagging, and budget reallocation. Run with:

    pytest -q

Author: Leo Chan -- github.com/GoldenCrate
"""

import math

import pytest

import analytics as a


# --------------------------- KPI helpers -----------------------------------

def test_safe_div_zero_denominator():
    assert a.safe_div(5, 0) == 0.0


def test_ctr_and_conversion_rate():
    assert a.ctr(50, 1000) == pytest.approx(0.05)
    assert a.conversion_rate(10, 200) == pytest.approx(0.05)


def test_cpa_cpc_roas():
    assert a.cpa(1000, 25) == pytest.approx(40.0)
    assert a.cpc(500, 1000) == pytest.approx(0.5)
    assert a.roas(4000, 1000) == pytest.approx(4.0)


def test_roas_no_spend_is_zero():
    assert a.roas(100, 0) == 0.0


# --------------------------- z-test math -----------------------------------

def test_ztest_detects_known_difference():
    # 500/10000 = 5% vs 650/10000 = 6.5%: clearly significant.
    r = a.two_proportion_ztest(500, 10000, 650, 10000)
    assert r.rate_a == pytest.approx(0.05)
    assert r.rate_b == pytest.approx(0.065)
    assert r.abs_lift == pytest.approx(0.015)
    assert r.rel_lift == pytest.approx(0.30)
    assert r.p_value < 0.001
    assert r.significant is True
    assert r.winner == "B"


def test_ztest_no_difference_not_significant():
    r = a.two_proportion_ztest(500, 10000, 505, 10000)
    assert r.significant is False
    assert r.winner == "No clear winner"
    assert r.p_value > 0.05


def test_ztest_identical_inputs_pvalue_one():
    r = a.two_proportion_ztest(500, 10000, 500, 10000)
    assert r.z_score == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert r.winner == "No clear winner"


def test_ztest_pvalue_matches_known_z():
    # Hand-check: z for 200/5000 vs 260/5000.
    r = a.two_proportion_ztest(200, 5000, 260, 5000)
    # Recompute expected two-sided p from the returned z.
    expected_p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(r.z_score) / math.sqrt(2))))
    assert r.p_value == pytest.approx(expected_p, rel=1e-9)
    assert r.significant is True


def test_ci_contains_point_estimate_and_brackets_zero_when_tie():
    r = a.two_proportion_ztest(500, 10000, 505, 10000)
    assert r.ci_low < r.abs_lift < r.ci_high
    # A non-significant result's 95% CI on lift should include 0.
    assert r.ci_low < 0 < r.ci_high


def test_ci_excludes_zero_when_significant():
    r = a.two_proportion_ztest(500, 10000, 700, 10000)
    assert r.ci_low > 0


def test_ztest_winner_a_when_b_worse():
    r = a.two_proportion_ztest(700, 10000, 500, 10000)
    assert r.winner == "A"
    assert r.abs_lift < 0


def test_ztest_rejects_bad_inputs():
    with pytest.raises(ValueError):
        a.two_proportion_ztest(10, 0, 5, 100)
    with pytest.raises(ValueError):
        a.two_proportion_ztest(200, 100, 5, 100)  # conv > n


def test_z_critical_values():
    assert a._z_critical(0.05) == pytest.approx(1.96, abs=0.01)
    assert a._z_critical(0.01) == pytest.approx(2.576, abs=0.01)
    # Arbitrary alpha falls through to the numerical inverse-CDF.
    assert a._z_critical(0.20) == pytest.approx(1.2816, abs=0.01)


# --------------------------- interpretation --------------------------------

def test_interpret_significant_and_practical():
    r = a.two_proportion_ztest(500, 10000, 650, 10000)  # +30% rel
    text = a.interpret_ab(r, practical_threshold_pct=5.0)
    assert "ship Variant B" in text


def test_interpret_significant_but_not_practical():
    # Large n makes a tiny lift significant but below a 5% practical bar.
    r = a.two_proportion_ztest(50000, 1000000, 50900, 1000000)
    assert r.significant is True
    assert abs(r.rel_lift_pct) < 5.0
    text = a.interpret_ab(r, practical_threshold_pct=5.0)
    assert "below" in text.lower()


def test_interpret_not_significant():
    r = a.two_proportion_ztest(500, 10000, 505, 10000)
    text = a.interpret_ab(r)
    assert "tie" in text.lower() or "not" in text.lower()


# --------------------------- flagging --------------------------------------

def test_flag_performance_higher_is_better():
    vals = [0.01, 0.02, 0.02, 0.20]  # last one clearly over
    labels = a.flag_performance(vals, higher_is_better=True)
    assert labels[-1] == "Over"


def test_flag_performance_lower_is_better_inverts():
    # For CPA, the LOW value should be flagged "Over" (over-performing/good).
    cpa_vals = [10.0, 50.0, 55.0, 60.0]
    labels = a.flag_performance(cpa_vals, higher_is_better=False)
    assert labels[0] == "Over"


def test_flag_performance_edge_cases():
    assert a.flag_performance([], higher_is_better=True) == []
    assert a.flag_performance([5.0], higher_is_better=True) == ["On-par"]
    # All identical -> no std -> all on-par.
    assert a.flag_performance([3.0, 3.0, 3.0], higher_is_better=True) == \
        ["On-par", "On-par", "On-par"]


# --------------------------- reallocation ----------------------------------

def test_reallocation_holds_total_spend_constant():
    names = ["Low ROAS", "High ROAS", "Mid ROAS"]
    spend = [1000.0, 1000.0, 1000.0]
    revenue = [1500.0, 5000.0, 3000.0]      # ROAS 1.5x, 5x, 3x
    conversions = [30.0, 100.0, 60.0]
    rows = a.recommend_reallocation(names, spend, revenue, conversions)
    total_before = sum(spend)
    total_after = sum(r.recommended_spend for r in rows)
    assert total_after == pytest.approx(total_before, abs=1e-6)


def test_reallocation_moves_money_toward_high_roas():
    names = ["Low ROAS", "High ROAS"]
    spend = [1000.0, 1000.0]
    revenue = [1200.0, 6000.0]
    conversions = [30.0, 100.0]
    rows = a.recommend_reallocation(names, spend, revenue, conversions)
    by_name = {r.name: r for r in rows}
    assert by_name["Low ROAS"].action == "Cut back"
    assert by_name["High ROAS"].action == "Scale up"
    assert by_name["High ROAS"].delta > 0


def test_reallocation_length_mismatch_raises():
    with pytest.raises(ValueError):
        a.recommend_reallocation(["a"], [1.0, 2.0], [1.0], [1.0])


def test_projected_revenue_gain_positive_when_reallocating_to_high_roas():
    names = ["Low ROAS", "High ROAS"]
    spend = [1000.0, 1000.0]
    revenue = [1200.0, 6000.0]
    conversions = [30.0, 100.0]
    rows = a.recommend_reallocation(names, spend, revenue, conversions)
    assert a.projected_revenue_gain(rows) > 0


# --------------------------- data generator --------------------------------

def test_generated_ab_data_reproducible_and_has_effect():
    pd = pytest.importorskip("pandas")
    import data_gen
    s1, _ = data_gen.generate_ab_test_data()
    s2, _ = data_gen.generate_ab_test_data()
    assert s1 == s2  # fixed seed -> reproducible
    rate_a = s1["conv_a"] / s1["n_a"]
    rate_b = s1["conv_b"] / s1["n_b"]
    assert rate_b > rate_a  # designed effect is present


def test_generated_campaign_data_shape():
    pytest.importorskip("pandas")
    import data_gen
    df = data_gen.generate_campaign_data()
    assert len(df) > 0
    assert set(["impressions", "clicks", "conversions", "spend", "revenue"]).issubset(df.columns)
    assert (df["impressions"] >= df["clicks"]).all()
