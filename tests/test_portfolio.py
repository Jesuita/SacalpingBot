"""Tests para el modulo de correlacion de portfolio."""
import pytest
from portfolio import calc_returns, pearson_correlation, get_high_correlations


class TestCalcReturns:
    def test_basic(self):
        r = calc_returns([100, 110, 105])
        assert len(r) == 2
        assert abs(r[0] - 0.1) < 0.001

    def test_empty(self):
        assert calc_returns([]) == []
        assert calc_returns([100]) == []


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [2, 4, 6, 8, 10, 12, 14]
        assert abs(pearson_correlation(x, y) - 1.0) < 0.001

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [14, 12, 10, 8, 6, 4, 2]
        assert abs(pearson_correlation(x, y) - (-1.0)) < 0.001

    def test_no_correlation(self):
        x = [1, -1, 1, -1, 1, -1, 1]
        y = [1, 1, -1, -1, 1, 1, -1]
        assert abs(pearson_correlation(x, y)) < 0.5

    def test_insufficient_data(self):
        assert pearson_correlation([1, 2], [3, 4]) == 0.0

    def test_constant_series(self):
        assert pearson_correlation([5, 5, 5, 5, 5], [1, 2, 3, 4, 5]) == 0.0


class TestHighCorrelations:
    def test_finds_correlated_pairs(self):
        result = {
            "symbols": ["A", "B", "C"],
            "matrix": [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ],
        }
        highs = get_high_correlations(result, threshold=0.7)
        assert len(highs) == 1
        assert highs[0]["pair_a"] == "A"
        assert highs[0]["pair_b"] == "B"

    def test_no_high_correlations(self):
        result = {
            "symbols": ["A", "B"],
            "matrix": [[1.0, 0.1], [0.1, 1.0]],
        }
        assert len(get_high_correlations(result)) == 0
