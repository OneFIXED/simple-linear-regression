"""
Simple Linear Regression Analysis (One-Factor Model)
=====================================================

Implementation of a one-factor linear econometric model:
    Y = b0 + b1 * X

Includes:
- Coefficient estimation by Ordinary Least Squares (OLS)
- Correlation and determination coefficients (r, R^2)
- Fisher's F-test for model adequacy
- Student's t-test for coefficient significance
- Elasticity coefficient analysis
- Point forecast for a new X value

Author: Vanin Dmytro
Course: Digital Information-Analytical Systems
University: V. N. Karazin Kharkiv National University
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["X"], dtype=float), np.array(data["Y"], dtype=float)


def estimate_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Estimate b0 (intercept) and b1 (slope) by OLS:
        b1 = Sxy / Sxx
        b0 = mean(Y) - b1 * mean(X)
    """
    x_mean, y_mean = x.mean(), y.mean()
    sxy = np.sum((x - x_mean) * (y - y_mean))
    sxx = np.sum((x - x_mean) ** 2)
    b1 = sxy / sxx
    b0 = y_mean - b1 * x_mean
    return b0, b1


def correlation_and_determination(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r = np.corrcoef(x, y)[0, 1]
    return r, r ** 2


def fisher_test(y: np.ndarray, y_hat: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Fisher's F-test for overall model adequacy.
        F = (SSR / df_reg) / (SSE / df_err)
    """
    n = len(y)
    y_mean = y.mean()
    ssr = np.sum((y_hat - y_mean) ** 2)
    sse = np.sum((y - y_hat) ** 2)

    df_reg = 1
    df_err = n - 2        

    f_fact = (ssr / df_reg) / (sse / df_err)
    f_crit = stats.f.ppf(1 - alpha, df_reg, df_err)

    return {
        "F_fact": f_fact,
        "F_crit": f_crit,
        "is_adequate": f_fact > f_crit,
        "SSR": ssr,
        "SSE": sse,
    }


def student_test(
    b0: float, b1: float,
    x: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Student's t-test for individual coefficient significance.
        t(b) = b / SE(b)
    """
    n = len(y)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    sse = np.sum((y - y_hat) ** 2)

    sigma2 = sse / (n - 2)
    se_b1 = np.sqrt(sigma2 / sxx)
    se_b0 = np.sqrt(sigma2 * (1 / n + x_mean ** 2 / sxx))

    t_b0 = b0 / se_b0
    t_b1 = b1 / se_b1
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)

    return {
        "t_b0": t_b0,
        "t_b1": t_b1,
        "t_crit": t_crit,
        "b0_significant": abs(t_b0) > t_crit,
        "b1_significant": abs(t_b1) > t_crit,
    }


def elasticity(b1: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Point elasticity coefficient for each observation:
        E_i = b1 * X_i / Y_i
    Shows percentage change in Y per 1% change in X.
    """
    return b1 * x / y


def forecast(b0: float, b1: float, x_new: float) -> float:
    return b0 + b1 * x_new


def print_report(
    b0: float, b1: float,
    r: float, r2: float,
    fisher: dict, student: dict,
    elast: np.ndarray,
    x_pr: float, y_pr: float,
    x: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
):
    print("=" * 60)
    print("SIMPLE LINEAR REGRESSION — RESULTS")
    print("=" * 60)

    print("\n1. REGRESSION EQUATION")
    print("-" * 60)
    print(f"   Y = {b0:.3f} + {b1:.3f} * X")

    print("\n2. CORRELATION & DETERMINATION")
    print("-" * 60)
    print(f"   Pearson correlation     r   = {r:.5f}")
    print(f"   Coefficient of determ.  R^2 = {r2:.5f}")
    print(f"   → Model explains ~{r2 * 100:.1f}% of Y's variation")

    print("\n3. FISHER'S F-TEST (model adequacy)")
    print("-" * 60)
    print(f"   F_fact = {fisher['F_fact']:.2f}")
    print(f"   F_crit = {fisher['F_crit']:.2f}")
    verdict = "ADEQUATE" if fisher["is_adequate"] else "NOT ADEQUATE"
    print(f"   → Model is {verdict} at alpha = 0.05")

    print("\n4. STUDENT'S T-TEST (coefficient significance)")
    print("-" * 60)
    print(f"   t(b0)  = {student['t_b0']:.2f}    "
          f"({'significant' if student['b0_significant'] else 'NOT significant'})")
    print(f"   t(b1)  = {student['t_b1']:.2f}    "
          f"({'significant' if student['b1_significant'] else 'NOT significant'})")
    print(f"   t_crit = {student['t_crit']:.2f}")

    print("\n5. ELASTICITY")
    print("-" * 60)
    print(f"   Mean elasticity = {elast.mean():.4f}")
    print(f"   Range           = [{elast.min():.4f} … {elast.max():.4f}]")
    print(f"   → A 1% increase in X changes Y by ~{elast.mean():.2f}% on average")

    print(f"\n6. POINT FORECAST")
    print("-" * 60)
    print(f"   At X = {x_pr} → predicted Y = {y_pr:.2f}")

    print("\n7. DATA TABLE")
    print("-" * 60)
    print(f"   {'X':>6} {'Y':>6} {'Y_hat':>8} {'Elasticity':>12}")
    for xi, yi, yhi, ei in zip(x, y, y_hat, elast):
        print(f"   {xi:>6.1f} {yi:>6.1f} {yhi:>8.4f} {ei:>12.4f}")
    print()


def main():
    data_path = Path(__file__).parent / "data.json"
    x, y = load_data(str(data_path))

    b0, b1 = estimate_coefficients(x, y)
    y_hat = b0 + b1 * x

    r, r2 = correlation_and_determination(x, y)

    fisher = fisher_test(y, y_hat)
    student = student_test(b0, b1, x, y, y_hat)

    elast = elasticity(b1, x, y)

    x_pr = 8.0
    y_pr = forecast(b0, b1, x_pr)

    print_report(b0, b1, r, r2, fisher, student, elast, x_pr, y_pr, x, y, y_hat)


if __name__ == "__main__":
    main()