import numpy as np

def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """Ackley function. Returns value >= 0. Global minimum at x=0 -> f(x)=0.

    x: array-like
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    n = x.size
    sum_sq = np.sum(x ** 2)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return term1 + term2 + a + np.e


if __name__ == '__main__':
    # quick sanity check
    print('Ackley(0..) ->', ackley(np.zeros(3)))