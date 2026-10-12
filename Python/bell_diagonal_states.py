"""
Bell-diagonal state parametrised by the QBERs (Qx, Qy, Qz).
Here Qx = Qz = 0.1 and Qy is swept. 
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from e91_common import (binary_entropy, von_neumann, hae_sdp,
                        PHI_PLUS, PHI_MINUS, PSI_PLUS, PSI_MINUS)


def bell_diagonal_params(QBER):
    Qx, Qy, Qz = QBER
    lambda00 = 1 - (Qx + Qy + Qz) / 2
    lambda01 = (Qx + Qy - Qz) / 2
    lambda10 = (-Qx + Qy + Qz) / 2
    lambda11 = (Qx - Qy + Qz) / 2
    return lambda00, lambda01, lambda10, lambda11


def bell_diagonal_state(l00, l01, l10, l11):
    return (l00 * np.outer(PHI_PLUS, PHI_PLUS)
            + l01 * np.outer(PHI_MINUS, PHI_MINUS)
            + l10 * np.outer(PSI_PLUS, PSI_PLUS)
            + l11 * np.outer(PSI_MINUS, PSI_MINUS))


def hab_e91(lambda10, lambda11):
    return binary_entropy(lambda10 + lambda11)


def e91_basis():
    x = np.array([[0, 1], [1, 0]])
    return np.kron(x, x)


def hae_e91(l00, l01, l10, l11, m):
    base = e91_basis()
    f_w = base @ bell_diagonal_state(l00, l01, l10, l11)  

    def make_constraints(rho):
        return [cp.real(base @ rho) == f_w]

    value, _ = hae_sdp(m, make_constraints)
    return value


def e91_qkd(QBER, m):
    l00, l01, l10, l11 = bell_diagonal_params(QBER)
    K_analytical = 1.0 - von_neumann(l00, l01, l10, l11)
    K_SDP = hae_e91(l00, l01, l10, l11, m) - hab_e91(l10, l11)
    return K_SDP, K_analytical


if __name__ == "__main__":
    Qy = np.linspace(0.0, 0.2, 10)
    m = 10

    key_sdp, key_analytical = [], []
    for q in Qy:
        k_sdp, k_an = e91_qkd([0.1, q, 0.1], m)
        key_sdp.append(k_sdp)
        key_analytical.append(k_an)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Qy, key_analytical, '-xb', label='Key Rate - Analytical')
    ax.plot(Qy, key_sdp, '-or', label='Key Rate - SDP')
    ax.set_xlabel('Qy')
    ax.set_ylabel('Key rate')
    ax.set_title('Bell Diagonal State Qx = Qz = 0.1')
    ax.grid(True)
    ax.legend(loc='upper right')
    # fig.savefig("belldiagonal_Qy.pdf", bbox_inches="tight")
    plt.show()
