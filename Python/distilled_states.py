"""
Distilled states using BBPSSW. The SDP here fixes rho to the Bell-diagonal state.
"""
import numpy as np
import matplotlib.pyplot as plt

from e91_common import (binary_entropy, von_neumann, bell_diagonal_state,
                        hae_e91_from_state)


def distilled_state(F0):
    F = (F0**2 + (1 / 9) * (1 - F0)**2) / (F0**2 + (2 / 3) * F0 * (1 - F0) + (5 / 9) * (1 - F0)**2)
    gamma = F
    sq_root = np.sqrt(max(-4 * F**2 + 6 * F - 1, 0.0))
    delta = (2 - F - sq_root) / 5
    epsilon = delta
    tau = (1 - 3 * F + 2 * sq_root) / 5
    return gamma, delta, epsilon, tau


def hab_e91(delta, epsilon):
    return binary_entropy(delta + epsilon)


def e91_qkd(F, m):
    gamma, delta, epsilon, tau = distilled_state(F)
    K_analytical = 1.0 - von_neumann(gamma, delta, epsilon, tau)

    # lambda00 = gamma, lambda01 = delta, lambda10 = epsilon, lambda11 = tau
    rho = bell_diagonal_state(gamma, delta, epsilon, tau)
    hae, _ = hae_e91_from_state(rho, m, fix_rho=True)
    K_SDP = hae - hab_e91(delta, epsilon)
    return K_SDP, K_analytical, gamma


if __name__ == "__main__":
    fidelity = np.linspace(0.25, 1, 30)
    m = 30

    key_sdp, key_analytical, F_out = [], [], []
    for F in fidelity:
        k_sdp, k_an, g = e91_qkd(F, m)
        key_sdp.append(k_sdp)
        key_analytical.append(k_an)
        F_out.append(g)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(F_out, key_analytical, '-xb', label='Key Rate - Analytical')
    ax.plot(F_out, key_sdp, '-or', label='Key Rate - SDP')
    ax.set_xlabel('Fidelity $F_{out}$')
    ax.set_ylabel('Key rate')
    ax.set_title('Distilled State')
    ax.grid(True)
    ax.legend(loc='upper left')
    # fig.savefig("distilled.pdf", bbox_inches="tight")
    plt.show()
