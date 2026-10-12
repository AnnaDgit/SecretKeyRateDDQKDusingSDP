"""
Comparison of Werner states, distilled (BBPSSW) states and bit-flip states
using the E91 protocol and the six-state analytical rate.
"""
import numpy as np
import matplotlib.pyplot as plt

from e91_common import (binary_entropy, von_neumann, werner_state,
                        bell_diagonal_state, bitflip_state, hae_e91_from_state)


# ---------------------------- Werner ---------------------------------
def werner_params(F):
    gamma = F
    delta = (1.0 - F) / 3.0
    eps   = (1.0 - F) / 3.0
    tau   = (1.0 - F) / 3.0
    return gamma, delta, eps, tau


def hab_e91(F):
    v = (4 * F - 1) / 3
    Q = (1 - v) / 2
    return binary_entropy(Q)


def e91_qkd(F, m):
    v = (4 * F - 1) / 3
    gamma, delta, eps, tau = werner_params(F)
    K_analytical = 1.0 - von_neumann(gamma, delta, eps, tau)

    hae, _ = hae_e91_from_state(werner_state(v), m)
    K_SDP = hae - hab_e91(F)
    return K_SDP, K_analytical


# --------------------------- Distilled -------------------------------
def distilled_state(F0):
    F = (F0**2 + (1 / 9) * (1 - F0)**2) / (F0**2 + (2 / 3) * F0 * (1 - F0) + (5 / 9) * (1 - F0)**2)
    gamma = F
    sq_root = np.sqrt(max(-4 * F**2 + 6 * F - 1, 0.0))
    delta = (2 - F - sq_root) / 5
    epsilon = delta
    tau = (1 - 3 * F + 2 * sq_root) / 5
    return gamma, delta, epsilon, tau


def hab_e91_distilled(delta, epsilon):
    return binary_entropy(delta + epsilon)


def e91_qkd_distilled(F, m):
    gamma, delta, epsilon, tau = distilled_state(F)
    K_analytical = 1.0 - von_neumann(gamma, delta, epsilon, tau)

    # lambda00 = gamma, lambda01 = delta, lambda10 = epsilon, lambda11 = tau
    # (rho is NOT fixed here, only the statistics are constrained)
    hae, _ = hae_e91_from_state(bell_diagonal_state(gamma, delta, epsilon, tau), m)
    K_SDP = hae - hab_e91_distilled(delta, epsilon)
    return K_SDP, K_analytical, gamma


# ---------------------------- Bit flip -------------------------------
def hab_e91_bitflip(F):
    return binary_entropy(1 - F)


def e91_qkd_bitflip(F, m):
    hae, _ = hae_e91_from_state(bitflip_state(F), m)
    K_SDP = hae - hab_e91_bitflip(F)
    K_analytical = 1 - hab_e91_bitflip(F)
    return K_SDP, K_analytical


if __name__ == "__main__":
    fidelity = np.linspace(0.25, 1, 100)
    m = 10

    key_sdp_werner, key_an_werner = [], []
    key_sdp_distilled, key_an_distilled, F_out = [], [], []
    key_sdp_bitflip, key_an_bitflip = [], []

    for F in fidelity:
        k_sdp, k_an = e91_qkd(F, m)
        key_sdp_werner.append(k_sdp)
        key_an_werner.append(k_an)

        k_sdp, k_an, g = e91_qkd_distilled(F, m)
        key_sdp_distilled.append(k_sdp)
        key_an_distilled.append(k_an)
        F_out.append(g)

        k_sdp, k_an = e91_qkd_bitflip(F, m)
        key_sdp_bitflip.append(k_sdp)
        key_an_bitflip.append(k_an)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fidelity, key_sdp_werner, 'b', label='E91: Werner State')
    ax.plot(F_out, key_sdp_distilled, 'r', label='E91: Distilled State')
    ax.plot(fidelity, key_sdp_bitflip, 'g', label='E91: Bit Flip State')
    ax.plot(fidelity, key_an_werner, '--b', label='Six State: Werner State')
    ax.plot(F_out, key_an_distilled, '--r', label='Six State: Distilled State')
    ax.plot(fidelity, key_an_bitflip, '--g', label='Six State: Bit Flip State')
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Secret Key Rate')
    ax.set_title('E91 and Six State: comparison between states shared by Alice and Bob')
    ax.grid(True)
    ax.legend(loc='best')
    fig.savefig("comparisonE91vsSixS.pdf", bbox_inches="tight")
    plt.show()
