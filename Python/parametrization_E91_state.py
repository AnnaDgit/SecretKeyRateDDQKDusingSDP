"""
For a Werner state shared by Alice and Bob, solve the E91 SDP for a set of
fidelities, collect the optimal rho_AB (the state Eve's attack is consistent
with) and fit each matrix element with a polynomial in the fidelity.
"""
import numpy as np
import matplotlib.pyplot as plt

from e91_common import binary_entropy, werner_state, hae_e91_from_state


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
    hae, state_min = hae_e91_from_state(werner_state(v), m)   # rho not fixed
    K_SDP = hae - hab_e91(F)

    gamma, delta, eps, tau = werner_params(F)
    K_analytical = 1.0 - binary_entropy(delta + eps) - binary_entropy(delta + tau)
    return K_SDP, K_analytical, state_min


def find_parametrization(i, j, fidelity, vector, ax, degree=2):
    ax.plot(fidelity, vector.real, '*r')
    coeffs = np.polyfit(fidelity, vector, degree)
    ax.plot(fidelity, np.polyval(coeffs, fidelity).real, '-b', linewidth=0.8)
    ax.set_title(f"({i+1},{j+1})", fontsize=8)
    print(f"interpolation of the matrix ({i+1}, {j+1})")
    print(coeffs)
    return coeffs


if __name__ == "__main__":
    np.set_printoptions(precision=15, suppress=False)

    n_points = 30
    fidelity = np.linspace(0.25, 1, n_points)
    m = 10

    key_sdp, key_analytical, states = [], [], []
    for F in fidelity:
        k_sdp, k_an, state = e91_qkd(F, m)
        key_sdp.append(k_sdp)
        key_analytical.append(k_an)
        states.append(state)

    n = states[0].shape[0]
    fig, axes = plt.subplots(n, n, figsize=(10, 8), sharex=True)
    for i in range(n):
        for j in range(n):
            vector = np.array([states[k][i, j] for k in range(n_points)])
            find_parametrization(i, j, fidelity, vector, axes[i, j])
    fig.supxlabel("Fidelity")
    fig.tight_layout()
    plt.show()
