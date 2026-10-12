"""
Six-state protocol with a Werner state: Alice measures Z, X, Y and Bob measures
Z, two CHSH-angle rotations and Y. The SDP key rate is compared with the
analytical six-state rate.
"""
import numpy as np
import matplotlib.pyplot as plt

from e91_common import (binary_entropy, von_neumann, werner_state, hae_sdp,
                        statistics_constraints, PHI_PLUS)


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


def six_state_basis():
    d = 2
    n_alice, n_bob = 3, 4

    y_basis = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)   # |+i>, |-i>

    # Alice: Z, X, Y   (columns are the basis vectors)
    alice_bases = np.zeros((2, 2, n_alice), dtype=complex)
    alice_bases[:, :, 0] = np.eye(2)
    alice_bases[:, :, 1] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    alice_bases[:, :, 2] = y_basis

    # Bob: Z, two CHSH-angle rotations, Y
    th1, th2 = np.pi / 8, 3 * np.pi / 8
    bob_bases = np.zeros((2, 2, n_bob), dtype=complex)
    bob_bases[:, :, 0] = np.eye(2)
    bob_bases[:, :, 1] = np.array([[np.cos(th1), -np.sin(th1)],
                                   [np.sin(th1),  np.cos(th1)]])
    bob_bases[:, :, 2] = np.array([[np.cos(th2), -np.sin(th2)],
                                   [np.sin(th2),  np.cos(th2)]])
    bob_bases[:, :, 3] = y_basis

    n_rows = n_alice * n_bob * d ** 2
    base = np.zeros((n_rows, d ** 4), dtype=complex)
    counter = 0
    for ka in range(n_alice):
        for kb in range(n_bob):
            for i in range(d):
                for j in range(d):
                    a, b = alice_bases[:, i, ka], bob_bases[:, j, kb]
                    pa = np.outer(a, a.conj())
                    pb = np.outer(b, b.conj())
                    E = np.kron(pa.T, pb.T)
                    base[counter, :] = E.flatten(order='F')   
                    counter += 1
    return base


def werner_statistics(v, base):
    rho_vec = werner_state(v).flatten(order='F')
    return np.real(base @ rho_vec)


def hae_e91(v, m):
    base = six_state_basis()
    f_w = werner_statistics(v, base)
    return hae_sdp(m, statistics_constraints(f_w, base))


def e91_qkd(F, m):
    v = (4 * F - 1) / 3
    hae, state_min = hae_e91(v, m)
    K_SDP = hae - hab_e91(F)

    gamma, delta, eps, tau = werner_params(F)
    K_analytical = 1.0 - von_neumann(gamma, delta, eps, tau)
    return K_SDP, K_analytical, state_min


if __name__ == "__main__":
    np.set_printoptions(precision=15)

    fidelity = np.linspace(0.25, 1, 30)
    m = 10

    key_sdp, key_analytical, states, overlap = [], [], [], []
    for F in fidelity:
        k_sdp, k_an, state = e91_qkd(F, m)
        key_sdp.append(k_sdp)
        key_analytical.append(k_an)
        states.append(state)
        overlap.append(np.real(PHI_PLUS @ state @ PHI_PLUS))   # <phi+|rho_opt|phi+>

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fidelity, key_analytical, '-xb', label='Analytical')
    ax.plot(fidelity, key_sdp, '-or', label='SDP')
    ax.set_xlabel('Fidelity F')
    ax.set_ylabel('Secret Key rate')
    ax.set_title('Six State Protocol')
    ax.grid(True)
    ax.legend(loc='upper left')
    # fig.savefig("SixStateWerner.pdf", bbox_inches="tight")
    plt.show()
