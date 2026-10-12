"""
Bit-flip state  rho = F |phi+><phi+| + (1-F) |psi+><psi+|.
"""
import numpy as np
import matplotlib.pyplot as plt

from e91_common import binary_entropy, bitflip_state, hae_sdp
import cvxpy as cp


def hab_e91(F):
    return binary_entropy(1 - F)


def e91_basis_zz():
    z = np.array([[1, 0], [0, -1]])
    return np.kron(z, z)


def hae_e91(F, m):
    base = e91_basis_zz()
    f_w = base @ bitflip_state(F)

    def make_constraints(rho):
        return [cp.real(base @ rho) == f_w]

    value, _ = hae_sdp(m, make_constraints)
    return value


def e91_qkd(F, m):
    K_SDP = hae_e91(F, m) - hab_e91(F)
    K_analytical = 1.0 - binary_entropy(F)
    return K_SDP, K_analytical


if __name__ == "__main__":
    fidelity = np.linspace(0, 1, 30)
    m = 8

    key_sdp, key_analytical = [], []
    for F in fidelity:
        k_sdp, k_an = e91_qkd(F, m)
        key_sdp.append(k_sdp)
        key_analytical.append(k_an)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fidelity, key_analytical, '-xb', label='Key Rate - Analytical')
    ax.plot(fidelity, key_sdp, '-or', label='Key Rate - SDP')
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Key rate')
    ax.set_title('Bit-Flip State')
    ax.grid(True)
    ax.legend(loc='best')
    # fig.savefig("bitflip.pdf", bbox_inches="tight")
    plt.show()
