"""
Shared helpers for the E91 key-rate SDP scripts (Gauss-Radau quadrature
relaxation of the key rate, solved with cvxpy + MOSEK).
"""
import math
import numpy as np
import cvxpy as cp

# Bell states (columns of the computational basis |00>,|01>,|10>,|11>)
PHI_PLUS  = np.array([1, 0, 0,  1]) / np.sqrt(2)
PHI_MINUS = np.array([1, 0, 0, -1]) / np.sqrt(2)
PSI_PLUS  = np.array([0, 1,  1, 0]) / np.sqrt(2)
PSI_MINUS = np.array([0, 1, -1, 0]) / np.sqrt(2)


def goodlog(s):
    #log2 that returns 0 for s <= 0
    return 0.0 if s <= 0 else math.log2(s)


def binary_entropy(x):
    x = min(max(x, 0.0), 1.0)
    return -x * goodlog(x) - (1 - x) * goodlog(1 - x)


def von_neumann(*probs):
    #Entropy of a diagonal state given its eigenvalues
    return -sum(p * goodlog(p) for p in probs)


# Gauss-Radau quadrature
def gauss_radau(m):
    J = np.zeros((m, m))
    for i in range(1, m):
        J[i - 1, i - 1] = 0.5
        J[i - 1, i] = i / (2 * np.sqrt(4 * (i ** 2) - 1))
        J[i, i - 1] = J[i - 1, i]
    J[m - 1, m - 1] = (3 * m - 1) / (4 * m - 2)

    t, v = np.linalg.eigh(J)
    w = v[0, :] ** 2
    return w, t


# E91 measurements
def e91_basis():
    d = 2

    # Alice's bases: Z and X
    alice_bases = np.zeros((2, 2, 2))
    alice_bases[:, :, 0] = np.eye(2)
    alice_bases[:, :, 1] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    # Bob's bases
    th1 = np.pi / 8
    th2 = 3 * np.pi / 8
    bob_bases = np.zeros((2, 2, 3))
    bob_bases[:, :, 0] = np.eye(2)
    bob_bases[:, :, 1] = np.array([[np.cos(th1), -np.sin(th1)],
                                   [np.sin(th1),  np.cos(th1)]])
    bob_bases[:, :, 2] = np.array([[np.cos(th2), -np.sin(th2)],
                                   [np.sin(th2),  np.cos(th2)]])

    n_rows = 2 * 3 * d ** 2
    base = np.zeros((n_rows, d ** 4))
    counter = 0
    for ka in range(2):
        for kb in range(3):
            for i in range(d):
                for j in range(d):
                    pa = np.outer(alice_bases[:, i, ka], alice_bases[:, i, ka])
                    pb = np.outer(bob_bases[:, j, kb], bob_bases[:, j, kb])
                    E = np.kron(pa, pb.T)
                    base[counter, :] = E.flatten()
                    counter += 1
    return base


# States

def werner_state(v):
    return v * np.outer(PHI_PLUS, PHI_PLUS) + (1 - v) * np.eye(4) / 4


def bell_diagonal_state(l00, l01, l10, l11):
    return (l00 * np.outer(PHI_PLUS, PHI_PLUS)
            + l01 * np.outer(PSI_MINUS, PSI_MINUS)
            + l10 * np.outer(PSI_PLUS, PSI_PLUS)
            + l11 * np.outer(PHI_MINUS, PHI_MINUS))


def bitflip_state(F):
    return F * np.outer(PHI_PLUS, PHI_PLUS) + (1 - F) * np.outer(PSI_PLUS, PSI_PLUS)


def statistics(rho, base):
    """f_k = tr(E_k rho) for each POVM element in base."""
    return base @ rho.flatten()



# Generic SDP for H(A|E)

def hae_sdp(m, make_constraints, d=2, solver=cp.MOSEK):

    w, t = gauss_radau(m)

    rho = cp.Variable((d**2, d**2), hermitian=True)
    z   = [[cp.Variable((d**2, d**2), complex=True)   for j in range(m)] for i in range(d)]
    zdz = [[cp.Variable((d**2, d**2), hermitian=True) for j in range(m)] for i in range(d)]
    zzd = [[cp.Variable((d**2, d**2), hermitian=True) for j in range(m)] for i in range(d)]

    objective_expr = 0
    for j in range(m):
        for i in range(d):
            proj_a = np.zeros((d, d))
            proj_a[i, i] = 1.0
            partAB = np.kron(proj_a, np.eye(d))
            partAB_vec = partAB.flatten(order='F')

            partE = z[i][j] + cp.conj(z[i][j]).T + (1 - t[j]) * zdz[i][j]
            partE_vec = cp.reshape(partE, (d**4,), order='F')

            term = partAB_vec @ partE_vec + t[j] * cp.trace(zzd[i][j])
            objective_expr = objective_expr + (w[j] / (t[j] * np.log(2))) * cp.real(term)

    constraints = [cp.trace(rho) == 1] + make_constraints(rho)

    for i in range(d):
        for j in range(m):
            block1 = cp.bmat([[rho, z[i][j]],
                              [cp.conj(z[i][j]).T, zdz[i][j]]])
            block2 = cp.bmat([[rho, cp.conj(z[i][j]).T],
                              [z[i][j], zzd[i][j]]])
            constraints += [block1 >> 0, block2 >> 0]

    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    problem.solve(solver=solver, verbose=False)

    objective_value = np.sum(w / t) / np.log(2) + objective_expr.value
    return objective_value, rho.value


def statistics_constraints(f, base, rho_fixed=None):
    
    def make(rho):
        rho_vec = cp.reshape(rho, (rho.shape[0] ** 2,), order='F')
        cons = [cp.real(base @ rho_vec) == f]
        if rho_fixed is not None:
            cons.append(rho == rho_fixed)
        return cons
    return make


def hae_e91_from_state(rho_target, m, fix_rho=False):
   #H(A|E) given the E91 statistics produced by rho_target
    base = e91_basis()
    f = statistics(rho_target, base)
    return hae_sdp(m, statistics_constraints(f, base, rho_target if fix_rho else None))
