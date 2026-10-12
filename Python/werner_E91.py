import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
from numpy.linalg import eig

def e91_qkd(F, m):
    v = (4*F - 1)/3

    holevo_part, rho_opt = hae_e91(v, m)
    K_SDP = holevo_part - hab_e91(v)

    [gamma, delta, eps, tau]  = werner_params(F)
    K_analytical = 1.0 - binary_entropy(delta+eps) - binary_entropy(delta+tau)

    return K_SDP, K_analytical, rho_opt

def werner_params(F):
    gamma = (1.0 - F)/3.0
    delta = (1.0 - F)/3.0
    eps   = (1.0 - F)/3.0
    tau   = F
    return [gamma, delta, eps, tau]

def von_neumann(gamma, delta, eps, tau):
    entropy = -math.log2(gamma)*gamma - math.log2(delta)*delta - math.log2(eps)*eps - math.log2(tau)*tau
    return entropy

def hab_e91(v):
    Q = (1 - v)/2
    hab = binary_entropy(Q)
    return hab

def binary_entropy(x):
    def goodlog(s):
        if s == 0:
            return 0
        else:
            return math.log2(s)
    h = -x*goodlog(x) - (1-x)*goodlog(1-x)
    return h

def kronecker(A, B):
    mA, nA = A.shape
    mB, nB = B.shape
    C = np.zeros((mA*mB, nA*nB))
    for i in range(mA):
        for j in range(nA):
            C[i*mB:(i+1)*mB, j*nB:(j+1)*nB] = A[i,j]*B
    return C

def e91_basis():
    d = 2

    # Alice's bases: Z and X
    alice_bases = np.zeros((2, 2, 2))  
    alice_bases[:, :, 0] = np.eye(2)                        # Z: |0>, |1>
    alice_bases[:, :, 1] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # X: |+>, |->

    # Bob's bases
    th1 = np.pi / 8
    th2 = 3 * np.pi / 8

    bob_bases = np.zeros((2, 2, 3))  
    bob_bases[:, :, 0] = np.eye(2)
    bob_bases[:, :, 1] = np.array([[np.cos(th1), -np.sin(th1)],
                                    [np.sin(th1),  np.cos(th1)]])
    bob_bases[:, :, 2] = np.array([[np.cos(th2), -np.sin(th2)],
                                    [np.sin(th2),  np.cos(th2)]])

    # Build POVM rows
    n_rows = 2 * 3 * d**2
    base = np.zeros((n_rows, d**4))

    counter = 0
    for ka in range(2):        # Alice's basis index
        for kb in range(3):    # Bob's basis index
            for i in range(d):
                for j in range(d):
                    pa = np.outer(alice_bases[:, i, ka], alice_bases[:, i, ka])
                    pb = np.outer(bob_bases[:, j, kb],  bob_bases[:, j, kb])
                    E  = np.kron(pa, pb.T)             
                    base[counter, :] = E.flatten()
                    counter += 1

    return base

def werner_statistics(v, base):
    d = 2
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2) 
    rho_W = v * (np.outer(phi_plus, phi_plus)) + (1 - v) * np.eye(d**2) / d**2  # Werner state
    rho_W_vec = rho_W.flatten()
    f = base @ rho_W_vec
    return f

def werner_state(v):
    d =2
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2) 
    rho_W = v * (np.outer(phi_plus, phi_plus)) + (1 - v) * np.eye(d**2) / d**2  # Werner state
    return rho_W

def ket(index, d):
    vec = np.zeros(d)
    vec[index] = 1
    return vec

def gauss_radau(m):
    J = np.zeros((m, m))
    for i in range(1,m):
        J[i-1,i-1] = 0.5
        J[i-1,i] = i/(2*np.sqrt(4*(i**2)-1))
        J[i,i-1] = J[i-1,i]
    J[m-1,m-1] = (3*m-1)/(4*m-2)

    d, v = eig(J)
    w = np.zeros(m)
    for i in range(m):
        w[i] = v[0,i]**2
    #print("Gauss-Radau nodes:", d)
    #print("Gauss-Radau weights:", w)
    return w, d


def hae_e91(v, m):
    d = 2

    w, t = gauss_radau(m)
    basis = e91_basis()
    f_w = werner_statistics(v, basis)

    rho  = cp.Variable((d**2, d**2), hermitian=True)
    z   = [[cp.Variable((d**2, d**2), complex=True)    for j in range(m)] for i in range(d)]
    zdz = [[cp.Variable((d**2, d**2), hermitian=True)  for j in range(m)] for i in range(d)]
    zzd = [[cp.Variable((d**2, d**2), hermitian=True)  for j in range(m)] for i in range(d)]

    # Statistics constraint
    rho_vec = cp.reshape(rho, (d**4,), order='F')  
    f_sdp = cp.real(basis @ rho_vec)

    # Objective (Gauss–Radau quadrature approximation of key rate)
    objective_expr = 0
    for j in range(m):
        for i in range(d):
            
            proj_a  = np.zeros((d, d))
            proj_a[i, i] = 1.0
            partAB  = np.kron(proj_a, np.eye(d))  

            partAB_vec = partAB.flatten(order='F')          

            z_ij   = z[i][j]
            zdz_ij = zdz[i][j]
            zzd_ij = zzd[i][j]

            partE = z_ij + cp.conj(z_ij).T + (1 - t[j]) * zdz_ij

            partE_vec = cp.reshape(partE, (d**4,), order='F')

            term = (partAB_vec @ partE_vec
                    + t[j] * cp.trace(zzd_ij))

            objective_expr = objective_expr + (w[j] / (t[j] * np.log(2))) * cp.real(term)

    # Constraints
    constraints = [
        cp.trace(rho) == 1,
        f_sdp == f_w,
    ]

    for i in range(d):
        for j in range(m):
            z_ij   = z[i][j]
            zdz_ij = zdz[i][j]
            zzd_ij = zzd[i][j]

    
            block1 = cp.bmat([[rho, z_ij],
                               [cp.conj(z_ij).T, zdz_ij  ]])
           
            block2 = cp.bmat([[rho, cp.conj(z_ij).T],
                               [z_ij, zzd_ij ]])

            constraints += [block1 >> 0, block2 >> 0]

    # Fix rho to Werner state ---
    #rho_werner = werner_state(v)
    #constraints.append(rho == rho_werner)

    # Solve 
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    objective_value = np.sum(w / t) / np.log(2) + objective_expr.value

    rho_opt = rho.value
    #print("Optimal rho:\n", rho_opt)

    return objective_value, rho_opt

if __name__ == "__main__":

    Fidelity_RANGE = np.linspace(0.3, 0.99, 10)
    key_rates_sdp = []
    key_rates_analytical = []
    m = 8

    for i in range(Fidelity_RANGE.size):
        key_SDP, key_analytical, rho_opt = e91_qkd(Fidelity_RANGE[i], m)
        key_rates_sdp.append(key_SDP)
        key_rates_analytical.append(key_analytical)
        #print(f"Fidelity: {Fidelity_RANGE[i]:.2f}, Optimal rho: {rho_opt}")
        #print("---------------------")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Fidelity_RANGE, key_rates_sdp, label='SDP Lower Bound', marker='o', color='blue')
    ax.plot(Fidelity_RANGE, key_rates_analytical, label='Analytical Key Rate', marker='x', color='red')
    ax.set_xlabel('Fidelity F')
    ax.set_ylabel('Key Rate')
    ax.set_title('E91 QKD Key Rates vs Fidelity')
    ax.legend()
    plt.show()


#if __name__ == "__main__":
 #   key_SDP, key_analytical = e91_qkd(0.95, 8)