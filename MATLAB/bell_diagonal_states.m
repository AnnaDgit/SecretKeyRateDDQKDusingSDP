clc
clear
close all

format short
function [K_SDP, K_analytical] = e91_qkd(QBER, m)
% E91 protocol with Werner state (d=2), visibility v, quadrature level m
    %d = 2; diagonal states in 4x4

    % key rate from analytical method
    [lambda00,lambda01,lambda10,lambda11] = bell_diagonal_params(QBER);
    K_analytical = 1.0 - von_neumann(lambda00,lambda01,lambda10,lambda11);
    % K_analytical = 1.0 - binary_entropy(QBER(1)) - binary_entropy(QBER(2));
   
    % F = lambda00
    K_SDP = hae_e91(lambda00,lambda01,lambda10,lambda11, m) - hab_e91(lambda10,lambda11);

end

function [lambda00,lambda01,lambda10,lambda11] =  bell_diagonal_params(QBER)
    Qx = QBER(1); 
    Qy = QBER(2);
    Qz = QBER(3);
    lambda00 = 1 -(Qx+Qy+Qz)/2;
    lambda01 = (Qx+Qy-Qz)/2;
    lambda10 = (-Qx+Qy+Qz)/2;
    lambda11 = (Qx-Qy+Qz)/2;
end

function entropy = von_neumann(gamma, delta, eps, tau)
    entropy = -goodlog(gamma) * gamma - goodlog(delta) * delta - goodlog(eps) * eps - goodlog(tau) * tau;
end

function l = goodlog(s)
	if s == 0 || s<0
		l = 0;
	else
		l = log2(s);
	end
end

function out = hab_e91(lambda10,lambda11)
    QBER = lambda10+lambda11;
    out = binary_entropy(QBER);
end

function base = e91_basis()
    base1  = [0, 1; 1, 0];
    base = kron(base1,base1);   
end


function f = bell_diagonal_statistics(lambda00,lambda01,lambda10,lambda11, base)
% Compute f_k = tr(E_k * rho_W) for each POVM element in base
    % d = 2;
    phi_plus = [1;0;0;1]/sqrt(2);
    phi_minus = [1;0;0;-1]/sqrt(2);
    psi_plus = [0;1;1;0]/sqrt(2);
    psi_minus = [0;1;-1;0]/sqrt(2);
    bell_d = lambda00*(phi_plus * phi_plus')+lambda01*(phi_minus * phi_minus')+lambda10*(psi_plus * psi_plus')+lambda11*(psi_minus * psi_minus');
    % bell_d_vec = bell_d(:);
    % base rows are vec(E_k)', so f = base * rho_vec
    f = base * bell_d;
end

function bell_d = bell_diagonal_state(lambda00,lambda01,lambda10,lambda11)
    phi_plus = [1;0;0;1]/sqrt(2);
    phi_minus = [1;0;0;-1]/sqrt(2);
    psi_plus = [0;1;1;0]/sqrt(2);
    psi_minus = [0;1;-1;0]/sqrt(2);
    bell_d = lambda00*(phi_plus * phi_plus')+lambda01*(phi_minus * phi_minus')+lambda10*(psi_plus * psi_plus')+lambda11*(psi_minus * psi_minus');
end

function objective = hae_e91(lambda00,lambda01,lambda10,lambda11, m)
    d = 2;
    [w, t] = gauss_radau(m);

    basis = e91_basis();
    % size(basis)
    % rank(basis)
    f_w   = bell_diagonal_statistics(lambda00,lambda01,lambda10,lambda11, basis);

    yalmip('clear');

    rho = sdpvar(d^2, d^2, 'hermitian', 'complex');
    z   = sdpvar(d^2, d^2, d, m, 'full',      'complex');
    zdz = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');
    zzd = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');

    f_sdp = real(basis * rho);
    
    objective = 0;
    for j = 1:m
        for i = 1:d
            partAB = kron(diag(ket(i,d)), eye(d));   
            partE  = z(:,:,i,j) + z(:,:,i,j)' ...
                     + (1 - t(j)) * zdz(:,:,i,j);
            objective = objective ...
                + (w(j)/(t(j)*log(2))) * ...
                  (partAB(:)' * partE(:) + t(j)*trace(zzd(:,:,i,j)));
        end
    end
    % fprintf("before SDP\n")
    constraints = [];
    constraints = [trace(rho) == 1, f_sdp == f_w];
    for i = 1:d
        for j = 1:m
            constraints = [constraints, ...
                [rho,         z(:,:,i,j);  z(:,:,i,j)',  zdz(:,:,i,j)] >= 0, ...
                [rho,         z(:,:,i,j)'; z(:,:,i,j),   zzd(:,:,i,j)] >= 0];
        end
    end


    ops = sdpsettings('verbose', 0, 'solver', 'mosek');
    optimize(constraints, objective, ops);
    objective = sum(w./t)/log(2) + value(objective);

end



Qy = linspace(0.0,0.2,30);
m = 30;
key_SDP = [];
key_analytical = [];

for i = 1:length(Qy)
    QBER = [0.1, Qy(i), 0.1];
    [key_SDP(i), key_analytical(i)] = e91_qkd(QBER,m);

end
fig = figure();
plot(Qy, key_analytical, '-xb')
hold on
plot(Qy, key_SDP, '-or')
legend("Key Rate - Analytical","Key Rate - SDP","Location","northeast")
grid on
xlabel("Qy")
ylabel("Key rate")
title("Bell Diagonal State Qx = Qz = 0.1")
% exportgraphics(fig, "belldiagonal_Qy.pdf", "ContentType", "vector", "BackgroundColor", "white");