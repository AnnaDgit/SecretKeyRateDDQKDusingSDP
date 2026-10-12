% DISTILLED STATES USING BBPSSW

clc
clear
close all

format short
function [K_SDP, K_analytical, gamma] = e91_qkd(F, m)
    % E91 protocol with Werner state (d=2), visibility v, quadrature level m
    %d = 2; diagonal states in 4x4

    % key rate from analytical method
    [gamma,delta,epsilon,tau] =  distilled_state(F);
    K_analytical = 1.0 - von_neumann(gamma,delta,epsilon,tau);
    % F = lambda00
    K_SDP = hae_e91(gamma,delta,epsilon,tau, m) - hab_e91(delta,epsilon);

end


function [gamma,delta,epsilon,tau] =  distilled_state(F0)
    % p = 1/9*(8*(F0)^2 - 4*F0 + 5);
    % F = (10*(F0)^2 - 2*F0 + 1)/(8*(F0)^2 - 4*F0 + 5);
    F = ((F0)^2 + 1/9*(1-F0)^2)/((F0)^2 +2/3*F0*(1-F0) +5/9*(1-F0)^2);
    gamma = F;
    % sqRoot = max(sqrt(-4*(F)^2 + 6*F - 1),0);
    sqRoot =sqrt(-4*(F)^2 + 6*F - 1);
    delta = 1/5*(2-F-sqRoot);
    epsilon = delta;
    tau = 1/5*(1-3*F + 2*sqRoot);
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

function out = hab_e91(delta,epsilon)
    Q = (delta + epsilon);
    out = binary_entropy(Q);
end

function base = e91_basis()
    d = 2;
    % base1  = [1, 0; 0, -1];
    % base = kron(base1,base1);

    % Alice's bases: Z and X
    aliceBases = zeros(2, 2, 2); % (vector, basis_index, basis_number)
    % fprintf("Alice's basis:")
    aliceBases(:,:,1) = eye(2);                          % Z: |0>,|1>
    aliceBases(:,:,2) = [1 1; 1 -1] / sqrt(2);          % X: |+>,|->

    % Bob's bases:
    th1 = pi/8;   
    th2 = 3*pi/8;
    % fprintf("Bob's basis:")
    bobBases = zeros(2, 2, 3);
    bobBases(:,:,1) = eye(2);
    bobBases(:,:,2) = [cos(th1) -sin(th1); sin(th1) cos(th1)];
    bobBases(:,:,3) = [cos(th2) -sin(th2); sin(th2) cos(th2)];

    % Build POVM rows: joint projectors E_{k,i,j} 
    nRows = 2 * 3 * (d^2 - 1);
    base = zeros(nRows, d^4);

    counter = 0;
    for ka = 1:2           % Alice's basis
        for kb = 1:3       % Bob's basis
            for i = 1:d
                for j = 1:d
                    if i ~= d || j ~= d    % drop last to avoid degeneracy
                        counter = counter + 1;
                        pa = aliceBases(:,i,ka) * aliceBases(:,i,ka)';
                        pb = bobBases(:,j,kb) * bobBases(:,j,kb)';
                        E  = kron(pa, pb.');     
                        base(counter,:) = E(:).';
                    end
                end
            end
        end
    end    
end


function f = bell_diagonal_statistics(lambda00,lambda01,lambda10,lambda11, base)
% Compute f_k = tr(E_k * rho_W) for each POVM element in base
    % d = 2;
    phi_plus = [1;0;0;1]/sqrt(2);
    phi_minus = [1;0;0;-1]/sqrt(2);
    psi_plus = [0;1;1;0]/sqrt(2);
    psi_minus = [0;1;-1;0]/sqrt(2);
    bell_d = lambda00*(phi_plus * phi_plus')+lambda01*(psi_minus * psi_minus')+lambda10*(psi_plus * psi_plus')+lambda11*(phi_minus * phi_minus');
    bell_d_vec = bell_d(:);
    % base rows are vec(E_k)', so f = base * rho_vec
    f = base * bell_d_vec;
end

function bell_d = bell_diagonal_state(lambda00,lambda01,lambda10,lambda11)
    phi_plus = [1;0;0;1]/sqrt(2);
    phi_minus = [1;0;0;-1]/sqrt(2);
    psi_plus = [0;1;1;0]/sqrt(2);
    psi_minus = [0;1;-1;0]/sqrt(2);
    bell_d = lambda00*(phi_plus * phi_plus')+lambda01*(psi_minus * psi_minus')+lambda10*(psi_plus * psi_plus')+lambda11*(phi_minus * phi_minus');
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

    f_sdp = real(basis * rho(:));
    
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
    % constraints = [];
    constraints = [trace(rho) == 1, f_sdp == f_w];
    for i = 1:d
        for j = 1:m
            constraints = [constraints, ...
                [rho,         z(:,:,i,j);  z(:,:,i,j)',  zdz(:,:,i,j)] >= 0, ...
                [rho,         z(:,:,i,j)'; z(:,:,i,j),   zzd(:,:,i,j)] >= 0];
        end
    end

    bell_d = bell_diagonal_state(lambda00,lambda01,lambda10,lambda11);
    constraints = [constraints, rho == bell_d];

    ops = sdpsettings('verbose', 0, 'solver', 'mosek');
    optimize(constraints, objective, ops);
    objective = sum(w./t)/log(2) + value(objective);

end



Fidelity = linspace(0.25,1,30);
m = 30;
key_SDP = [];
key_analytical = [];

for i = 1:length(Fidelity)
   
    [key_SDP(i), key_analytical(i), F(i)] = e91_qkd(Fidelity(i),m);

end
fig = figure();
plot(F, key_analytical, '-xb')
hold on
plot(F, key_SDP ,'-or')
legend("Key Rate - Analytical","Key Rate - SDP","Location","northwest")
grid on
xlabel("Fidelity F_{out}")
ylabel("Key rate")
title("Distilled State")
% exportgraphics(fig, "distilled.pdf", "ContentType", "vector", "BackgroundColor", "white");
