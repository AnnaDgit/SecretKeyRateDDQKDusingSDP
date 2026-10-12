%Comparison of werner states, distilled states, bell diagonal states using
%the six state protocol and the E91

clc
clear
close all

format short
function [K_SDP, K_analytical] = e91_qkd(F, m)
% E91 protocol with Werner state (d=2), visibility v, quadrature level m
    %d = 2; werner states in 4x4
    v = (4*F - 1)/3;
    % K_SDP = hae_e91(v, m) - hab_e91(F);
    
    % key rate from analytical method
    [gamma, delta, eps, tau] = werner_params(F);
    K_analytical = 1.0 - von_neumann(gamma, delta, eps, tau);
    K_SDP = hae_e91(v, m) - hab_e91(gamma);
end

function [gamma, delta, eps, tau] =  werner_params(F)
    gamma = F;
    delta = (1.0 - F) / 3.0;
    eps = (1.0 - F) / 3.0;
    tau = (1.0 - F) / 3.0;
end

function out = hab_e91(F)
    v = (4*F - 1)/3;
    Q = (1 - v) / 2;
    out = binary_entropy(Q);
    

end

function base = e91_basis()
    d = 2;

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
    % bobBases(:,:,2) = [1 1; 1 -1] / sqrt(2);
    bobBases(:,:,3) = [cos(th2) -sin(th2); sin(th2) cos(th2)];
    % bobBases(:,:,3) = eye(2);

    % Build POVM rows
    nRows = 3 * 3 * (d^2 - 1);
    base = zeros(nRows, d^4);

    % base1  = [1, 0; 0, -1];
    % base = kron(base1,base1);
    counter = 0;
    for ka = 1:2           % Alice's basis
        for kb = 1:3       % Bob's basis
            for i = 1:d
                for j = 1:d
                    if i ~= d || j ~= d    
                        counter = counter + 1;
                        pa = aliceBases(:,i,ka) * aliceBases(:,i,ka)';
                        pb = bobBases(:,j,kb) * bobBases(:,j,kb)';
                        E  = kron(pa, pb.')  ;   
                        base(counter,:) = E(:).';
                    end
                end
            end
        end
    end    
    % base
end

function f = werner_statistics(v, base)
% Compute f_k = tr(E_k * rho_W) for each POVM element in base
    d = 2;
    phi_plus = [1;0;0;1]/sqrt(2);
    rho_W = v * (phi_plus * phi_plus') + (1-v)/4 * eye(d^2);
    
    rho_vec = rho_W(:);
    % base rows are vec(E_k)', so f = base * rho_vec
    f = base * rho_vec;
end


function objective = hae_e91(v, m)
    d = 2;
    [w, t] = gauss_radau(m);

    basis = e91_basis();
    
    f_w   = werner_statistics(v, basis);

    yalmip('clear');

    rho = sdpvar(d^2, d^2, 'hermitian', 'complex');
    z   = sdpvar(d^2, d^2, d, m, 'full',      'complex');
    zdz = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');
    zzd = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');

    f_sdp = real(basis * rho(:));
    
    % Key generated in the basis Z: A^a_0 = |a><a|, a in {0,1}
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

function [K_SDP, K_analytical, gamma] = e91_qkd_distilled(F, m)
    
    % key rate from analytical method
    [gamma,delta,epsilon,tau] =  distilled_state(F);
    K_analytical = 1.0 - von_neumann(gamma,delta,epsilon,tau);
    % F = lambda00
    K_SDP = hae_e91_distilled(gamma,delta,epsilon,tau, m) - hab_e91_distilled(delta,epsilon);

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

function out = hab_e91_distilled(delta,epsilon)
    % v = (4*F - 1)/3;
    % Q = (1 - v) / 2;
    Q = (delta + epsilon);
    out = binary_entropy(Q);


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

function objective = hae_e91_distilled(lambda00,lambda01,lambda10,lambda11, m)
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
    
    % Key generated in the basis Z: A^a_0 = |a><a|, a in {0,1}
    objective = 0;
    for j = 1:m
        for i = 1:d
            partAB = kron(diag(ket(i,d)), eye(d));   % |i><i| ⊗ I_B
            partE  = z(:,:,i,j) + z(:,:,i,j)' ...
                     + (1 - t(j)) * zdz(:,:,i,j);
            objective = objective ...
                + (w(j)/(t(j)*log(2))) * ...
                  (partAB(:)' * partE(:) + t(j)*trace(zzd(:,:,i,j)));
        end
    end
   
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


function objective = hae_e91_bitflip(F, m)
    d = 2;
    [w, t] = gauss_radau(m);

    basis = e91_basis();
    f_w   = bitflip_statistics(F, basis);

    yalmip('clear');

    rho = sdpvar(d^2, d^2, 'hermitian', 'complex');
    z   = sdpvar(d^2, d^2, d, m, 'full',      'complex');
    zdz = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');
    zzd = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');

    f_sdp = real(basis * rho(:));
    
    % Key generated in the basis Z: A^a_0 = |a><a|, a in {0,1}
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

function out = hab_e91_bitflip(F)
    Q = (1 - F);
    out = binary_entropy(Q);

end

function f = bitflip_statistics(F, base)
    phi_plus = [1;0;0;1]/sqrt(2);
    psi_plus = [0;1;1;0]/sqrt(2);
    rho_W = F * (phi_plus * phi_plus') + (1-F) * (psi_plus * psi_plus');
    f = base * rho_W(:);
end


Fidelity = linspace(0.25,1,100);
m = 10;
key_SDP_werner = [];
key_analytical_werner = [];
key_SDP_distilled = [];
key_analytical_distilled = [];
F_min = [];
px = 0.2;
for i = 1:length(Fidelity)

    [key_SDP_werner(i), key_analytical_werner(i)] = e91_qkd(Fidelity(i),m);
    [key_SDP_distilled(i), key_analytical_distilled(i), F(i)] = e91_qkd_distilled(Fidelity(i),m);
    key_SDP_bitflip(i) = hae_e91_bitflip(Fidelity(i), m) - hab_e91_bitflip(Fidelity(i));
    key_analytical_bitflip(i) = 1-hab_e91_bitflip(Fidelity(i));
    %v = (4*Fidelity(i) - 1)/3;
    %S = 2*sqrt(2)*v;
    %QBER = 1/2 - v/2;

    %rDI(i) = 1-binary_entropy(QBER)-binary_entropy((1+sqrt((S/2)^2 - 1))/2);
    %rDD(i) = 1 - binary_entropy(QBER) - binary_entropy(QBER + S/(2*sqrt(2)));
end

fig = figure();
plot(Fidelity, key_SDP_werner, 'b')
hold on
plot(F, key_SDP_distilled, 'r')
plot(Fidelity, key_SDP_bitflip, 'g')

plot(Fidelity, key_analytical_werner,'--b')
plot(F, key_analytical_distilled, '--r')
plot(Fidelity, key_analytical_bitflip,'--g')

legend("E91: Werner State","E91: Distilled State","E91: Bit Flip State", "Six State: Werner State","Six State: Distilled State","Six State: Bit Flip State","Location","best")
xlabel("Fidelity")
ylabel("Secret Key Rate")
grid on
title("E91 and Six State: comparison between states shared by Alice and Bob")
% exportgraphics(fig, "comparisonE91vsSixS.pdf", "ContentType", "vector", "BackgroundColor", "white");

