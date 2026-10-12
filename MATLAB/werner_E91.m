clc
clear
close all

format long
function [K_SDP, K_analytical, state_min] = e91_qkd(F, m)
% E91 protocol with Werner state (d=2), visibility v, quadrature level m
    %d = 2; werner states in 4x4
    v = (4*F - 1)/3;
    [Hae, state_min] = hae_e91(v, m);
    K_SDP = Hae - hab_e91(F);
    
    % key rate from analytical method
    [gamma, delta, eps, tau] = werner_params(F);
    % K_analytical = 1.0 - von_neumann(gamma, delta, eps, tau);
    K_analytical = 1.0 - binary_entropy(delta+eps) - binary_entropy(delta+tau);

end

function [gamma, delta, eps, tau] =  werner_params(F)
    gamma = F;
    delta = (1.0 - F) / 3.0;
    eps = (1.0 - F) / 3.0;
    tau = (1.0 - F) / 3.0;
end

function entropy = von_neumann(gamma, delta, eps, tau)
    entropy = -goodlog(max(gamma,0)) * gamma - goodlog(max(delta,0)) * delta - goodlog(max(eps,0)) * eps - goodlog(max(tau,0)) * tau;
end

function l = goodlog(s)
	if s == 0
		l = 0;
	else
		l = log2(s);
	end
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

    % Build POVM row
    nRows = 3 * 3 * (d^2 - 1);
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
                        E  = kron(pa, pb.')  ;   % note transpose on Bob's side
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

function rho_werner = werner_state(v)
    d = 2;
    phi_plus = [1;0;0;1]/sqrt(2);
    rho_werner = v * (phi_plus * phi_plus') + (1-v)/4 * eye(d^2);
end


function [objective, rho_min] = hae_e91(v, m)
    d = 2;
    [w, t] = gauss_radau(m);

    basis = e91_basis();
    % size(basis)
    % rank(basis)
    f_w   = werner_statistics(v, basis);

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

    constraints = [trace(rho) == 1, f_sdp == f_w];
    for i = 1:d
        for j = 1:m
            constraints = [constraints, ...
                [rho,         z(:,:,i,j);  z(:,:,i,j)',  zdz(:,:,i,j)] >= 0, ...
                [rho,         z(:,:,i,j)'; z(:,:,i,j),   zzd(:,:,i,j)] >= 0];
        end
    end

    % 
    % rho_werner = werner_state(v);
    % constraints = [constraints, rho == rho_werner];

    ops = sdpsettings('verbose', 0, 'solver', 'mosek');
    optimize(constraints, objective, ops);
    objective = sum(w./t)/log(2) + value(objective);
    % fprintf("after SDP\n")
    rho_min = value(rho);

end



Fidelity = linspace(0.25,1,30);
m = 10;
key_SDP = [];
key_analytical = [];
% state = cell(1,30);

for i = 1:length(Fidelity)
    F = Fidelity(i);
    [key_SDP(i), key_analytical(i), state{i}] = e91_qkd(Fidelity(i),m);
    % phi_col = [1;0;0;1]/sqrt(2);
    % phi_riga = [1 0 0 1]/sqrt(2);
    % out(i) = phi_riga*state{i}*phi_col;
    out(i) = (2*F+1)^2/9;
end
fig = figure();
plot(Fidelity, key_analytical, '-xb')
hold on
plot(Fidelity, key_SDP, '-or')
title("E91 Protocol")
legend("Analytical","SDP","Location","northwest")
grid on
xlabel("Fidelity F")
ylabel("Secret Key rate")
% exportgraphics(fig, "E91Werner.pdf", "ContentType", "vector", "BackgroundColor", "white");