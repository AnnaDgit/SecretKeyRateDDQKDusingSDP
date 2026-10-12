clc
clear
close all

format short
function [K_SDP, K_analytical] = e91_qkd(F, m)
% E91 protocol with Werner state (d=2), visibility v, quadrature level m
    %d = 2; werner states in 4x4
    K_SDP = hae_e91(F, m) - hab_e91(F);
    
    % key rate from analytical method
    [gamma, delta] = werner_params(F);
    % K_analytical = 1.0 - von_neumann(gamma, delta);
    K_analytical = 1.0 - binary_entropy(F);

end

function [gamma, delta] =  werner_params(F)
    gamma = F;
    delta = (1.0 - F);
end

function entropy = von_neumann(gamma, delta)
    entropy = -goodlog(max(gamma,0)) * gamma - goodlog(max(delta,0)) * delta;
end

function l = goodlog(s)
	if s == 0
		l = 0;
	else
		l = log2(s);
	end
end

function out = hab_e91(F)
    Q = (1 - F);
    out = binary_entropy(Q);

end

function base = e91_basis()
    base1  = [1, 0; 0, -1];
    base = kron(base1,base1);

end


function f = bitflip_statistics(F, base)
% Compute f_k = tr(E_k * rho_W) for each POVM element in base
    d = 2;
    phi_plus = [1;0;0;1]/sqrt(2);
    psi_plus = [0;1;1;0]/sqrt(2);
    rho_W = F * (phi_plus * phi_plus') + (1-F) * (psi_plus * psi_plus');
    
    rho_vec = rho_W(:);
    % base rows are vec(E_k)', so f = base * rho_vec
    f = base * rho_W;
end

function rho_werner = werner_state(v)
    d = 2;
    phi_plus = [1;0;0;1]/sqrt(2);
    rho_werner = v * (phi_plus * phi_plus') + (1-v)/4 * eye(d^2);
end

function objective = hae_e91(F, m)
    d = 2;
    [w, t] = gauss_radau(m);

    basis = e91_basis();
    f_w   = bitflip_statistics(F, basis);

    yalmip('clear');

    rho = sdpvar(d^2, d^2, 'hermitian', 'complex');
    z   = sdpvar(d^2, d^2, d, m, 'full',      'complex');
    zdz = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');
    zzd = sdpvar(d^2, d^2, d, m, 'hermitian', 'complex');

    f_sdp = real(basis * rho);
    
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



Fidelity = linspace(0,1,30);
m = 8;
key_SDP = [];
key_analytical = [];

for i = 1:length(Fidelity)

    [key_SDP(i), key_analytical(i)] = e91_qkd(Fidelity(i),m);

end
fig = figure();
plot(Fidelity, key_analytical, '-xb')
hold on
plot(Fidelity, key_SDP, '-or')
legend("Key Rate - Analytical","Key Rate - SDP", "Location","best")
grid on
xlabel("Fidelity")
ylabel("Key rate")
title("Bit-Flip State")
% exportgraphics(fig, "bitflip.pdf", "ContentType", "vector", "BackgroundColor", "white");




