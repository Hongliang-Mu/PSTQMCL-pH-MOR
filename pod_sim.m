function [Q,P] = pod_sim(A,tspan, z0, Z_ref,V,para)
    %Symplectic integral with shifted initial values.
    [Q,P] = AVF_ROM(A,V,Z_ref,z0,tspan,para);
end

function [Q,P] = AVF_ROM(A,V,Z_ref,z0,tspan,para)
    tol_norm_res = 10^-12; %Tolerance for the Newton method
    max_it_newton = 10;    %The maximal Newton iteration times
    N = size(V,1);
    r = size(V,2);
    offline_terms.eye_2N = speye(2*N);
    offline_terms.eye_2r = speye(2*r);
    offline_terms.eye_r = speye(r);
    J2N = zeros([2*N, 2*N]);
    J2N(sub2ind(size(J2N), 1:N, N+(1:N))) = 1;
    J2N(sub2ind(size(J2N), N+(1:N), 1:N)) = -1;
    offline_terms.J2N = sparse(J2N);
    V = [V,0*V;0*V,V];
    offline_terms.J2r = V'*offline_terms.J2N*V;
    offline_terms.para = para;
    offline_terms.V = V;
    offline_terms.V_T_A = V'*A;
    offline_terms.A_r = offline_terms.V_T_A*V;
    idx_diag_upper_left = (((1:N)-1) * 2*N + (1:N))';
    idx_diag_upper_right = idx_diag_upper_left+N;
    idx_diag_lower_left = idx_diag_upper_left+2*N^2;
    idx_diag_lower_right = idx_diag_lower_left+N;
    offline_terms.idx_diag_upper_left=idx_diag_upper_left;
    offline_terms.idx_diag_upper_right=idx_diag_upper_right;
    offline_terms.idx_diag_lower_left=idx_diag_lower_left;
    offline_terms.idx_diag_lower_right=idx_diag_lower_right;
    offline_terms.Z_ref = Z_ref;
    
    Z = z0;
    delta_t = tspan(2)-tspan(1);
    z_old = z0;
    for i_t = 2:length(tspan)
        assert((i_t-1) * delta_t - tspan(i_t) < 10^3*eps, 'non-equidistant time-stepping not implemented')
        [z, ~, ~] = quasi_newton_AVF(z_old, delta_t, ...
            offline_terms,max_it_newton, tol_norm_res);
        Z = [Z,z];
        z_old = z;
    end
    Q = Z(1:r,:);
    P = Z(r+1:2*r,:);
end

function [x, norm_res, it_newton] = quasi_newton_AVF(x_old, dt, offline_terms, max_it_newton, tol_norm_res)
    x = x_old;
    res = x - x_old - (dt/8) * (eval_rhs(x_old, offline_terms)+3*eval_rhs((2*x_old+x)/3, offline_terms)+...
        3*eval_rhs((x_old+2*x)/3, offline_terms)+eval_rhs(x, offline_terms));
    norm_res = norm(res);
    it_newton = 0;
    while (it_newton < max_it_newton) && (norm_res > tol_norm_res)
        % update the state
        jac = offline_terms.eye_2r - dt/8 *( eval_jac_rhs((2*x_old+x)/3, offline_terms)+ ...
            2*eval_jac_rhs((x_old+2*x)/3, offline_terms)+eval_jac_rhs(x, offline_terms));
        x = x - jac \ res;
        % update residual
        res = x - x_old - (dt/8) * (eval_rhs(x_old, offline_terms)+3*eval_rhs((2*x_old+x)/3, offline_terms)+...
            3*eval_rhs((x_old+2*x)/3, offline_terms)+eval_rhs(x, offline_terms));
        norm_res = norm(res);
        it_newton = it_newton + 1;
    end
    if it_newton == max_it_newton
        warning('Max Newton iteration reached wir norm_res=%4.2e > %4.2e', norm_res, tol_norm_res);
    end
end

function rhs = eval_rhs(z, offline_terms)
    N = size(offline_terms.V,1)/2;
    z_full = offline_terms.V*z+offline_terms.Z_ref;
    q = z_full(1:N);
    q_cubic = q.^3;
    rhs = offline_terms.J2r *(offline_terms.V_T_A*z_full+offline_terms.V'*offline_terms.para*[q_cubic;0*q_cubic]);
end

function jac_rhs = eval_jac_rhs(z, offline_terms)
    jac_linear = offline_terms.J2r*offline_terms.A_r;
    N = size(offline_terms.V,1)/2;
    z_full = offline_terms.V*z+offline_terms.Z_ref;
    q = z_full(1:N);
    q_squa = q.^2;
    jac_nonlinear = sparse(2*N,2*N);
    jac_nonlinear(offline_terms.idx_diag_upper_left) = 3*q_squa;
    jac_nonlinear = offline_terms.J2r*offline_terms.para*offline_terms.V'*jac_nonlinear*offline_terms.V;
    jac_rhs = jac_linear+jac_nonlinear;
end