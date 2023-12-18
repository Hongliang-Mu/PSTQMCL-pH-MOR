function [z_rec,z_red,run_time,newton_iterations]  = pstqmcl(z0,tspan,H_2_L,H_2_R,V,A,Z_ref,para)
    tol_norm_res = 10^-12; %Tolerance for the Newton method
    max_it_newton = 10;    %The maximal Newton iteration times
    update_tol = 10^-4;    %The tolerance of updating the piece-wise linear approximation.
    %Indeces respect to the unduplicate parts
    index = 1;
    r = size(V,2);  
    N = size(V,1);
    for i = 1:r
        for j = 1:i
            i_index(index) = i;
            j_index(index) = j;
            index = index+1;
        end
    end
    N = length(Z_ref)/2;
    Q_ref = Z_ref(1:N,:);
    P_ref = Z_ref(N+1:end,:);
    %Only H_2_L is used in offline part
    V_pro_mat = [zeros(N,1),V,H_2_L];
    pro_mat_q = [Q_ref, V, H_2_L];
    pro_mat_p = [P_ref, V, H_2_L];
    pro_mat = [pro_mat_q,0*pro_mat_q;0*pro_mat_p,pro_mat_p];
    offline_terms.pro_mat = pro_mat; 
    A_r = pro_mat'*A*pro_mat;
    offline_terms.A_r = A_r; 
    offline_terms.lin_q_lin_q = pro_mat_q' * pro_mat_q;

    idx_diag_upper_left = (((1:N)-1) * 2*N + (1:N))';
    idx_diag_upper_right = idx_diag_upper_left+N;
    idx_diag_lower_left = idx_diag_upper_left+2*N^2;
    idx_diag_lower_right = idx_diag_lower_left+N;
    offline_terms.idx_diag_upper_left=idx_diag_upper_left;
    offline_terms.idx_diag_upper_right=idx_diag_upper_right;
    offline_terms.idx_diag_lower_left=idx_diag_lower_left;
    offline_terms.idx_diag_lower_right=idx_diag_lower_right;
    offline_terms.para = para;

    lin_quad = zeros(size(H_2_L, 2), r, r);
    for ii = 1:length(i_index)
        lin_quad(ii, i_index(ii), j_index(ii)) = 1;
    end
    lin_quad = lin_quad + permute(lin_quad, [1, 3, 2]);
    %The computation online is done by the tensor product as follows.
    offline_terms.lin_quad = tensorprod(H_2_R',lin_quad, 1, 1);
    offline_terms.eye_2r = speye(2*r);
    offline_terms.eye_r = speye(r);
    J2r = zeros([2*r, 2*r]);
    J2r(sub2ind(size(J2r), 1:r, r+(1:r))) = 1;
    J2r(sub2ind(size(J2r), r+(1:r), 1:r)) = -1;
    offline_terms.J2r = sparse(J2r);
    
    %%
    %Simulation
    dt = tspan(2) - tspan(1);
    n_t = length(tspan);
    z_red = zeros(2*r, n_t);
    z_rec = zeros(2*N, n_t);
    z_red(:, 1) = z0;
    [term_d(:,1), term_Dd] = eval_gamma_terms(z_red(:, 1), offline_terms,  false);
    term_d_test = term_d;
    z_rec(:, 1) = pro_mat * term_d(:,1);
    newton_iterations = zeros(n_t - 1,1);
    norm_residuals = zeros(n_t - 1);
    tic
    for i_t = 2:1:n_t
        % check if time-stepping is equidistant
        assert((i_t-1) * dt - tspan(i_t) < 10^3*eps, 'non-equidistant time-stepping not implemented')
        % term_Dd_T_A_r = term_Dd'*A_r;
        [delta_z,term_Dd_PC, ~, newton_iterations(i_t-1)] = quasi_newton_AVF(z_red(:, i_t-1),term_d(:,i_t-1), ...
            dt, offline_terms, max_it_newton, tol_norm_res,update_tol);
        z_red(:, i_t) = z_red(:, i_t-1) + delta_z;
        term_d(:,i_t) = term_d(:,i_t-1) + term_Dd_PC*delta_z;
        [term_d_test(:,i_t), term_Dd] = eval_gamma_terms(z_red(:, i_t), offline_terms, false);
    end
    run_time = toc;
    z_rec = pro_mat * term_d;
end


function [x, term_Dd,norm_res, it_newton] = quasi_newton_AVF(x_ori,term_d, dt, offline_terms, max_it_newton, tol_norm_res,update_tol)
    r = size(offline_terms.eye_r,1);
    x = zeros(2*r,1);
    term_d_old = term_d;
    x_old = x;
    norm_res = 1;
    it_newton = 0;
        updata_flag = 0;
    %update_tol = update_tol;
    [~, term_Dd] = eval_gamma_terms(x_ori+x/2, offline_terms, false);
    online_terms.term_Dd = term_Dd;
    online_terms.term_d_old = term_d_old;
    online_terms.x_old = x_old;
    online_terms.term_Dd_T_A_r = term_Dd'*offline_terms.A_r;
    online_terms.pro_mat_term_Dd = offline_terms.pro_mat*term_Dd;
    online_terms.jac_linear =  online_terms.term_Dd_T_A_r*term_Dd;
    term_Dd_list{1} = term_Dd;
    while (it_newton < max_it_newton) && (norm_res > tol_norm_res)
        res = x - x_old - (dt/8) * (eval_rhs(x_old, online_terms , offline_terms)+3*eval_rhs((2*x_old+x)/3, online_terms , offline_terms)+...
            3*eval_rhs((x_old+2*x)/3, online_terms , offline_terms)+eval_rhs(x, online_terms , offline_terms));
        jac = offline_terms.eye_2r - dt/8 *( eval_jac_rhs((2*x_old+x)/3, online_terms , offline_terms)+ ...
            2*eval_jac_rhs((x_old+2*x)/3, online_terms , offline_terms)+eval_jac_rhs(x, online_terms , offline_terms));
        if it_newton ~= 0 && updata_flag ==0
            debug_value = (term_Dd_list{it_newton+1}-term_Dd_list{it_newton});
            debug_value = norm(debug_value,'fro')/norm(term_Dd_list{it_newton+1},'fro');
            if debug_value < update_tol
                updata_flag =1;
            end
        end
        x = x - jac \ res;
        if updata_flag == 0
            online_terms_old = online_terms;
            [~, term_Dd] = eval_gamma_terms(x_ori+x/2, offline_terms, false);
            online_terms.term_Dd = term_Dd;
            online_terms.term_d_old = term_d_old;
            online_terms.x_old = x_old;
            online_terms.term_Dd_T_A_r = term_Dd'*offline_terms.A_r;
            online_terms.pro_mat_term_Dd = offline_terms.pro_mat*term_Dd;
            online_terms.jac_linear =  online_terms.term_Dd_T_A_r*term_Dd;
            term_Dd_list{it_newton+2} = term_Dd;
        end
        norm_res = norm(res);
        it_newton = it_newton + 1;
    end
    if it_newton == max_it_newton
        warning('Max Newton iteration reached with norm_res=%4.2e > %4.2e', norm_res, tol_norm_res);
    end
end

function res = eval_rhs(x, online_terms ,offline_terms)
    term_Dd = online_terms.term_Dd;
    term_d_old = online_terms.term_d_old;
    x_old = online_terms.x_old;
    delta_x = x-online_terms.x_old;   
    term_d = term_d_old+term_Dd*(delta_x);
    term_Dd_T_A_r = online_terms.term_Dd_T_A_r;
    pro_mat_term_Dd = online_terms.pro_mat_term_Dd;
    rhs_linear = term_Dd_T_A_r*term_d;
    x_full = offline_terms.pro_mat*term_d;
    N = size(x_full,1)/2;
    q = x_full(1:N,:);
    q_cubic = q.^3;
    rhs_nonlinear = offline_terms.para*...
        (pro_mat_term_Dd'*[q_cubic;0*q_cubic]);
    res = offline_terms.J2r*(rhs_linear+rhs_nonlinear);
end

function jac = eval_jac_rhs(x, online_terms ,offline_terms)
    term_Dd = online_terms.term_Dd;
    term_d_old = online_terms.term_d_old;
    x_old = online_terms.x_old;
    delta_x = x-online_terms.x_old;  
    term_d = term_d_old+term_Dd*(delta_x);
    term_Dd_T_A_r = online_terms.term_Dd_T_A_r;
    pro_mat_term_Dd = online_terms.pro_mat_term_Dd;
    jac_linear =  online_terms.jac_linear;
    %Here the full-order state variable is computed
    x_full = offline_terms.pro_mat*term_d;
    N = size(x_full,1)/2;
    q = x_full(1:N,:);
    q_squa = q.^2;
    jac_nonlinear = sparse(2*N,2*N);
    jac_nonlinear(offline_terms.idx_diag_upper_left) = 3*q_squa;
    jac_nonlinear = offline_terms.para*...
        pro_mat_term_Dd'*jac_nonlinear*pro_mat_term_Dd;
    jac = offline_terms.J2r*(jac_linear+jac_nonlinear);
end

function [term_d, term_Dd] = eval_gamma_terms(x_red, offline_terms, eval_only_gamma)
    if nargin < 3
        eval_only_gamma = false;
    end
    r = size(x_red, 1) / 2;
    q_red = x_red(1:r);
    p_red = x_red(r+1:end);
    eval_lin_quad = tensorprod(offline_terms.lin_quad, q_red, 3, 1);
    dq_map_Q = [zeros(1, r);%const part
               offline_terms.eye_r;%lin part
               eval_lin_quad]; %quad part
    map_Q = [1; q_red; 1/2 * eval_lin_quad * q_red]; % see latex scirpt, formulation with B in Section 3.3
    mass_red = dq_map_Q' * offline_terms.lin_q_lin_q * dq_map_Q;
    if eval_only_gamma
        term_p = dq_map_Q * (mass_red \ p_red);
        term_Dd = false; % dummy value
        term_p(1) = 1; %enable constant term
        term_d = [map_Q; term_p];
        return
    else
        dq_map_Q_inv_mass_red = dq_map_Q / mass_red;
        term_p = dq_map_Q_inv_mass_red * p_red;
        term_p(1) = 1; %enable constant term
        term_d = [map_Q; term_p];
    end
    d2q_map_Q = [zeros(r+1, r, r); offline_terms.lin_quad];
    inv_mass_red_pr = mass_red \ p_red;
    dqr_mass_red = tensorprod(d2q_map_Q, offline_terms.lin_q_lin_q * dq_map_Q, 1, 1);
    dqr_mass_red = permute(dqr_mass_red, [1 3 2]);
    dqr_mass_red = dqr_mass_red + permute(dqr_mass_red, [2 1 3]);
    dqr_tan_map_part = tensorprod(d2q_map_Q, inv_mass_red_pr, 2, 1);
    dqr_inv_metric_part = dq_map_Q * ( ...
        mass_red \ tensorprod(dqr_mass_red, inv_mass_red_pr, 2, 1));
    dqr_mp_inv = dqr_tan_map_part - dqr_inv_metric_part;
    % naming from python implementation
    term_Dd = [dq_map_Q,    zeros(size(dq_map_Q));
               dqr_mp_inv, dq_map_Q_inv_mass_red];
end
