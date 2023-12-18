function [Q,P,Y,A2,hamiltonian] = comp_snap_2d(nx,ny,delta_t,time,para)
    %AVF means the average vector field
    %fom is setup from jiang et al. 2020
    tol_norm_res = 10^-12;
    max_it_newton = 10;
    xu = 10;
    xl = -10;
    yu = 10;
    yl = -10;
    dt = delta_t;
    T = time;
    tspan = 0:delta_t:time;
    max_iter_newton = 10;
    tol_abs_err_newton = 1e-8;
    grid_x = linspace(xl, xu, nx+1);
    grid_x = grid_x(1:end-1); % truncate right boundary
    grid_y = linspace(yl, yu, ny+1);
    grid_y = grid_y(1:end-1); % truncate right boundary
    [meshgrid_y, meshgrid_x] = meshgrid(grid_y, grid_x);
    meshgrid_x = meshgrid_x(:);
    meshgrid_y = meshgrid_y(:);
    %
    lx = xu-xl;
    ly = yu-yl;
    dx = lx/nx;
    dy = ly/ny;
    n = nx*ny;
    N = n;
    nt = T/dt;
    % spatial discretization
    tic;
    Aq = space_disc(n,nx,ny,dx,dy);
    Ap = speye(n);
    A = [sparse(n,n), -Ap;
        Aq,          sparse(n,n)];
    A2 = [Aq          sparse(n,n);
        sparse(n,n),Ap         ];
    rt_spatial_discretization = toc();

    offline_terms.eye_2N = speye(2*N);
    offline_terms.eye_N = speye(N);
    J2N = zeros([2*N, 2*N]);
    J2N(sub2ind(size(J2N), 1:N, N+(1:N))) = 1;
    J2N(sub2ind(size(J2N), N+(1:N), 1:N)) = -1;
    offline_terms.J2N = sparse(J2N);
    offline_terms.para = para;
    offline_terms.A2 = A2;
    idx_diag_upper_left = (((1:N)-1) * 2*N + (1:N))';
    idx_diag_upper_right = idx_diag_upper_left+N;
    idx_diag_lower_left = idx_diag_upper_left+2*N^2;
    idx_diag_lower_right = idx_diag_lower_left+N;
    offline_terms.idx_diag_upper_left=idx_diag_upper_left;
    offline_terms.idx_diag_upper_right=idx_diag_upper_right;
    offline_terms.idx_diag_lower_left=idx_diag_lower_left;
    offline_terms.idx_diag_lower_right=idx_diag_lower_right;
    
    % init arrays
    Q = zeros(n, nt);
    P = zeros(n, nt);
    hamiltonian = zeros(1, nt+1);
    
    % initial value
    [q0,p0] = init(meshgrid_x, meshgrid_y, n);
    Q(:, 1) = q0;
    P(:, 1) = p0;
    hamiltonian(1) = compute_hamiltonian(Q(:, 1), P(:, 1), Aq, Ap ,para);
    xg = [q0; p0];
    

    X = xg;
    %delta_t = tspan(2)-tspan(1);
    x_old = xg;
    for i_t = 2:length(tspan)
        assert((i_t-1) * delta_t - tspan(i_t) < 10^3*eps, 'non-equidistant time-stepping not implemented')
        if mod(i_t, floor(nt/10)) == 0
            fprintf('time integration step: %d / %d\n', i_t, nt)
        end
        [x, ~, ~] = quasi_newton_AVF(x_old, delta_t, ...
            offline_terms,max_it_newton, tol_norm_res);
        X = [X,x];
        x_old = x;
        Q(:,i_t)=x(1:n,1);
        P(:,i_t)=x(n+1:2*n,1);
        hamiltonian(i_t) = compute_hamiltonian(Q(:, i_t), P(:, i_t), Aq, Ap,para);
    end
    Q = X(1:N,:);
    P = X(N+1:2*N,:);
    hamiltonian = hamiltonian*dx*dy;
    Y = [Q;P];
end


function [x, norm_res, it_newton] = quasi_newton_AVF(x_old, dt, offline_terms, max_it_newton, tol_norm_res)
    x = x_old;
    res = x - x_old - (dt/8) * (eval_rhs(x_old, offline_terms)+3*eval_rhs((2*x_old+x)/3, offline_terms)+...
        3*eval_rhs((x_old+2*x)/3, offline_terms)+eval_rhs(x, offline_terms));
    norm_res = norm(res);
    it_newton = 0;
    while (it_newton < max_it_newton) && (norm_res > tol_norm_res)
        % update the state
        jac = offline_terms.eye_2N - dt/8 *( eval_jac_rhs((2*x_old+x)/3, offline_terms)+ ...
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



function ham = compute_hamiltonian(qk, pk, Aq, Ap,para)
    ham = (0.5*qk'*Aq*qk) + (0.5*pk'*Ap*pk) +  para*sum(qk.^4)/4;
end


%% initial condition
function [q0,p0]=init(grid_x, grid_y, n)
    q0 = 2*sech(cosh(grid_x.^2 + grid_y.^2));
    p0=zeros(n,1);
end

%% nonlinear solver
function rhs = eval_rhs(x, offline_terms)
    N = length(x)/2;
    q = x(1:N);
    p = x(N+1:2*N);
    fq = offline_terms.para*x(1:N).^3;
    rhs = offline_terms.J2N *(offline_terms.A2*x+[fq;0*fq]);
end

function jac_rhs = eval_jac_rhs(x, offline_terms)
    jac_rhs = offline_terms.J2N*offline_terms.A2;
    N = length(x)/2;
    q = x(1:N);
    q_squa = q.^2;
    jac_nonlinear = sparse(2*N,2*N);
    jac_nonlinear(offline_terms.idx_diag_upper_left) = 3*q_squa;
    jac_nonlinear = offline_terms.J2N*offline_terms.para*jac_nonlinear;
    jac_rhs = jac_rhs+jac_nonlinear;
end

%% function to obtain space-discretized
function Aq = space_disc(n,nx,ny,dx,dy)
%
idx_1 = zeros(5*n, 1);
idx_2 = zeros(5*n, 1);
val = zeros(5*n, 1);
k_idx = 1;
for i=1:nx
    for j=1:ny
        in=nx*(j-1) + i;
        %
        % product for node itself
        idx_1(k_idx+(0:4)) = in;
        idx_2(k_idx) = in;
        val(k_idx) = (2/(dx*dx)) + (2/(dy*dy));
        % right neighbor
        if i == nx
            idx_2(k_idx+1) = nx*(j-1) + 1;
        else
            idx_2(k_idx+1) = in+1;
        end
        val(k_idx+1) = -1/(dx*dx);
        % left neighbor
        if i == 1
            idx_2(k_idx+2) = nx*(j-1) + nx;
        else
            idx_2(k_idx+2) = in-1;
        end
        val(k_idx+2) = -1/(dx*dx);
        % lower neighbor
        if j == 1
            idx_2(k_idx+3) = nx*(ny-1) + i;
        else
            idx_2(k_idx+3) = in-nx;
        end
        val(k_idx+3) = -1/(dy*dy);
        % upper neighbor
        if j == ny
            idx_2(k_idx+4) = nx*(1-1) + i;
        else
            idx_2(k_idx+4) = in+nx;
        end
        val(k_idx+4) = -1/(dy*dy);
        k_idx = k_idx + 5;
    end
end
Aq = sparse(idx_1, idx_2, val, n, n);
end