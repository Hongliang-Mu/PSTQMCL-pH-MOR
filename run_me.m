clear all
close all
%Initial settings for the wave equation.
delta_t  = 0.1; 
time = 8; % The simulation time  
nx = 50; 
ny = 50; % The size of space discritization
N = nx*ny;
delta_x = 400/N;
tspan = 0:delta_t:time; % The time steps for simulations
%%
%Set the training set and the testing set.
Q_snap =[];
P_snap =[];
%The testing parameter set and the training parameter set.
train_size =5;
train_set = 0.2:0.2:1;
test_set = 0.3:0.2:0.9;
para_set = [train_set,test_set];
para_size = length(para_set);
para_ind = 1:1:para_size;
train_ind = 1:1:train_size;
test_ind = train_size+1:1:para_size;
%Compute snapshots.
for i = 1:1:para_size
    [Q{i},P{i},U{i},A,Ham_ori(:,i)] =  comp_snap_2d(nx,ny,delta_t,time,para_set(i));
    Q_Ref{i} = 1*Q{i}(:,1); 
    P_Ref{i} = 1*P{i}(:,1); 
    Q_snap_para{i} = Q{i}-Q_Ref{i}*ones(1,size(Q{i},2));
    P_snap_para{i} = P{i}-P_Ref{i}*ones(1,size(P{i},2));
    %Snapshots of the training set.
    if ismember(i,train_ind)
        Q_snap = [Q_snap,Q_snap_para{i}]; 
        P_snap = [P_snap,P_snap_para{i}];
    end
    % Precompute the norm of snapshots for the relative error.
    norm_Q_para(i) = norm(Q{i},"fro");
    norm_P_para(i) = norm(P{i},"fro");
    norm_U_para(i) = norm(U{i},"fro");
end
%Compute the POD modes by PSD (Cotangent lift)
[V,~,~] = svd([Q_snap,P_snap],'econ');
%%
r_base = 6:2:16; %size of the ROMs and the setting 
lambda = 10^-2;  %Regularization parameter
para_ind = test_ind;
for i = 1:1:length(r_base)
    %The reduced order is r.
    r = r_base(i);
    V1 = V(:,1:r);
    Q_r = V1'*Q_snap;
    res_Q = Q_snap-V1*Q_r;
    kron_Q = col_kron(Q_r);
    [svd_L,Lam,svd_R] = svd(kron_Q,'econ'); 
    r_q_base = r*(0.5:0.5:2);
    norm_Q = norm(Q_snap,'fro');
    for j = 1:length(r_q_base)
        r_q = r_q_base(j);
        %Apply the SVD truncation to the quadratic terms
        kron_Q_L = svd_L(:,1:r_q);
        kron_Q_R = Lam(1:r_q,1:r_q)*svd_R(:,1:r_q)';
        svd_trun_err(r,j) = 1 - sum(diag(Lam(1:r_q,1:r_q)))/sum(diag(Lam));
        [H_2_L] = regu_solver(kron_Q_R,res_Q,lambda);
        H_2_R = kron_Q_L';
        norm_Vbar_err(r,j) = norm(H_2_L,'fro');
        nonL_res_Q = res_Q-H_2_L*kron_Q_R;
        nonL_res_Q = res_Q-H_2_L*H_2_R*kron_Q;
        Q_nonL_err(r,j) = norm(nonL_res_Q,'fro')/norm_Q;
        for k= para_ind
            %Compute the projection error of the TQMCL method.
            [q0,p0,Q_TQMCL_proj_error(r,j,k),P_TQMCL_proj_error(r,j,k)] = proj_error(Q{k},P{k}, ...
                norm_Q_para(k), norm_P_para(k),Q_Ref{k},P_Ref{k},H_2_L*H_2_R,V1);
            Z_Ref = [Q_Ref{k};P_Ref{k}];
            z0 = [q0;q0];
             %Simulate the PSTQMCL ROMs
            [z_rec,~,run_time]  = pstqmcl(z0,tspan,H_2_L,H_2_R,V1,A,Z_Ref,para_set(k));
            [Q_PSTQMCL_sim_error(r,j,k),P_PSTQMCL_sim_error(r,j,k),U_PSTQMCL_sim_error(r,j,k),Ham_PSTQMCL(:,r,j,k)] = red_Ham_error(z_rec,...
                Q{k},P{k},U{k},norm_Q_para(k),norm_P_para(k),norm_U_para(k),A,para_set(k),delta_x);
            %Simulate the TQMCL ROMs
            [z_rec,~,run_time_TQMCL(r,j,k)] =  tqmcl_sim(z0,tspan,H_2_L,H_2_R,V1,A,Z_Ref,para_set(k));
            [Q_TQMCL_sim_error(r,j,k),P_TQMCL_sim_error(r,j,k),U_TQMCL_sim_error(r,j,k),Ham_TQMCL(:,r,j,k)] = red_Ham_error(z_rec,...
                Q{k},P{k},U{k},norm_Q_para(k),norm_P_para(k),norm_U_para(k),A,para_set(k),delta_x);
        end
    end
    for k= para_ind
        %Simulate the PSD ROMs
        Z_Ref = [Q_Ref{k};P_Ref{k}];
        z0 = [V1'*(Q{k}(:,1)-Q_Ref{k});V1'*(P{k}(:,1)-P_Ref{k})];
        P_L_error(r,k) = norm((eye(N) -V1*V1')*(P{k}-P_Ref{k}*ones(1,size(P{k},2))),"fro")/norm_Q_para(k);
        Q_L_error(r,k) = norm((eye(N) -V1*V1')*(Q{k}-Q_Ref{k}*ones(1,size(Q{k},2))),"fro")/norm_Q_para(k);
        [Q_l_r,P_l_r] = pod_sim(A,tspan, z0, Z_Ref,V1,para_set(k));
        Q_l = V1*Q_l_r+Q_Ref{k}*ones(1,size(Q_l_r,2));
        P_l = V1*P_l_r+P_Ref{k}*ones(1,size(P_l_r,2));
        U_l = [Q_l;P_l];
        [Q_L_sim_error(r,k),P_L_sim_error(r,k),U_L_sim_error(r,k),Ham_L(:,r,k)] = red_Ham_error(U_l,...
            Q{k},P{k},U{k},norm_Q_para(k),norm_P_para(k),norm_U_para(k),A,para_set(k),delta_x);
    end
    [H_2] = regu_solver(kron_Q,res_Q,lambda);
    for k= para_ind
         %Simulate the QMCL
        [q0,p0,Q_QMCL_proj_error(r,k),P_QMCL_proj_error(r,k)] = proj_error(Q{k},P{k}, ...
                norm_Q_para(k), norm_P_para(k),Q_Ref{k},P_Ref{k},H_2,V1);
        z0 = [q0;q0];
        Z_Ref = [Q_Ref{k};P_Ref{k}];
        [z_rec,~,run_time_QMCL(r,j)] =  qmcl_sim(z0,tspan,H_2,V1,A,Z_Ref,para_set(k));
        [Q_QMCL_sim_error(r,k),P_QMCL_sim_error(r,k),U_QMCL_sim_error(r,k),Ham_QMCL(:,r,k)] = red_Ham_error(z_rec,...
            Q{k},P{k},U{k},norm_Q_para(k),norm_P_para(k),norm_U_para(k),A,para_set(k),delta_x);
    end
end
%%
%Save the data
save("Data/sim_error.mat",'U_L_sim_error','Q_L_sim_error','P_L_sim_error', ...
    'U_PSTQMCL_sim_error','Q_PSTQMCL_sim_error','P_PSTQMCL_sim_error',...
     "U_QMCL_sim_error","Q_QMCL_sim_error","P_QMCL_sim_error",...
     "U_TQMCL_sim_error","Q_TQMCL_sim_error","P_TQMCL_sim_error")
save("Data/pro_error.mat",'Q_L_error','P_L_error','Q_TQMCL_proj_error', ...
    'P_TQMCL_proj_error','Q_QMCL_proj_error','P_QMCL_proj_error');
save("Data/ham_value.mat",'Ham_ori','Ham_L','Ham_PSTQMCL','Ham_QMCL','Ham_TQMCL')
save("Data/setting.mat",'r_base','r_q_base','para_set')
%Function
%%
%function to compute the matrix V_bar. Here kron_Q_r represents the
%quadratic terms of q_r, res_Q represents the residual of the linear
%projection and lambda is the regularization parameter.
function [H_2] = regu_solver(kron_Q_r,res_Q,lambda)
    H_2 =[res_Q,zeros(size(res_Q,1),size(kron_Q_r,1))]/[kron_Q_r,sqrt(lambda)*eye(size(kron_Q_r,1))];
end

%function to compute the Hamiltonian energy
function ham = comp_ham(Q,P,A,para,delta_x)
    N = size(Q,1);
    for i = 1:1:size(Q,2)
        q_i = Q(:,i);
        p_i = P(:,i);
        x_i = [q_i;p_i];
        ham(i) = 0.5*x_i'*A*x_i+0.25*para*sum(q_i.^4);
    end
    ham = delta_x*ham;
end

%function to compute the projection error of the QMCL algorithm. This
%function is suitable for both the QMCL and the TQMCL method
function [q0,p0,Q_N_error,P_N_error] = proj_error(Q_snap,P_snap,norm_Q,norm_P,Q_ref,P_ref,H_2,V1)
    %Reproduce of Q
    p = size(Q_snap,2);
    Q_snap = Q_snap-Q_ref*ones(1,p);
    P_snap = P_snap-P_ref*ones(1,p);
    r = size(V1,2);
    ind = 1;
    for i = 1:r
        for j = 1:i
            i_ind(ind) = i;
            j_ind(ind) = j;
            ind = ind+1;
        end
    end
    Q_r = V1'*(Q_snap);
    %Nonlinear projection for Q
    Q_app = Q_ref*ones(1,p)+V1*Q_r+H_2*(Q_r(i_ind,:).*Q_r(j_ind,:));
    %Nonlinear projection error for Q
    Q_N_error = norm(Q_app-Q_snap-Q_ref*ones(1,p),"fro")/norm_Q;
    %%Reproduce of P
    P_V = V1'*(P_snap);
    %"Nonlinear projection" for P.
    P_H = P_snap-V1*V1'*(P_snap);
    M_V = H_2'*H_2;
    N_size = length(i_ind);
    mat_one = eye(r);
    N = size(V1,1);
    for i = 1:r
        M_2{i} =  kron_div_trun(mat_one(:,i),i_ind,j_ind);
    end
    for i = 1:p
        Q_i = Q_r(:,i);
        %Compute the Jacobian matrix for the Kronecker product
        L_q = kron_div_trun(Q_i,i_ind,j_ind);
        M1_Q = inv(eye(r)+L_q'*M_V*L_q);
        P_i = P_V(:,i)+L_q'*H_2'*P_H(:,i);%(V1+H_2*L_q)'*(P_snap_test(:,i)-P_ref_test);
        P_r(:,i) = P_i;
        P_app(:,i) = P_ref+V1*M1_Q'*P_i+H_2*L_q*M1_Q'*P_i;
    end
    %Nonlinear projection error
    P_N_error = norm(P_app-P_snap-P_ref*ones(1,p),"fro")/norm_P;
    %q0 and p0 must be obtained as they the initial values for the ROMs.
    q0 = Q_r(:,1);
    p0 = P_r(:,1);
end

%function to compute the nonredundant quadratic terms
function kron_x1 = col_kron(x1)
    r = size(x1,1);
    ind = 1;
    for i = 1:r
        for j = 1:i
            i_ind(ind) = i;
            j_ind(ind) = j;
            ind = ind+1;
        end
    end
    kron_x1 = x1(i_ind,:).*x1(j_ind,:);
end

%function to compute the Jacobian of the nonredundant quadratic terms.
function Kron_x1_div = kron_div_trun(x1,i_index,j_index)
    %Derivative of the "truncated" Kronecker product
    r = length(x1);
    I_r = speye(r);
    for i = 1:length(i_index)
        Kron_x1_div(i,:) = x1(i_index(i))*I_r(j_index(i),:)+x1(j_index(i))*I_r(i_index(i),:);
    end
end

%function to compute the reduction error and Hamiltonian energy of ROMs
function [Q_sim_error,P_sim_error,U_sim_error,Ham] = red_Ham_error(z_rec,...
    Q_snap,P_snap,U_snap,norm_Q,norm_P,norm_U,A,para,delta_x)
    N = size(z_rec,1)/2;
    Q_sim_error= norm(z_rec(1:N,:)-Q_snap,"fro")/norm_Q;
    P_sim_error= norm(z_rec(N+1:end,:)-P_snap,"fro")/norm_P;
    U_sim_error= norm(z_rec-U_snap,"fro")/norm_U;
    Ham = comp_ham(z_rec(1:N,:),z_rec(N+1:end,:),A,para,delta_x);
end
