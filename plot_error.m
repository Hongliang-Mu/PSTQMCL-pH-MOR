close all
clear all
%%
%Hamiltonian energy
load('Data/Ham_value.mat',"Ham_ori",'Ham_QMCL','Ham_L','Ham_TQMCL','Ham_PSTQMCL');
load("Data/setting.mat",'r_base','r_q_base','para_set')
%Here the parameter is fixed and the size of r_q is 2*r.
ind_para = 6:1:6;
r_q_over_2 = 4; 
for r = r_base
    for j = ind_para
        Ham_QMCL_error(r,j) = mean(mean(abs(Ham_QMCL(:,r,j)- Ham_ori(:,j))));
        Ham_L_error(r,j) = mean(mean(abs(Ham_L(:,r,j)- Ham_ori(:,j))));
        Ham_TQMCL_error(r,j) = mean(mean(abs(Ham_TQMCL(:,r,r_q_over_2,j)- Ham_ori(:,j))));
        Ham_PSTQMCL_error(r,j) = mean(mean(abs(Ham_PSTQMCL(:,r,r_q_over_2,j)- Ham_ori(:,j))));
    end
end
figure()
p11 = semilogy(2*r_base,mean(Ham_L_error(r_base,ind_para),2),'b-o',LineWidth=4);
hold on
p12 = semilogy(2*r_base,mean(Ham_QMCL_error(r_base,ind_para),2),'m-s',LineWidth=4);
p13 = semilogy(2*r_base,mean(Ham_TQMCL_error(r_base,ind_para),2),'r-^',LineWidth=4);
p14 = semilogy(2*r_base,mean(Ham_PSTQMCL_error(r_base,ind_para),2),'g-*',LineWidth=4);
p11.MarkerSize = 20;
p12.MarkerSize = 20;
p13.MarkerSize = 20;
p14.MarkerSize = 20;
hold off
set(gca,'FontSize',18)
xlim([12,32])
xticks([12,16,20,24,28,32])
xticklabels({"12","16","20","24","28","32"})
xlabel("Reduced order 2r",'FontSize',28)
ylabel('error_{\rm AveHam}','FontSize',28)
legend('PSD','QMCL','TQMCL','PSTQMCL','FontSize',20)
title("Hamiltonian error ",'FontSize',32)
saveas(gcf,"Figures/Ham_error.fig")
saveas(gcf,"Figures/Ham_error.jpg")
%%
%Reduction error
load('Data/sim_error.mat',"Q_L_sim_error",'Q_PSTQMCL_sim_error','Q_TQMCL_sim_error','Q_QMCL_sim_error', ...
    "P_L_sim_error",'P_PSTQMCL_sim_error','P_TQMCL_sim_error','P_QMCL_sim_error')
figure()
p21 = semilogy(2*r_base,mean(Q_L_sim_error(r_base,ind_para),2),'b-o',LineWidth=4);
hold on 
p22 = semilogy(2*r_base,mean(Q_QMCL_sim_error(r_base,ind_para),2),'m-s',LineWidth=4);
p23 = semilogy(2*r_base,mean(Q_TQMCL_sim_error(r_base,r_q_over_2,ind_para),3),'r-^',LineWidth=4);
p24 = semilogy(2*r_base,mean(Q_PSTQMCL_sim_error(r_base,r_q_over_2,ind_para),3),'g-*',LineWidth=4);
p21.MarkerSize = 20;
p22.MarkerSize = 20;
p23.MarkerSize = 20;
p24.MarkerSize = 20;
set(gca,'FontSize',18)
xlim([12,32])
xticks([12,16,20,24,28,32])
xticklabels({"12","16","20","24","28","32"})
xlabel("Reduced order 2r",'FontSize',28)
ylabel('error_{\rm Red}','FontSize',28)
ylim([10^-2,10^0])
legend('PSD','QMCL','TQMCL','PSTQMCL','FontSize',20)
title("Reduction error",'FontSize',32)
hold off
saveas(gcf,"Figures/reduction_error.fig")
saveas(gcf,"Figures/reduction_error.jpg")