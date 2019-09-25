clear all;close all;clc;

addr_prefix='/Users/alimarjaninejad/Documents/MATLAB/G2P_ActivationPlots/reinforcement_with_adaptation_9-20190710T185555Z-001/reinforcement_with_adaptation_9/';

file_name='matfile_experimentid_reinforcement_with_adaptation_9_attempt_';
attempts_no=1;
name_extention = '_adaptation_T';



load([addr_prefix,file_name,num2str(attempts_no),name_extention])
start_to_plot = 80;
end_the_plot = 160;

comet3(...
    run1_A_all_pred(start_to_plot:end_the_plot,1),...
    run1_A_all_pred(start_to_plot:end_the_plot,2),...
    run1_A_all_pred(start_to_plot:end_the_plot,3))

%plot(run1_A_all_pred(:,1))
%%
close all
plot_min=250;
plot_max=900;
t = -pi:pi/300:pi; 
axis([plot_min plot_max plot_min plot_max plot_min plot_max]) 
hold on
for ii=start_to_plot:1:end_the_plot
    
T=[t(ii) t(ii+1)]; 
plot3(...
    run1_A_all_pred(ii,1),...
    run1_A_all_pred(ii,2),...
    run1_A_all_pred(ii,3),'ro') 
plot3(...
    run1_A_all_pred(start_to_plot:ii,1),...
    run1_A_all_pred(start_to_plot:ii,2),...
    run1_A_all_pred(start_to_plot:ii,3),'b') 
pause(0.1) 
end 
hold off
%%
