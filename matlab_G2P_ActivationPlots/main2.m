clear all;close all;clc;
%%
start_attempt2plot=1;

addr_prefix='/Users/alimarjaninejad/Documents/MATLAB/G2P_ActivationPlots/reinforcement_with_adaptation_9-20190710T185555Z-001/reinforcement_with_adaptation_9/';
file_name='matfile_experimentid_reinforcement_with_adaptation_9_attempt_';
name_extention = '_adaptation_T';
attempt_no_to_plot = [start_attempt2plot:start_attempt2plot+3];
for jj = attempt_no_to_plot+1
load([addr_prefix,file_name,num2str(jj),name_extention])
Acts2plot{jj-start_attempt2plot} = run1_A_all_pred;
end
start_to_plot = 80;
end_the_plot = 160;
% 
% comet3(...
%     Acts2plot{jj}(start_to_plot:end_the_plot,1),...
%     Acts2plot{jj}(start_to_plot:end_the_plot,2),...
%     Acts2plot{jj}(start_to_plot:end_the_plot,3))

%plot(run1_A_all_pred(:,1))
%
%close all
plot_min=180;
plot_max=1000;
t = -pi:pi/300:pi;
figure(1)
hold on
colors_tmp = ['r','g','b'];
for ii=start_to_plot:1:end_the_plot
    T=[t(ii) t(ii+1)]; 
    for jj = 1:3
        axis([plot_min plot_max plot_min plot_max plot_min plot_max]) 
        plot3(...
            Acts2plot{jj}(start_to_plot:ii,1),...
            Acts2plot{jj}(start_to_plot:ii,2),...
            Acts2plot{jj}(start_to_plot:ii,3),'Color', 'k') 
        view(74,35)
        pause(0.01)
    end
end 
hold off
%%
