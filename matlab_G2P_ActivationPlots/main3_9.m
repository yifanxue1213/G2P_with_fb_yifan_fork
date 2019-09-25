clear all;close all;clc;
%
start_attempt2plot=1;

addr_prefix='/Users/alimarjaninejad/Documents/MATLAB/G2P_ActivationPlots/reinforcement_with_adaptation_9-20190710T185555Z-001/reinforcement_with_adaptation_9/';
file_name='matfile_experimentid_reinforcement_with_adaptation_9_attempt_';
name_extention = '_adaptation_T';
attempt_no_to_plot = start_attempt2plot:start_attempt2plot+3;

kk=0;
for jj = [1:3 37:39]
    kk=kk+1;
    load([addr_prefix,file_name,num2str(jj),name_extention])
    Acts2plot{kk} = run1_A_all_pred;
end
start_to_plot = 80;
end_the_plot = 160;
%%
timestamps=start_to_plot:end_the_plot;
timestamps=timestamps-start_to_plot;
color1=ones(length(timestamps),3).*[150, 10, 40]/255;
color2=ones(length(timestamps),3).*[10, 200, 30]/255;

obj1= [Acts2plot{1}(start_to_plot:end_the_plot,:) timestamps'...
    1*ones(length(timestamps),1) color1];
obj2= [Acts2plot{2}(start_to_plot:end_the_plot,:) timestamps'...
    2*ones(length(timestamps),1) color1];
obj3= [Acts2plot{3}(start_to_plot:end_the_plot,:) timestamps'...
    3*ones(length(timestamps),1) color1];

obj4= [Acts2plot{4}(start_to_plot:end_the_plot,:) timestamps'...
    4*ones(length(timestamps),1) color2];
obj5= [Acts2plot{5}(start_to_plot:end_the_plot,:) timestamps'...
    5*ones(length(timestamps),1) color2];
obj6= [Acts2plot{6}(start_to_plot:end_the_plot,:) timestamps'...
    6*ones(length(timestamps),1) color2];

src = cat(1, obj1, obj2, obj3, obj4, obj5, obj6);
%
figure(1)
pause()
comet3n(src, 'speed', 1, 'taillength', 100, 'tailwidth', 2)