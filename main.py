import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from warnings import simplefilter
from all_functions import *
from feedback_functions import *

simplefilter(action='ignore', category=FutureWarning)

# # [babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
# # model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)
# # cum_kinematics = babbling_kinematics
# # cum_activations = babbling_activations



# # pickle.dump([model,cum_kinematics, cum_activations],open("results/mlp_model.sav", 'wb'))

[model,cum_kinematics, cum_activations] = pickle.load(open("results/mlp_model.sav", 'rb')) # loading the model
np.random.seed(0)

P = np.array([10, 15])
I = np.array([2, 6])
trial_number = 50

experiments_switch=[1, 1, 1, 1, 1, 1, 1]
for ii in range(len(experiments_switch)):
	globals()["exp{}_average_error".format(ii+1)]=np.zeros([2,1])

if experiments_switch[0] ==1:
	features=np.ones(10,)
	cycle_durations = np.linspace(.1,10,trial_number)
	test1_no = cycle_durations.shape[0]
	exp1_average_error = np.zeros([2,test1_no]) # first row open-loop and second row close-loop
	#cycle length experiment
	for cycle_duration_in_seconds, ii in zip(cycle_durations, range(test1_no)):
		[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = cycle_duration_in_seconds, show=False)
		q0_filtered_10 = np.tile(q0_filtered,10)
		q1_filtered_10 = np.tile(q1_filtered,10)
		desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)
		exp1_average_error[0,ii], _, _ = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
		exp1_average_error[1,ii], _, _ = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]

if experiments_switch[1] ==1:
	test2_no = trial_number
	exp2_average_error = np.zeros([2,test2_no])
	for ii in range(test2_no):
		features = np.random.rand(10)*.8+.2
		[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 3.3, show=False)
		#import pdb; pdb.set_trace()
		q0_filtered_10 = np.tile(q0_filtered,10)
		q1_filtered_10 = np.tile(q1_filtered,10)
		desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)
		exp2_average_error[0,ii], _, _ = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
		exp2_average_error[1,ii], _, _ = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]
		#print("error_without: ", exp2_average_error[0,0], "error with: ", exp2_average_error[1,0])

if experiments_switch[2] ==1:
	test3_no = trial_number
	exp3_average_error = np.zeros([2,test3_no])
	for ii in range(test3_no):
		q0 = p2p_positions_gen_fcn(low=-np.pi/3, high=np.pi/3, number_of_positions=10, duration_of_each_position=3, timestep=.005)
		q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=0, number_of_positions=10, duration_of_each_position=3, timestep=.005)
		desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
		exp3_average_error[0,ii], _, _ = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, plot_outputs=False, Mj_render=False)
		exp3_average_error[1,ii], _, _ = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics,  P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]

if experiments_switch[3] ==1:
	test4_no = 1
	exp4_average_error = np.zeros([2,test4_no])
	q0 = p2p_positions_gen_fcn(low=np.pi/3, high=np.pi/3, number_of_positions=1, duration_of_each_position=1, timestep=.005)
	q0 = np.append(q0,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=14, timestep=.005))
	q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=-np.pi/2, number_of_positions=1, duration_of_each_position=1, timestep=.005)
	q1 = np.append(q1,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=14, timestep=.005))
	desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
	exp4_average_error[0,:], _, _ = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, model_ver=3, plot_outputs=False, Mj_render=False)

	q0 = p2p_positions_gen_fcn(low=np.pi/3, high=np.pi/3, number_of_positions=1, duration_of_each_position=1, timestep=.005)
	q0 = np.append(q0,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=4, timestep=.005))
	q1 = p2p_positions_gen_fcn(low=-np.pi/2, high=-np.pi/2, number_of_positions=1, duration_of_each_position=1, timestep=.005)
	q1 = np.append(q1,p2p_positions_gen_fcn(low=0, high=0, number_of_positions=1, duration_of_each_position=4, timestep=.005))
	desired_kinematics = positions_to_kinematics_fcn(q0, q1, timestep = 0.005)
	exp4_average_error[1,:], _, _ = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics,  P=P, I=I, model_ver=3, plot_outputs=False, Mj_render=False)

if experiments_switch[4] == 1:
	test5_no = trial_number
	exp5_average_error = np.zeros([2,test5_no])
	for ii in range(test5_no):
		##########################
		features = np.random.rand(10)*.8+.2
		[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 4, show=False)
		q0_filtered_10 = np.tile(q0_filtered,10)
		q1_filtered_10 = np.tile(q1_filtered,10)
		desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)
		exp5_average_error[0,ii], _, _ = openloop_run_fcn(model=model, desired_kinematics=desired_kinematics, model_ver=2, plot_outputs=False, Mj_render=False)
		exp5_average_error[1,ii], _, _ = closeloop_run_fcn(model=model, desired_kinematics=desired_kinematics, model_ver=2, P=P, I=I, plot_outputs=False, Mj_render=False) # K = [10, 15]

if experiments_switch[5] == 1: # everlearn ones
	[babbling_kinematics_1min, babbling_activations_1min] = babbling_fcn(simulation_minutes=1)
	model_1min = inverse_mapping_fcn(kinematics=babbling_kinematics_1min, activations=babbling_activations_1min)
	cum_kinematics_ol = deepcopy(babbling_kinematics_1min)
	cum_activations_ol = deepcopy(babbling_activations_1min)
	exp6_model_ol = deepcopy(model_1min)
	cum_kinematics_cl = deepcopy(babbling_kinematics_1min)
	cum_activations_cl = deepcopy(babbling_activations_1min)
	exp6_model_cl = deepcopy(model_1min)
	test6_no = trial_number
	exp6_average_error = np.zeros([2,test6_no])
	for ii in range(test6_no):
		features = np.ones(10,)
		print(features)
		[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 2, show=False) #1sec also fine
		q0_filtered_10 = np.tile(q0_filtered,10)
		q1_filtered_10 = np.tile(q1_filtered,10)
		desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)

		exp6_average_error[0,ii], real_attempt_kinematics_ol, real_attempt_activations_ol = openloop_run_fcn(model=exp6_model_ol, desired_kinematics=desired_kinematics, model_ver=0, plot_outputs=False, Mj_render=False)
		cum_kinematics_ol, cum_activations_ol = concatinate_data_fcn( cum_kinematics_ol, cum_activations_ol, real_attempt_kinematics_ol, real_attempt_activations_ol, throw_percentage = 0.20)
		exp6_model_ol = inverse_mapping_fcn(cum_kinematics_ol, cum_activations_ol, prior_model = exp6_model_ol)

		exp6_average_error[1,ii], real_attempt_kinematics_cl, real_attempt_activations_cl = closeloop_run_fcn(model=exp6_model_cl, desired_kinematics=desired_kinematics, model_ver=0, P=P, I=I, plot_outputs=False, Mj_render=False)
		cum_kinematics_cl, cum_activations_cl = concatinate_data_fcn( cum_kinematics_cl, cum_activations_cl, real_attempt_kinematics_cl, real_attempt_activations_cl, throw_percentage = 0.20)
		exp6_model_cl = inverse_mapping_fcn(cum_kinematics_cl, cum_activations_cl, prior_model = exp6_model_cl)

if experiments_switch[6] == 1: # everlearn random
	[babbling_kinematics_1min, babbling_activations_1min] = babbling_fcn(simulation_minutes=1)
	model_1min = inverse_mapping_fcn(kinematics=babbling_kinematics_1min, activations=babbling_activations_1min)
	cum_kinematics_ol = deepcopy(babbling_kinematics_1min)
	cum_activations_ol = deepcopy(babbling_activations_1min)
	exp7_model_ol = deepcopy(model_1min)
	cum_kinematics_cl = deepcopy(babbling_kinematics_1min)
	cum_activations_cl = deepcopy(babbling_activations_1min)
	exp7_model_cl = deepcopy(model_1min)
	test7_no = trial_number
	exp7_average_error = np.zeros([2,test7_no])
	for ii in range(test7_no):
		features = np.random.rand(10)*.8+.2
		print(features)
		[q0_filtered, q1_filtered]  = feat_to_positions_fcn(features, timestep=0.005, cycle_duration_in_seconds = 2, show=False) #1sec also fine
		q0_filtered_10 = np.tile(q0_filtered,10)
		q1_filtered_10 = np.tile(q1_filtered,10)
		desired_kinematics = positions_to_kinematics_fcn(q0_filtered_10, q1_filtered_10, timestep = 0.005)

		exp7_average_error[0,ii], real_attempt_kinematics_ol, real_attempt_activations_ol = openloop_run_fcn(model=exp7_model_ol, desired_kinematics=desired_kinematics, model_ver=0, plot_outputs=False, Mj_render=False)
		cum_kinematics_ol, cum_activations_ol = concatinate_data_fcn( cum_kinematics_ol, cum_activations_ol, real_attempt_kinematics_ol, real_attempt_activations_ol, throw_percentage = 0.20)
		exp7_model_ol = inverse_mapping_fcn(cum_kinematics_ol, cum_activations_ol, prior_model = exp6_model_ol)

		exp7_average_error[1,ii], real_attempt_kinematics_cl, real_attempt_activations_cl = closeloop_run_fcn(model=exp7_model_cl, desired_kinematics=desired_kinematics, model_ver=0, P=P, I=I, plot_outputs=False, Mj_render=False)
		cum_kinematics_cl, cum_activations_cl = concatinate_data_fcn( cum_kinematics_cl, cum_activations_cl, real_attempt_kinematics_cl, real_attempt_activations_cl, throw_percentage = 0.20)
		exp7_model_cl = inverse_mapping_fcn(cum_kinematics_cl, cum_activations_cl, prior_model = exp6_model_cl)


errors_all = [exp1_average_error, exp2_average_error, exp3_average_error, exp4_average_error, exp5_average_error, exp6_average_error, exp7_average_error]
pickle.dump([errors_all],open("results/feedback_errors_test.sav", 'wb')) # saving the results with only P
#[errors_all] = pickle.load(open("results/feedback_errors_test.sav", 'rb')) # loading the results with only P
#import pdb; pdb.set_trace()
plt.figure()
plt.plot(exp6_average_error[0,:])
plt.plot(exp6_average_error[1,:])
plt.show(block=True)
plt.figure()
plt.plot(exp7_average_error[0,:])
plt.plot(exp7_average_error[1,:])
plt.show(block=True)
#plot_comparison_figures_fcn(errors_all)

#import pdb; pdb.set_trace()
