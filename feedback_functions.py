from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from numpy import matlib
from scipy import signal
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
#import pickle
import os
from copy import deepcopy
from mujoco_py.generated import const
from all_functions import *

def calculate_closeloop_inputkinematics(step_number, real_attempt_positions, desired_kinematics, q_error_cum, P, I, gradient_edge_order=1, timestep=.005):
	q_desired =  desired_kinematics[step_number, np.ix_([0,3])][0]
	q_dot_desired = desired_kinematics[step_number, np.ix_([1,4])][0]
	q_error = q_desired - real_attempt_positions[step_number-1,:]
	q_error_cum[step_number,:] = q_error
	#import pdb; pdb.set_trace()
	q_dot_in = q_dot_desired + np.array(P)*q_error + np.array(I)*q_error_cum.sum(axis=0)
	q_double_dot_in = [
		np.gradient(desired_kinematics[step_number-gradient_edge_order:step_number+1,1],edge_order=gradient_edge_order)[-1]/timestep,
		np.gradient(desired_kinematics[step_number-gradient_edge_order:step_number+1,4],edge_order=gradient_edge_order)[-1]/timestep]
		#desired_kinematics[step_number, np.ix_([2,5])][0]#
	desired_kinematics = [q_desired[0], q_dot_in[0], q_double_dot_in[0], q_desired[1], q_dot_in[1], q_double_dot_in[1]]
	return desired_kinematics, q_error_cum

def closeloop_run_fcn(model, desired_kinematics, P, I, model_ver=0, plot_outputs=True, Mj_render=False, timestep=.005):
	est_activations = estimate_activations_fcn(model, desired_kinematics)
	number_of_task_samples = desired_kinematics.shape[0]
	chassis_pos=np.zeros(number_of_task_samples,)
	input_kinematics = np.zeros(desired_kinematics.shape)
	real_attempt_positions = np.zeros([number_of_task_samples,2])
	real_attempt_activations = np.zeros([number_of_task_samples,3])
	q_error_cum = np.zeros([number_of_task_samples,2]) # sample error history

	Mj_model = load_model_from_path("./models/nmi_leg_w_chassis_v{}.xml".format(model_ver))
	sim = MjSim(Mj_model)

	if Mj_render:
		viewer = MjViewer(sim)
		# viewer.cam.fixedcamid += 1
		# viewer.cam.type = const.CAMERA_FIXED
	sim_state = sim.get_state()
	control_vector_length=sim.data.ctrl.__len__()
	print("control_vector_length: "+str(control_vector_length))

	sim.set_state(sim_state)
	gradient_edge_order = 1
	for ii in range(number_of_task_samples):
		if ii < gradient_edge_order:
			print(ii)
			input_kinematics[ii,:] = desired_kinematics[ii,:]
		else:
			[input_kinematics[ii,:], q_error_cum] = calculate_closeloop_inputkinematics(
				step_number=ii,
				real_attempt_positions=real_attempt_positions,
				desired_kinematics=desired_kinematics,
				q_error_cum=q_error_cum,
				P=P,
				I=I,
				gradient_edge_order=gradient_edge_order,
				timestep=timestep)
		est_activations[ii,:] = model.predict([input_kinematics[ii,:]])[0,:]
		sim.data.ctrl[:] = est_activations[ii,:]
		sim.step()
		chassis_pos[ii]=sim.data.get_geom_xpos("Chassis_frame")[0]
		current_positions_array = sim.data.qpos[-2:]
		real_attempt_positions[ii,:] = current_positions_array
		real_attempt_activations[ii,:] = sim.data.ctrl
		if Mj_render:
			viewer.render()
	real_attempt_kinematics = positions_to_kinematics_fcn(
		real_attempt_positions[:,0],
		real_attempt_positions[:,1],
		timestep=timestep)
	error0 = error_cal_fcn(desired_kinematics[:,0], real_attempt_kinematics[:,0])
	error1 = error_cal_fcn(desired_kinematics[:,3], real_attempt_kinematics[:,3])
	average_error = 0.5*(error0+error1)
	if plot_outputs:
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.plot(range(desired_kinematics.shape[0]), desired_kinematics[:,0], range(desired_kinematics.shape[0]), real_attempt_kinematics[:,0])
		plt.ylabel("q0 desired vs. simulated")
		plt.subplot(2, 1, 2)
		plt.plot(range(desired_kinematics.shape[0]), desired_kinematics[:,3], range(desired_kinematics.shape[0]), real_attempt_kinematics[:,3])
		plt.ylabel("q1  desired vs. simulated")
		plt.xlabel("Sample #")
		plt.show(block=True)
	return average_error

def openloop_run_fcn(model, desired_kinematics, model_ver=0, plot_outputs=False, Mj_render=False):
	est_activations = estimate_activations_fcn(model, desired_kinematics)
	[real_attempt_kinematics, real_attempt_activations, chassis_pos] = run_activations_fcn(est_activations, model_ver=model_ver, timestep=0.005, Mj_render=Mj_render)
	error0 = error_cal_fcn(desired_kinematics[:,0], real_attempt_kinematics[:,0])
	error1 = error_cal_fcn(desired_kinematics[:,3], real_attempt_kinematics[:,3])
	average_error = 0.5*(error0+error1)
	if plot_outputs:
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.plot(range(desired_kinematics.shape[0]), desired_kinematics[:,0], range(desired_kinematics.shape[0]), real_attempt_kinematics[:,0])
		plt.ylabel("q0 desired vs. simulated")
		plt.subplot(2, 1, 2)
		plt.plot(range(desired_kinematics.shape[0]), desired_kinematics[:,3], range(desired_kinematics.shape[0]), real_attempt_kinematics[:,3])
		plt.ylabel("q1  desired vs. simulated")
		plt.xlabel("Sample #")
		plt.show(block=True)
	return average_error

def p2p_positions_gen_fcn(low, high, number_of_positions, duration_of_each_position, timestep):
	sample_no_of_each_position = duration_of_each_position / timestep
	random_array = np.zeros(int(np.round(number_of_positions*sample_no_of_each_position)),)
	for ii in range(number_of_positions):
		random_value = ((high-low)*(np.random.rand(1)[0])) + low
		random_array_1position = np.repeat(random_value,sample_no_of_each_position)
		random_array[int(ii*sample_no_of_each_position):int((ii+1)*sample_no_of_each_position)] = random_array_1position
	return random_array

def plot_comparison_figures_fcn(errors_all):
	trial_number = errors_all[0].shape[1]
	plt.figure()
	plt.plot(np.linspace(.1,10,trial_number), errors_all[0][0,:], np.linspace(.1,10,trial_number), errors_all[0][1,:], marker='.')

	plt.ylim(0,.75)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[0][0,:].mean(),errors_all[0][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[0][1,:].mean(),errors_all[0][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error as a function of cycle period")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("cycle period (s)")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp1.png')
	plt.show()

	plt.figure()
	plt.plot(range(errors_all[1][0,:].shape[0]), errors_all[1][0,:], range(errors_all[1][0,:].shape[0]), errors_all[1][1,:], marker='.')
	plt.ylim(0,.75)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[1][0,:].mean(),errors_all[1][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[1][1,:].mean(),errors_all[1][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error value over a set of cyclical tasks")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp2.png')
	plt.show()

	plt.figure()
	plt.plot(range(errors_all[2][0,:].shape[0]), errors_all[2][0,:], range(errors_all[2][0,:].shape[0]), errors_all[2][1,:], marker='.')
	plt.ylim(0,.75)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[2][0,:].mean(),errors_all[2][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[2][1,:].mean(),errors_all[2][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error value over a set of point-to-point tasks")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp3.png')
	plt.show()
	# plotting mean error for each experiment
	plt.figure()
	plt.bar(range(3), [errors_all[0][0,:].mean(axis=0), errors_all[1][0,:].mean(axis=0), errors_all[2][0,:].mean(axis=0)])
	plt.bar(range(3), [errors_all[0][1,:].mean(axis=0), errors_all[1][1,:].mean(axis=0), errors_all[2][1,:].mean(axis=0)])
	plt.ylim(0,.5)
	plt.legend(["without feedback",'with feedback'])
	plt.ylabel("mean error (rads)")
	plt.xticks(range(3),('cycle period','cyclical','point-to-point'))
	plt.savefig('./results/P_I/mean_errors.png')
	plt.show()
	# errors_all = [exp2_average_error]
	# plt.figure()
	# plt.plot(range(errors_all[0][0,:].shape[0]), errors_all[0][0,:], range(errors_all[0][0,:].shape[0]), errors_all[0][1,:])
	# plt.show(block=True)

#import pdb; pdb.set_trace()
