from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from numpy import matlib
from scipy import signal
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.lines as mlines
#import pickle
import os
from copy import deepcopy
from mujoco_py.generated import const
from all_functions import *

def calculate_closeloop_inputkinematics(step_number, real_attempt_positions, desired_kinematics, q_error_cum, P, I, delay_timesteps, gradient_edge_order=1, timestep=.005):
	q_desired =  desired_kinematics[step_number, np.ix_([0,3])][0]
	q_dot_desired = desired_kinematics[step_number, np.ix_([1,4])][0]
	q_error = q_desired - real_attempt_positions[step_number-1-delay_timesteps,:]
	q_error_cum[step_number,:] = q_error
	#import pdb; pdb.set_trace()
	q_dot_in = q_dot_desired + np.array(P)*q_error + np.array(I)*(q_error_cum.sum(axis=0)*timestep)
	q_double_dot_in = [
		np.gradient(desired_kinematics[step_number-gradient_edge_order:step_number+1,1],edge_order=gradient_edge_order)[-1]/timestep,
		np.gradient(desired_kinematics[step_number-gradient_edge_order:step_number+1,4],edge_order=gradient_edge_order)[-1]/timestep]
		#desired_kinematics[step_number, np.ix_([2,5])][0]#
	input_kinematics = [q_desired[0], q_dot_in[0], q_double_dot_in[0], q_desired[1], q_dot_in[1], q_double_dot_in[1]]
	# There are multiple ways of calculating q_double_dot:
		# 1- d(v_input(last_step)-v_desired(last_step))/dt
		# 2- d(v_input(current)-v_observed(last_step))/dt
		# 3- d(v_input(current)-v_input(last_step))/dt

		# 1 nad 2 will have jumps and therefore can cause huge values while differentiated. 3 on the otherhand will not make much physical sense
		# and will not be helping the system reach its goal since it is it is disregardin considering the plant or the goal velocity alltogether.
		# We observed that using the acceleration values coming from the feedforward system (desired q_double_dot) works better than the alternatives
		# however, acceleration value can be set differently for other specific goals. For example, for velocity tracking (the main focus of this 
		# project is position tracking), acceleration can be set corresponding to the velocity error to compensate for this error. 
	return input_kinematics, q_error_cum

def closeloop_run_fcn(model, desired_kinematics, P, I, delay_timesteps=0, model_ver=0, plot_outputs=True, Mj_render=False, timestep=.005):
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
		if ii < max(gradient_edge_order, delay_timesteps+1):
			#print(ii)
			input_kinematics[ii,:] = desired_kinematics[ii,:]
		else:
			[input_kinematics[ii,:], q_error_cum] = calculate_closeloop_inputkinematics(
				step_number=ii,
				real_attempt_positions=real_attempt_positions,
				desired_kinematics=desired_kinematics,
				q_error_cum=q_error_cum,
				P=P,
				I=I,
				delay_timesteps=delay_timesteps,
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
	return average_error, real_attempt_kinematics, real_attempt_activations

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
	return average_error, real_attempt_kinematics, real_attempt_activations

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
	# plt 1: vs cycle period
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
	#plt 2: 50 cyclical
	plt.figure()
	plt.plot(range(errors_all[1][0,:].shape[0]), errors_all[1][0,:], range(errors_all[1][0,:].shape[0]), errors_all[1][1,:], marker='.')
	plt.ylim(0,.75)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[1][0,:].mean(),errors_all[1][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[1][1,:].mean(),errors_all[1][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error values over a set of cyclical tasks")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp2.png')
	plt.show()
	#plt 3: 50 p2p
	plt.figure()
	plt.plot(range(errors_all[2][0,:].shape[0]), errors_all[2][0,:], range(errors_all[2][0,:].shape[0]), errors_all[2][1,:], marker='.')
	plt.ylim(0,.75)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[2][0,:].mean(),errors_all[2][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[2][1,:].mean(),errors_all[2][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error values over a set of point-to-point tasks")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp3.png')
	plt.show()
	# plt 4: compare all
	plt.figure()
	plt.bar(range(6), [errors_all[1][0,:].mean(axis=0), errors_all[2][0,:].mean(axis=0), errors_all[0][0,:].mean(axis=0), 
			errors_all[4][0,:].mean(axis=0), errors_all[5][0,:].mean(axis=0), errors_all[6][0,:].mean(axis=0)])
	plt.bar(range(6), [errors_all[1][1,:].mean(axis=0), errors_all[2][1,:].mean(axis=0), errors_all[0][1,:].mean(axis=0),
			errors_all[4][1,:].mean(axis=0), errors_all[5][1,:].mean(axis=0), errors_all[6][1,:].mean(axis=0)])
	plt.ylim(0,.8)
	plt.legend(["without feedback",'with feedback'])
	plt.ylabel("mean error (rads)")
	plt.xticks(range(6),('cyclical','point-to-point', 'cycle period', 'with contact', 'refinements same', 'refinements random'), rotation=10)
	plt.savefig('./results/P_I/mean_errors.png')
	plt.show()
	# errors_all = [exp2_average_error]
	# plt.figure()
	# plt.plot(range(errors_all[0][0,:].shape[0]), errors_all[0][0,:], range(errors_all[0][0,:].shape[0]), errors_all[0][1,:])
	# plt.show(block=True)

	#plt 5: with contact
	plt.figure()
	plt.plot(range(errors_all[4][0,:].shape[0]), errors_all[4][0,:], range(errors_all[4][0,:].shape[0]), errors_all[4][1,:], marker='.')
	plt.ylim(0,1)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[4][0,:].mean(),errors_all[4][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[4][1,:].mean(),errors_all[4][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error values when intense contact dynamics are introduced")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp5.png')
	plt.show()

	#plt 6: ever learn ones
	plt.figure()
	plt.plot(range(errors_all[5][0,:].shape[0]), errors_all[5][0,:], range(errors_all[5][0,:].shape[0]), errors_all[5][1,:], marker='.')
	plt.ylim(0,.5)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[5][0,:].mean(),errors_all[5][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[5][1,:].mean(),errors_all[5][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error values as a function of refinements (same desired movements)")
	plt.legend(["without feedback",'with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp6.png')
	plt.show()

	#plt 7: ever learn random
	plt.figure()
	plt.plot(range(errors_all[6][0,:].shape[0]), errors_all[6][0,:], range(errors_all[6][0,:].shape[0]), errors_all[6][1,:], marker='.')
	plt.ylim(0,.5)
	ax = plt.gca()
	xmin, xmax = ax.get_xbound()
	mean_error_wo = mlines.Line2D([xmin,xmax], [errors_all[6][0,:].mean(),errors_all[6][0,:].mean()],color='C0', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wo)
	mean_error_wf = mlines.Line2D([xmin,xmax], [errors_all[6][1,:].mean(),errors_all[6][1,:].mean()],color='C1', linestyle='--', alpha=.7)
	ax.add_line(mean_error_wf)
	plt.title("Error values as a function of refinements (different desired movements)")
	plt.legend(['without feedback','with feedback'])
	plt.xlabel("trial #")
	plt.ylabel("error (rads)")
	plt.savefig('./results/P_I/exp7.png')
	plt.show()

	#plt 8: delay
	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111, projection='3d')

	# Grab some test data.
	X, Y, Z = axes3d.get_test_data(0.05)
	exp8_average_error = errors_all[7]
	X_1 = np.linspace(0,20*5,11)
	X = np.tile(X_1, [exp8_average_error.shape[1], 1]).transpose()
	Y_1 = np.arange(1,exp8_average_error.shape[1]+1)
	Y = np.tile(Y_1, [11,1])
	Z = exp8_average_error[1:,:]
	# Plot a basic wireframe.

	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=.9)
	Z_ol_1 = exp8_average_error[0,:]
	Z_ol = np.tile(Z_ol_1,[11,1])
	ax.plot_wireframe(X, Y, Z_ol, rstride=5, cstride=10, color="lightcoral", alpha=.7)
	ax.view_init(elev=21., azim=-114.)
	ax.set_xlabel('delays (ms)')
	ax.set_ylabel('trial #')
	ax.set_zlabel('mean error (rads)')
	plt.title('Error for a set of cyclical trials as a function of delay')
	plt.savefig('./results/P_I/exp8.png')
	plt.show()

#import pdb; pdb.set_trace()
