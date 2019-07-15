#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import datetime as dt
import re, os

import parameters

pd.options.display.float_format = '{:f}'.format

def channels_name(folder):
	names = np.empty((0,1), dtype='S20')

	files = np.array(os.listdir(folder))

	for a in range(0, len(files), 1):
		if files[a][0] == '.':
			files = np.delete(files, a, axis=0)

	for a in range(0, len(files), 1):
		filename, file_extension = os.path.splitext(files[a])
		names = np.append(names, filename)

	files = files.reshape(len(files),1)
	names = names.reshape(len(names),1)
	channels = np.column_stack((files, names))

	return channels

def calc_ground_truth(folder, power_1hz):
	channels = channels_name(folder)

	ground_truth = np.empty((0,6), dtype=float)

	#pbar1 = tqdm(total=len(channels))
	for a in range(0, len(channels), 1):
		appliance_events = np.loadtxt(folder+channels[a,0], dtype=float, delimiter=' ')

		# Selezione intervallo
		start = np.abs((appliance_events[:,2]*1000).astype(int)-power_1hz[0,0]).argmin()
		if (appliance_events[start,2]*1000).astype(int) < power_1hz[0,0]:
			start += 1
		#print "START:", start
		end = np.abs((appliance_events[:,2]*1000).astype(int)-power_1hz[-1,0]).argmin()
		if (appliance_events[end,2]*1000).astype(int) > power_1hz[-1,0]:
			end -= 1
		#print "END:", end
		appliance_events = appliance_events[start:end+1,:]

		#pbar1.update(a)
		ground_truth = np.append(ground_truth, appliance_events, axis=0)

	#pbar1.close()

	ground_truth = ground_truth[np.argsort(ground_truth[:,0])]
	ground_truth[:,2] *= 1000
	ground_truth = ground_truth.astype(int)

	return ground_truth

def label_calc(gt):
	if gt[4] == 1:
		label_gt = str(parameters.appliance(gt[5])+'_ON')

	elif gt[4] == 0:
		label_gt = str(parameters.appliance(gt[5])+'_OFF')

	return label_gt

def debug(ground_truth, combined, decision_nan, uncombined):
	print " "
	print "COMBINED GROUND_TRUTH DEBUG"

	print "COMBINED:"
	print combined
	print combined.shape, combined.dtype, "|", combined.nbytes, "bytes"
	print " "

	print "GROUND_TRUTH:"
	print ground_truth
	print ground_truth.shape, ground_truth.dtype, "|", ground_truth.nbytes, "bytes"
	print " "
	
	print "DECISION_NAN:"
	print decision_nan
	print decision_nan.shape, decision_nan.dtype, "|", decision_nan.nbytes, "bytes"
	print " "

	count = 0
	for a in range(0, len(combined), 1):
		if combined[a,2][-1] != combined[a,4][-1]:
			count += 1
			print int(combined[a,3].astype(float))/1000, combined[a,4], count
	
	print " "
	print "DIFFERENCE GROUND_TRUTH-COMBINED:", ground_truth.shape[0] - combined.shape[0]
	print "UNCOMBINED:"
	print uncombined
	print uncombined.shape, uncombined.dtype, "|", uncombined.nbytes, "bytes"
	
def save_array(array, name, train_test):
	folder = str('./'+train_test+'/results_'+train_test+'/')
	if not os.path.exists(folder):
		os.makedirs(folder)

	filename = str(folder+name+'.dat')
	np.savetxt(filename, array, delimiter=' ', fmt='%s')

# MAIN
def main(power_1hz, decision, events, folder, train_test):
	print "COMBINED GOUND_TRUTH..."

	combined_buffer = np.empty((0,7), dtype=float)
	combined = np.empty((0,6), dtype=float)
	decision_nan = np.empty((0,6), dtype=float)

	graph, ax = plt.subplots()
	plt.title("OUTPUT PLOT")

	ground_truth = calc_ground_truth(folder, power_1hz)

	a = 0
	while a < len(decision):
		pos = abs(decision[a,0].astype(float)-(ground_truth[:,2]).astype(int)).argmin()
		label_decision = str(decision[a,3]+"_"+decision[a,4])
		label_gt = label_calc(ground_truth[pos,:])
	
		if a+1 == len(decision) or pos != abs(decision[a+1,0].astype(float)-(ground_truth[:,2])).argmin():
			combined_element = np.concatenate(([int(decision[a,0].astype(float))],[decision[a,1].astype(float)],[label_decision],[ground_truth[pos,2]],[label_gt],[decision[a,2].astype(float)])).reshape(1,6)
			combined = np.append(combined, combined_element, axis=0)

		else:
			# Rende la funzione iniettiva
			while a+1 != len(decision) and pos == abs(decision[a+1,0].astype(float)-(ground_truth[:,2])).argmin():
				min_buffer = np.amin(abs(decision[a,0].astype(float)-(ground_truth[:,2])))
				buffer_element = np.concatenate(([int(decision[a,0].astype(float))],[decision[a,1].astype(float)],[label_decision],[ground_truth[pos,2]],[label_gt],[decision[a,2].astype(float)],[min_buffer])).reshape(1,7)
				combined_buffer = np.append(combined_buffer, buffer_element, axis=0)

				a += 1

				pos = abs(decision[a,0].astype(float)-(ground_truth[:,2])).argmin()
				label_decision = str(decision[a,3]+"_"+decision[a,4])
				label_gt = label_calc(ground_truth[pos,:])

			# Considera anche l'ultimo elemento del buffer
			if a+1 != len(decision) and pos != abs(decision[a+1,0].astype(float)-(ground_truth[:,2])).argmin() and pos == abs(decision[a-1,0].astype(float)-(ground_truth[:,2])).argmin():
				min_buffer = np.amin(abs(decision[a,0].astype(float)-(ground_truth[:,2])))
				buffer_element = np.concatenate(([int(decision[a,0].astype(float))],[decision[a,1].astype(float)],[label_decision],[ground_truth[pos,2]],[label_gt],[decision[a,2].astype(float)],[min_buffer])).reshape(1,7)
				combined_buffer = np.append(combined_buffer, buffer_element, axis=0)

			pos_buffer = (combined_buffer[:,6].astype(float)).argmin()
			combined_element = combined_buffer[pos_buffer,0:6].reshape(1,6)
			combined = np.append(combined, combined_element, axis=0)

			decision_nan = np.append(decision_nan, np.delete(combined_buffer[:,0:6], pos_buffer, axis=0), axis=0)
			decision_nan[:,4] = "NAN_NAN"
			combined_buffer = np.empty((0,7), dtype=float)

		a += 1

	uncombined = ground_truth
	for a in range(0, len(combined), 1):
		pos = np.where(combined[a,3].astype(float) == uncombined[:,2])
		uncombined = np.delete(uncombined, pos[0][0], axis=0)

	save_array(decision, "decision", train_test)
	save_array(ground_truth, "ground_truth", train_test)
	save_array(combined, "combined", train_test)
	save_array(decision_nan, "decision_nan", train_test)
	save_array(uncombined, "uncombined", train_test)
	
	#debug(ground_truth, combined, decision_nan, uncombined)
	print "... DONE"
