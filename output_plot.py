#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import datetime as dt
import re, os

# use LaTeX fonts in the plot
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

import parameters

pd.options.display.float_format = '{:f}'.format

def read_file(filename, train_test):
	data = np.loadtxt(str('./'+train_test+'/results_'+train_test+'/'+filename+'.dat'), delimiter=' ', dtype=str)

	return data

def calc_confusion_matrix(combined, label_cf):
	data_table = np.empty((0,7), dtype=float)
	out = parameters.output_plot()

	conf_matrix = confusion_matrix(combined[:,4], combined[:,2], labels=label_cf)
	plot_confusion_matrix(conf_matrix, label_cf)

	print "CONFUSION MATRIX"
	print conf_matrix
	print conf_matrix.shape, type(conf_matrix)
	print " "

	# CALC TRUE POSITIVE, TRUE NEGATIVE, FALSE POSITIVE, FALSE NEGATIVE
	diag = np.diag(conf_matrix)
	for a in range (0, len(diag), 1):
		tp = diag[a]
		fp = conf_matrix[:,a].sum() - tp
		fn = conf_matrix[a,:].sum() - tp
		tn = conf_matrix.sum() - (tp + fp + fn)

		# CALC INDEX: PRECISION, RECALL, F_SCORE
		ppv = float(tp) / (tp + fp)
		tpr = float(tp) / (tp + fn)
		f_score = float(2 * tp) / (2*tp + fp + fn)
		
		data_element = np.array([tp, tn, fp, fn, ppv, tpr, f_score]).reshape(1,7)
		data_table = np.append(data_table, data_element, axis=0)

	index_table = plt.table(cellText=[['%.2f' % j for j in i] for i in data_table],
                            rowLabels=label_cf,
                            colLabels=out.label_index,
                            colWidths=[0.03 for x in out.label_index],
                            bbox=[-1.36, 0.0, 1.0, 1.0],
                            cellLoc='center',
                            loc='left')

	for a in range (4, 7, 1):
		for b in range (0, len(label_cf)+1, 1):
			index_table._cells[(b,a)].set_facecolor("orange")

	for (row, col), cell in index_table.get_celld().items():
		if (row == 0) or (col == 0):
			pass
			#index_table.rows[row].cells[column].paragraphs[0].runs[0].font.bold = True
			#cell.set_text_props(fontproperties=FontProperties(weight='bold'))

	index_table.auto_set_font_size(False)
	index_table.set_fontsize(9)
	index_table.scale(2, 2)
	index_table.set_fontsize(12)
	plt.subplots_adjust(left=0.2, top=0.8)

	text = plt.text(-13.00, +11.5, text_str, fontsize=12)#, verticalalignment='top')
	text.set_bbox(dict(facecolor='r', alpha=0.5, edgecolor='k'))

def plot_confusion_matrix(cm,
                      target_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=False):
	import itertools

	accuracy = np.trace(cm) / float(np.sum(cm))

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	plt.figure(str(title+'_'+target_names[-1][-3::]))#figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45, ha="right")
		plt.yticks(tick_marks, target_names)

	if normalize:
	    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.4f}".format(cm[i, j]),
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('TRUE LABELS')
	plt.xlabel('PREDICTED LABELS')
	text_str = 'ACCURATEZZA TOTALE ALGORITMO = {:0.4f}'.format(accuracy)

# MAIN
def main(power_1hz, train_test):
	print "PLOTTING..."

	decision = read_file("decision", train_test)
	ground_truth = read_file("ground_truth", train_test)
	combined = read_file("combined", train_test)
	decision_nan = read_file("decision_nan", train_test)
	uncombined = read_file("uncombined", train_test)

	global text_str
	text_str = " "

	graph, ax = plt.subplots()


	# Plot ground_truth
	for a in range(0, len(ground_truth), 1):
		if ground_truth[a,4].astype(float) == 1:
			ax.scatter(int(ground_truth[a,2].astype(float)), 0, color='g', marker='^', s=60)

		if ground_truth[a,4].astype(float) == 0:
			ax.scatter(int(ground_truth[a,2].astype(float)), 0, color='r', marker='^', s=60)

	# Plot traccia aggregato
	ax.plot(power_1hz[:,0].astype(int), power_1hz[:,1].astype(int), color='b', label="aggregato 1Hz")

	# Plot scatter decision
	for a in range (0, len(decision), 1):
		if decision[a,4] == 'ON':
			ax.scatter(decision[a,0].astype(float), decision[a,1].astype(float), c='g', s=60)
		elif decision[a,4] == 'OFF':
			ax.scatter(decision[a,0].astype(float), decision[a,1].astype(float), c='r', s=60)

#		label = str(decision[a,3] + decision[a,4])
#		ax.annotate(label, (decision[a,0].astype(float), decision[a,1].astype(float)))

	# Plot line combined
	for a in range (0, len(combined), 1):
		ax.plot ([combined[a,0].astype(float), combined[a,3].astype(float)], [combined[a,1].astype(float),0], 'g')

	'''
	# PRINT DISAGGREGATED FRIDGE
	file_fridge = './disaggregated_test/fridge.dat'
	fridge_consumption = upsample(file_fridge)
	ax.plot(fridge_consumption[:,0]*1000, fridge_consumption[:,1], c='m')
	print fridge_consumption[0,0]*1000, power_1hz[0,0]
	'''

	# Print label
	for a in range(0, len(combined), 1):
		label = str(combined[a,2] + "\n" + combined[a,4])
		ax.annotate(label, (combined[a,3].astype(float),0), xytext=(0, -10), textcoords='offset pixels', horizontalalignment='right', verticalalignment='top', rotation=45)

	text_str = str('EVENTI DEL GROUND_TRUTH NON DETECTATI = '+str(ground_truth.shape[0] - combined.shape[0])+'\n')

	plt.xlabel("TIME")
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax.set_ylabel("DISAGGREGATED")
	ax.set_ylim(-1500, 6000)
	ax.grid(True)

	# LEGEND OF PLOT
	ax.scatter([], [], color='g', marker='^', s=60, label='ground_truth_ON')
	ax.scatter([], [], color='r', marker='^', s=60, label='ground_truth_OFF')
	ax.scatter([], [], color='g', marker='o', s=60, label='events_ON')
	ax.scatter([], [], color='r', marker='o', s=60, label='events_OFF')
	ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., scatterpoints=1)

	print " "
	print "ERROR COMBINED"
	count = 0
	for a in range(0, len(combined), 1):
		if combined[a,2][-1] != combined[a,4][-1]:
			if combined[a,2] != 'NAN_NAN':
				count += 1
				print combined[a,2], int(combined[a,3].astype(float))/1000, combined[a,4], count
#			plt.arrow(combined[a,0].astype(float), combined[a,1].astype(float), 1, 200000, width=10000, head_width=50000, head_length=10000, fc='r', ec='r')
	print " "

	out = parameters.output_plot()
	# Confusion matrix
	calc_confusion_matrix(combined, out.label_confusion_matrix)
	
	# Confusion matrix NAN
	combined_nan = np.concatenate((combined,decision_nan), axis=0)
	calc_confusion_matrix(combined_nan, out.label_confusion_matrix_nan)

	'''
	print "CONFUSION MATRIX NAN"
	print conf_matrix_nan
	print conf_matrix_nan.shape, type(conf_matrix_nan)
	print " "
	
	print "COMBINED_NAN:"
	print combined_nan
	print combined_nan.shape, combined_nan.dtype, "|", combined_nan.nbytes, "bytes"
	print " "
	'''

	# Istogramma NAN
	for a in range (0, len(out.label_confusion_matrix), 1):
		correct_combined = np.empty((0,6), dtype=float)
		for b in range(0, len(combined), 1):
				if combined[b,4] == combined[b,2] and combined[b,2] == out.label_confusion_matrix[a]:
					correct_combined = np.append(correct_combined, combined[b,:].reshape(1,6), axis=0)
				
		nan_combined = np.empty((0,6), dtype=float)
		for b in range(0, len(decision_nan), 1):
				if decision_nan[b,2] == out.label_confusion_matrix[a]:
					nan_combined = np.append(nan_combined, decision_nan[b,:].reshape(1,6), axis=0)

		g_features = plt.figure(out.label_confusion_matrix[a])
		plt.hist(nan_combined[:,5].astype(float), 100, color='r', label='Events OFF')
		plt.hist(correct_combined[:,5].astype(float), 100, color='g', label='Events ON')
		plt.xlabel("Euclidean distance")
		plt.ylabel("Frequency")
		plt.legend(loc='best')

	plt.show(graph)
