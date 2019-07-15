#!/usr/bin/python

import pandas as pd, numpy as np, os, re, datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
import pickle as pkl
from scipy import stats

import calc_features
import parameters
import event_detection

pd.options.display.float_format = '{:f}'.format
#plt.rcParams['figure.figsize'] = 40, 20

def channels_name(folder):
	names = np.empty((0,1), dtype='S20')

	files = np.array(os.listdir(folder))

	for a in range(0, len(files), 1):
		if files[a][0] == '.':
			files = np.delete(files, a, axis=0)
			break

	for a in range(0, len(files), 1):
		filename, file_extension = os.path.splitext(files[a])
		names = np.append(names, filename)

	files = files.reshape(len(files),1)
	names = names.reshape(len(names),1)
	channels = np.column_stack((files, names))

	return channels

def upsample(filename):
	print "RESAMPLING....."
	
	data = pd.read_csv(filename, names=['date','consumption'], delim_whitespace=True, header=None, index_col=0)
	data.index = pd.to_datetime(data.index, unit='s')
	upsampled = data.resample('1S')
	appliance_consumption = upsampled.interpolate(method='linear')

	appliance_consumption = appliance_consumption.reset_index()	
	appliance_consumption['date'] = (appliance_consumption['date'] - dt.datetime(1970,1,1)).dt.total_seconds()
	
	if debug == 'yes':
		print "APPLIANCE_CONSUMPTION:"
		print appliance_consumption
		print " "

	appliance_consumption = appliance_consumption.as_matrix()
	
	print "...DONE!"
	print " "
	if debug == 'yes':
		print "D- APPLIANCE_CONSUMPTION", appliance_consumption

	return appliance_consumption
	
def import_events_file(appliance_consumption, appliance_name):
	print "######################### FOUNDED FILE EVENTS ###########################"
	print "Procedo con il caricamento..."

	events_on = np.zeros(0, dtype=int)
	events_off = np.zeros(0, dtype=int)

	data = np.loadtxt(str('./manual_event_detection/'+appliance_name+'_events_train.dat'), delimiter=' ', dtype=int)
	if debug == 'yes':
		print "CONTENUTO DEL FILE EVENTS:"
		print data
		print " "

	data = data.reshape(len(data),1)

	events = np.empty((0,5))
	for a in range(0, len(data), 1):
		pos_event = np.where(data[a,0]==appliance_consumption[:,0])[0][0]
		#pos_end = np.where(data[a,1]==appliance_consumption[:,0])[0][0]

		c = pos_event
		e = parameters.event(appliance_name)
		event_detection.import_parameters(appliance_name)
		mean_prev, mean_foll = event_detection.calc_mean(appliance_consumption, c)

		if a%2 == 0:
			while mean_prev < mean_foll and abs(mean_prev-mean_foll) > e.wavefront_threshold:
				c += 1
				mean_prev, mean_foll = event_detection.calc_mean(appliance_consumption, c)
			if c != pos_event:
				c -= 1
			pos_end = c

			event = np.array([pos_event, pos_end, data[a,0], appliance_consumption[pos_event,1], 1]).reshape(1,5)

		else:
			while  mean_prev > mean_foll and abs(mean_prev-mean_foll) > e.wavefront_threshold:
				c += 1
				mean_prev, mean_foll = event_detection.calc_mean(appliance_consumption, c)
			if c != pos_event:
				c -= 1
			pos_end = c

			event = np.array([pos_event, pos_end, data[a,0], appliance_consumption[pos_event,1], 0]).reshape(1,5)

		events = np.append(events, event, axis=0)

	return events
	
def graph_events(appliance_consumption, events):

	g_events = plt.figure(10)
	g_events.suptitle(appliance_name, fontsize=18)
	plt.plot(appliance_consumption[:,0], appliance_consumption[:,1], c='b')
	
	for a in range(0, len(events), 1):
		if events[a,4] == 1:
			plt.scatter(events[a,2], events[a,3], c='g', s=60)
			end = events[a,1].astype(int)
			plt.scatter(appliance_consumption[end,0], appliance_consumption[end,1], c='g', s=60, marker='^')

		elif events[a,4] == 0:
			plt.scatter(events[a,2], events[a,3], c='r', s=60)
			end = events[a,1].astype(int)
			plt.scatter(appliance_consumption[end,0], appliance_consumption[end,1], c='r', s=60, marker='^')
	
	plt.ylabel("APPLIANCE CONSUMPTION")
	plt.xlabel("TIME")
	ax10 = plt.axes()
	ax10.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	plt.grid(True)
	
	if save_graph == 'yes':
		graph_name = str('events_'+appliance_name+'.png')
		save_graphs(g_events, str(graph_name))

def features(appliance_consumption, events, appliance_name):
	appliance_features_on = np.empty((0,5))
	appliance_features_off = np.empty((0,5))

	pbar_features = tqdm(total=len(events))
	for a in range(0, len(events), 1):
		if events[a,4] == 1:
			features_evento = calc_features.f_evento(appliance_consumption, int(events[a,0]), int(events[a,1]))
			appliance_features_on = np.append(appliance_features_on, features_evento, axis=0)
		pbar_features.update(1)

		if events[a,4] == 0:
			features_evento = calc_features.f_evento(appliance_consumption, int(events[a,0]), int(events[a,1]))
			appliance_features_off = np.append(appliance_features_off, features_evento, axis=0)
		pbar_features.update(1)

	pbar_features.close()
	print " "

	pos = np.where(appliance_name == f.statistica)

	# CAL FEATURES ON WITH STATISTIC
	features_stats_on = np.empty((1,0), dtype=float)
	for a in range(0, f.number_features_event, 1):

		# MEAN
		if f.statistica[pos[0]][0,2+a] == 'mean':
			print "ON", f.statistica[pos[0]][0,2+a]
			f_stats = np.array([np.mean(appliance_features_on[:,a], axis=0)])

		# MODE
		elif f.statistica[pos[0]][0,2+a] == 'mode':
			print "ON", f.statistica[pos[0]][0,2+a]
			feat_on_freq, feat_on_interval = np.histogram(appliance_features_on[:,a], 100, density=False)
			#print " "
			#print "FEAT_ON_FREQ:", feat_on_freq, feat_on_freq.shape
			#print "FEAT_ON_INTERVAL:", feat_on_interval, feat_on_interval.shape
			mode_on = np.argmax(feat_on_freq)
			#print "MODE_ON:", mode_on
			f_stats = float(feat_on_interval[mode_on+1] + feat_on_interval[mode_on]) / 2
			#print "F_STATS:", f_stats

		f_stats = np.array([f_stats]).reshape(1,1)
		features_stats_on = np.append(features_stats_on, f_stats, axis=1)

	# CAL FEATURES OFF WITH STATISTIC
	features_stats_off = np.empty((1,0), dtype=float)
	for a in range(0, f.number_features_event, 1):

		# MEAN
		if f.statistica[pos[0]][1,2+a] == 'mean':
			print "OFF", f.statistica[pos[0]][1,2+a]
			f_stats = np.array([np.mean(appliance_features_off[:,a], axis=0)])

		# MODE
		elif f.statistica[pos[0]][1,2+a] == 'mode':
			print "OFF", f.statistica[pos[0]][1,2+a]
			feat_off_freq, feat_off_interval = np.histogram(appliance_features_off[:,a], 100, density=False)
			#print " "
			#print "FEAT_OFF_FREQ:", feat_off_freq, feat_off_freq.shape
			#print "FEAT_OFF_INTERVAL:", feat_off_interval, feat_off_interval.shape
			mode_off = np.argmax(feat_off_freq)
			#print "MODE_OFF:", mode_off
			f_stats = float(feat_off_interval[mode_off+1] + feat_off_interval[mode_off]) / 2
			#print "F_STATS:", f_stats

		f_stats = np.array([f_stats]).reshape(1,1)
		features_stats_off = np.append(features_stats_off, f_stats, axis=1)

	return appliance_features_on, features_stats_on, appliance_features_off, features_stats_off

def graph_features(appliance_features_on, features_stats_on, appliance_features_off, features_stats_off):
	for a in range(0, 5, 1):
		g_features = plt.figure(f.name_features_event[a])
		plt.hist(appliance_features_on[:,a], 100, color='g')
		plt.axvline(features_stats_on[0,a], color='g', linestyle='--', linewidth=4)
		plt.hist(appliance_features_off[:,a], 100, color='r')
		plt.axvline(features_stats_off[0,a], color='r', linestyle='--', linewidth=4)
		plt.ylabel("FREQUENCY")
		plt.xlabel("BINS")
		plt.suptitle(f.name_features_event[a])
		if save_graph == 'yes':
			graph_name = str('features_'+f.name_features_event[a]+'.png')
			save_graphs(g_features, graph_name)

def save_graphs(graph, graph_name):
	print " "
	print "SALVO IL GRAFICO IN .png..."
	main_folder = '/home/rod/'+ appliance_name + '/'
	if not os.path.exists(main_folder):
		os.makedirs(main_folder)
	#pkl.dump(graph, open(str(main_folder+'graph/'+graph_name), 'wb'))
	plt.savefig(main_folder+graph_name, dpi=100)
	print "...DONE!"
	print " "
	
def save_database(database_appliances):
	f = open('./database_appliances.dat', 'a+b')
	np.savetxt(f, database_appliances, delimiter=' ', fmt='%s')
	f.close()
	
	print "########################### DATABASE SAVED CORRECTLY ###########################"

# MAIN
global f
f = parameters.features()

# DEBUG
global debug
debug = raw_input("VUOI EFFETTUARE DEBUG? (yes|no) ")
print " "

# SHOW GRAPHS
global view_graph
view_graph = raw_input("VUOI GRAFICARE GLI EVENTI E LE FEATURES? (yes|no) ")
print " "

# SALVATAGGIO GRAPHS IN PNG E PICKLE FORMAT
global save_graph
save_graph = raw_input("VUOI SALVARE I GRAFICI IN .png? (yes|no) ")
print " "

# SALVATAGGIO EVENTS
global save_events
save_events = raw_input("VUOI SALVARE IL GLI EVENTI? (yes|no) ")
print " "

# SALVATAGGIO DATABASE
global database
database = raw_input("VUOI SALVARE IL DATABASE? (yes|no) ")
print " "

folder = str("./train/disaggregated_train/")

channels = channels_name(folder)
print "CHANNELS:"
print channels
print " "
num_channel = eval(raw_input("CHE CHANNEL VUOI ANALIZZARE? [0:4] "))
print " "

global appliance_name
appliance_name = channels[num_channel,1]

print "#########################", channels[num_channel], "###########################"
appliance_consumption = upsample(folder + channels[num_channel,0])

if channels[num_channel,1] == "washing_machine" or channels[num_channel,1] == "dish_washer":
	events = import_events_file(appliance_consumption, channels[num_channel,1])

else:
	events = event_detection.main(appliance_consumption, channels[num_channel,1])
	
if debug == 'yes':
	print "EVENTS:"
	print events
	print events.shape, events.dtype, "|", events.nbytes, "bytes"
	print " "
	
if save_events == 'yes':
	save_events = np.empty((0,6), dtype='float')
	for a in range (0, len(events), 1):
		save_event = np.concatenate((events[a,:].reshape(1,5), [[num_channel]]), axis=1).reshape(1,6)
		save_events = np.append(save_events, save_event, axis=0)
	np.savetxt('./appliances_events_train/'+channels[num_channel,1]+'.dat', save_events, delimiter=' ', fmt='%s')
	print " "
	print "########################### EVENTS SAVED CORRECTLY ###########################"
	print " "

if view_graph == 'yes':
	graph_events(appliance_consumption, events)

print "########################## CALCOLO FEATURES ############################"
appliance_features_on, features_stats_on, appliance_features_off, features_stats_off = features(appliance_consumption, events, channels[num_channel,1])

if debug == 'yes':
	print "FEATURES_ON:"
	print appliance_features_on
	print appliance_features_on.shape, appliance_features_on.dtype, "|", appliance_features_on.nbytes, "bytes"
	print " "
print "FEATURES_STATS_ON:", features_stats_on
print features_stats_on.shape, features_stats_on.dtype, "|", features_stats_on.nbytes, "bytes"
print "--------------------------------------------------------------------------------"
if debug == 'yes':
	print "FEATURES_OFF:"
	print appliance_features_off
	print appliance_features_off.shape, appliance_features_off.dtype, "|", appliance_features_off.nbytes, "bytes"
	print " "
print "FEATURES_STATS_OFF:", features_stats_off
print features_stats_off.shape, features_stats_off.dtype, "|", features_stats_off.nbytes, "bytes"
print " "

if view_graph == 'yes':
	graph_features(appliance_features_on, features_stats_on, appliance_features_off, features_stats_off)

print "######################### CREATION DATABASE_APPLIANCES ###########################"
database_appliances = np.hstack(([[channels[num_channel,1]]], [['ON']], features_stats_on))
database_appliances = np.append(database_appliances, np.hstack(([[channels[num_channel,1]]], [['OFF']], features_stats_off)), axis=0)

print "DATABASE_APPLIANCES:"
print database_appliances
print database_appliances.shape, database_appliances.dtype, "|", database_appliances.nbytes, "bytes"
print " "

if database == 'yes':
	save_database(database_appliances)

if view_graph == 'yes':
	mng = plt.get_current_fig_manager()
	#mng.window.showMaximized()
	plt.show()
