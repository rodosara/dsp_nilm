#!/usr/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance as spatial

import parameters
import calc_features

def calc_features_evento(power_app, events):
	f = parameters.features()
	f_events = np.empty((0,f.number_features_event), dtype=float)

	pbar_features = tqdm(total=len(events))
	for a in range(0, len(events), 1):
		features_evento = calc_features.f_evento(power_app, int(events[a,0]), int(events[a,1]))
		f_events = np.append(f_events, features_evento, axis=0)

		pbar_features.update(1)

	pbar_features.close()
	print " "

	return f_events

def normalize_features(events, f_events):
	f_events_norm = np.empty((len(events),0), dtype=float)

	extreme = np.loadtxt('./normalize_value.dat', delimiter=' ', dtype=float)
	print "DATA:"
	print extreme
	print extreme.shape
	extreme_on = extreme[0:6,:]
	print "EXTREME_ON:"
	print extreme_on
	print extreme_on.shape
	extreme_off = extreme[6:12,:]
	print "EXTREME_OFF:"
	print extreme_off
	print extreme_off.shape
	print " "

	for a in range(0, f_events.shape[1], 1):
		element_norm = np.empty((0,1), dtype=float)

		for b in range(0, len(events), 1):
			if a == 5:
				value_norm = 0
			elif events[b,4] == 1:
				f_min = extreme_on[a,0]
				f_max = extreme_on[a,1]
				value_norm = (f_events[b,a] - f_min) / (f_max - f_min)
			elif events[b,4] == 0:
				f_min = extreme_off[a,0]
				f_max = extreme_off[a,1]
				value_norm = (f_events[b,a] - f_min) / (f_max - f_min)

			element_norm = np.append(element_norm, [[value_norm]], axis=0)
			#print "A:", a
			#print "FEATURE NORM:"
			#print feature_norm
			#print feature_norm.shape

		f_events_norm = np.append(f_events_norm, element_norm, axis=1)
		print "F_EVENTS_NORM:"
		print f_events_norm
		print f_events_norm.shape
		print " "
		print " "

	return f_events_norm
	
def normalize_database(database):
	database_norm = np.empty((len(database),0), dtype=float)

	extreme = np.loadtxt('./normalize_value.dat', delimiter=' ', dtype=float)
	extreme_on = extreme[0:6,:]
	extreme_off = extreme[6:12,:]

	for a in range(0, 6, 1):
		element_norm = np.empty((0,1), dtype=float)

		for b in range(0, len(database), 1):
			if a == 5:
				value_norm = 0
			elif database[b,1] == 'ON':
				f_min = extreme_on[a,0]
				f_max = extreme_on[a,1]
				database_to_norm = database[:,2:8]
				value_norm = (database_to_norm[b,a] - f_min) / (f_max - f_min)
			elif database[b,1] == 'OFF':
				f_min = extreme_off[a,0]
				f_max = extreme_off[a,1]
				database_to_norm = database[:,2:8]
				value_norm = (database_to_norm[b,a] - f_min) / (f_max - f_min)

			element_norm = np.append(element_norm, [[value_norm]], axis=0)
			#print "A:", a
			#print "FEATURE NORM:"
			#print feature_norm
			#print feature_norm.shape

		database_norm = np.append(database_norm, element_norm, axis=1)
		print "DATABASE_NORM:"
		#print database_norm
		print database_norm.shape
		print " "

	database[:,2:8] = database_norm

	return database
	
def save_norm_extreme(events, f_events):
	f_events_on = np.empty((0,6), dtype=float)
	f_events_off = np.empty((0,6), dtype=float)
	
	for a in range(0, len(events), 1):
		if events[a,4] == 1:
			f_events_on=np.append(f_events_on, f_events[a,:].reshape(1,6), axis=0)
		elif events[a,4] == 0:
			f_events_off=np.append(f_events_off, f_events[a,:].reshape(1,6), axis=0)
	
	print len(events), len(f_events_on), len(f_events_off), len(f_events_on) + len(f_events_off)

	f = open('./normalize_value.dat', 'a+b')
					
	for a in range(0, f_events_on.shape[1], 1):
		f_min = np.amin(f_events_on[:,a])
		f_max = np.amax(f_events_on[:,a])
		val = np.array([f_min, f_max]).reshape(1,2)
		np.savetxt(f, val, delimiter=' ', fmt='%s')

	for a in range(0, f_events_off.shape[1], 1):
		f_min = np.amin(f_events_off[:,a])
		f_max = np.amax(f_events_off[:,a])
		val = np.array([f_min, f_max]).reshape(1,2)
		np.savetxt(f, val, delimiter=' ', fmt='%s')
	
	f.close()

def features_comparison(event, f_events, database_on, database_off):
	distance_array = np.empty((0,3))

	# EVENTI DI ON --> DATABASE_ON
	if event[4] == 1:
		for a in range(0, len(database_on), 1):
			dist = np.array([[spatial.euclidean(f_events, database_on[a][2:7].astype(float)), database_on[a][0], database_on[a][1]]])
			#print "DIST: ", dist, dist.shape
			distance_array = np.append(distance_array, dist, axis=0)
		#print "DISTANCE_ARRAY:", distance_array, distance_array.shape
		matching = distance_array[np.argmin(distance_array[:,0].astype(float)),:].reshape(1,3)
		#print "MATCHING:", print matching, print matching.shape

		# CHECK THRESHOLD FOR NAN MATCHING
		threshold_nan_min = database_on[np.where(matching[0,1] == database_on)[0], 7]
		threshold_nan_max = database_on[np.where(matching[0,1] == database_on)[0], 8]
		if matching[0,0].astype(float) < threshold_nan_min or matching[0,0].astype(float) > threshold_nan_max:
			matching[0,1:3] = 'NAN'

		selection = np.concatenate((np.array(event[2:4]).reshape(1,2), matching), axis=1)
		#print "SELECTION:", selection, selection.shape

	# EVENTI DI OFF --> DATABASE_OFF
	elif event[4] == 0:
		for a in range(0, len(database_off), 1):
			dist = np.array([[spatial.euclidean(f_events, database_off[a][2:7].astype(float)), database_off[a][0], database_off[a][1]]])
			#print "DIST: ", dist, dist.shape
			distance_array = np.append(distance_array, dist, axis=0)
		#print "DISTANCE_ARRAY:", distance_array, distance_array.shape
		matching = distance_array[[np.argmin(distance_array[:,0].astype(float))]]
		#print "MATCHING:", print matching, print matching.shape
		
		# CHECK THRESHOLD FOR NAN MATCHING
		threshold_nan_min = database_off[np.where(matching[0,1] == database_off)[0], 7]
		threshold_nan_max = database_off[np.where(matching[0,1] == database_off)[0], 8]
		if matching[0,0].astype(float) < threshold_nan_min or matching[0,0].astype(float) > threshold_nan_max:
			matching[0,1:3] = 'NAN'

		selection = np.concatenate((np.array(event[2:4]).reshape(1,2), matching), axis=1)
		#print "SELECTION: ", selection, selection.shape

	return selection

def import_database():
	database_on = np.empty((0,9))
	database_off = np.empty((0,9))

	database = pd.read_csv('./database_appliances.dat', sep=' ', index_col=False, header=None)
	database = database.as_matrix()
	
	for a in range(0, len(database), 1):
		if database[a,1] == 'ON':
			database_on = np.append(database_on, [database[a]], axis=0)
		if database[a,1] == 'OFF':
			database_off = np.append(database_off, [database[a]], axis=0)
	
	return database_on, database_off

def debug(f_events, events, database_on, database_off, decision):
	print "DEBUG DECISION_ALGORITHM"

	np.set_printoptions(threshold='nan')
	print "F_EVENTS:"
	print f_events
	print f_events.shape, f_events.dtype, "|", f_events.nbytes, "bytes"
	print " "
	
	print "EVENTS:"
	print events
	print events.shape, events.dtype, "|", events.nbytes, "bytes"
	print " "
	
	print "DATABASE_ON:"
	print database_on
	print database_on.shape, database_on.dtype, "|", database_on.nbytes, "bytes"
	print " "
	
	print "DATABASE_OFF:"
	print database_off
	print database_off.shape, database_off.dtype, "|", database_off.nbytes, "bytes"
	print " "
	
	print "DECISION:"
	print decision
	print decision.shape, decision.dtype, "|", decision.nbytes, "bytes"
	print " "

# MAIN
def main(power_app, events):
	print "DECISION ALGORITHM"

	decision = np.empty((0,5))

	f_events = calc_features_evento(power_app, events)
	database_on, database_off = import_database()
	
	#save_norm_extreme(events, f_events_nn)

	'''
	# Normalizzazione valori features e database appliaces
	f_events = normalize_features(events, f_events)
	database_on = normalize_database(database_on)
	database_off = normalize_database(database_off)
	'''

	for a in range(0, len(events), 1):
		selection = features_comparison(events[a], f_events[a], database_on, database_off)
		decision = np.append(decision, selection, axis=0)

	#debug(f_events, events, database_on, database_off, decision)

	return decision
