#!/usr/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import parameters

np.set_printoptions(edgeitems=15, suppress=True)

# IMPORT PARAMETERS
def import_parameters(appliance_name):
	global e
	e = parameters.event(appliance_name)

# BACKGROUND LEVEL COMPUTATION
def background_level_computation(data_input):
	minn = np.min(data_input[:,1])
	bcz_max = minn + e.epsilon
    
	return bcz_max
    
# COMPUTE MEAN
def calc_mean(data_input, sample):
	#print "SAMPLE:", sample
	# PREVIOUS AVERAGE
	mean_prev = np.mean(data_input[sample-e.window:sample+1,1])
	#print "MEAN_PREV:", mean_prev, "START:", sample-e.window, "-->", data_input[sample-e.window,1], "SAMPLE:", sample, "-->", data_input[sample,1]
	#print data_input[sample-e.window:sample+1,1]

	# FOLLOWING AVERAGE
	mean_foll = np.mean(data_input[sample:sample+e.window+1,1])
	#print "MEAN_FOLL:", mean_foll, "SAMPLE:", sample, "-->", data_input[sample,1], "END:", sample+e.window, "-->", data_input[sample+e.window,1]
	#print data_input[sample:sample+e.window+1,1]
	#print " "
	
	return mean_prev, mean_foll

def debug(events, bcz_max):

	print "DETECTION EVENT DEBUG"
	print "BACKGROUND LEVEL COMPUTATION:", bcz_max, type(bcz_max)
	print " "
	print "EVENTS:"
	print events
	print events.shape, events.dtype, "|", events.nbytes, "bytes"
	print " "

def graph(data_input, events):

	graph, ax = plt.subplots()
	plt.xlabel("TIME")

	ax.plot(data_input[:,0], data_input[:,1], c='b')
	#ax.scatter(data_input[:,0], data_input[:,1], c='b', s=40)
	for a in range(0, len(events), 1):
		if events[a,4] == 1:
			ax.scatter(events[a,2], events[a,3], c='g', s=60)
			#ax.annotate(events[a,0], (events[a,2], events[a,3]), xytext=(20, -10), textcoords='offset pixels', rotation=45, ha="right")
			ax.scatter(data_input[events[a,1].astype(int),0], data_input[events[a,1].astype(int),1], c='g', s=60, marker='^')
		elif events[a,4] == 0:
			ax.scatter(events[a,2], events[a,3], c='r', s=60)
			ax.scatter(data_input[events[a,1].astype(int),0], data_input[events[a,1].astype(int),1], c='r', s=60, marker='^')

	#ax.set_xlim(data_input[0,0])
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax.grid(True)
	plt.ylabel("POWER 1HZ")
	
	# LEGEND OF PLOT
	ax.scatter([], [], color='g', marker='^', s=60, label='Event ON')
	ax.scatter([], [], color='r', marker='^', s=60, label='Event OFF')
	ax.scatter([], [], color='g', marker='o', s=60, label='Transient end ON')
	ax.scatter([], [], color='r', marker='o', s=60, label='Transient end OFF')
	plt.legend(loc='best')
	
	plt.show()

# MAIN
def main(data_input, appliance_name):
	print "DETECTION EVENT"

	import_parameters(appliance_name)

	events = np.zeros((0,5), dtype=int)

	c = e.window
	bar_pos = 0
	bcz_max = background_level_computation(data_input)

	pbar_database = tqdm(total=(len(data_input)-e.window))
	while c < (len(data_input)-e.window-1):
		bar_pos = c
		if data_input[c,1] > bcz_max:
			mean_prev, mean_foll = calc_mean(data_input, c)
			#events, c = detection(data_input, events, mean_prev, mean_foll, c)		

			# ON DETECTION
			if mean_prev < mean_foll and abs(mean_prev-mean_foll) > e.wavefront_threshold:
				start = c
				# WAVEFRONT_WAIT
				while mean_prev < mean_foll and abs(mean_prev-mean_foll) > e.wavefront_threshold:
					c += 1
					mean_prev, mean_foll = calc_mean(data_input, c)
#					print "ON WAVEFRONT WAIT:", c
#				print "MEAN_PREV:", mean_prev, "START:", c-e.window, "-->", data_input[c-e.window,1], "SAMPLE:", c, "-->", data_input[c,1]
#				print "MEAN_FOLL:", mean_foll, "SAMPLE:", c, "-->", data_input[c,1], "END:", c+e.window, "-->", data_input[c+e.window,1]
#				print " "
				if c != start:
					c -= 1
				end = c

				# SPIKE DETECTION
				if abs(data_input[start,1] - data_input[end,1]) > e.threshold_mean_on:
#					print " "
#					print "DIFFERENCE:", data_input[start,1] - data_input[end,1], "SAMPLE:", c, "START:", start, "END:", end

					event_on = np.concatenate(([start],[end],data_input[start],[1])).reshape(1,5)
#					print "SAVE EVENTO ON:", event_on
#					print " "
					events = np.append(events, event_on, axis=0)
					# EVENT INTERVAL ON
					c += e.event_interval_on
			
			# OFF DETECTION				
			elif  mean_prev > mean_foll and abs(mean_prev-mean_foll) > e.wavefront_threshold:			
				start = c
				# WAVEFRONT_WAIT
				while  mean_prev > mean_foll and abs(mean_prev-mean_foll) > e.wavefront_threshold:
					c += 1
					mean_prev, mean_foll = calc_mean(data_input, c)
#					print "OFF WAVEFRONT WAIT:", c
#				print "MEAN_PREV:", mean_prev, "START:", c-e.window, "-->", data_input[c-e.window,1], "SAMPLE:", c, "-->", data_input[c,1]
#				print "MEAN_FOLL:", mean_foll, "SAMPLE:", c, "-->", data_input[c,1], "END:", c+e.window, "-->", data_input[c+e.window,1]
#				print " "
				if c != start:
					c -= 1
				end = c

				# SPIKE DETECTION
				if abs(data_input[start,1] - data_input[end,1]) > e.threshold_mean_off:
#					print " "
#					print "DIFFERENCE:", data_input[start,1] - data_input[end,1], "SAMPLE:", c, "START:", start, "END:", end

					event_off = np.concatenate(([start],[end],data_input[start],[0])).reshape(1,5)
#					print "SAVE EVENTO OFF:", event_off
#					print " "
					events = np.append(events, event_off, axis=0)
					# EVENT INTERVAL OFF
					c += e.event_interval_off

		#else:
		c += 1
		#print "C+=1:", c
		pbar_database.update(c-bar_pos)
	pbar_database.close()

	#debug(events, bcz_max)
	#graph(data_input, events)

	print " "
	return events
