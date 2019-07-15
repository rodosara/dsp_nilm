#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm

import parameters

pd.options.display.float_format = '{:f}'.format
np.set_printoptions(edgeitems=20, suppress=True)

# IMPORTAZIONE WAVE FILE
def importazione(filename, train_test):
	if train_test == 'train':
		mains_1hz = np.genfromtxt(filename, delimiter=' ', dtype=float)#, max_rows=1000000)
	elif train_test == 'test':
		mains_1hz = np.genfromtxt(filename, delimiter=' ', dtype=float)

	#mains_1hz = np.genfromtxt('./disaggregated_train/washing_machine.dat', delimiter=' ', dtype=float)#, max_rows=100000)

	return mains_1hz

# DEBUG
def debug(mains_1hz, power_1hz):

	print "MAINS_1HZ"
	print mains_1hz
	print mains_1hz.shape, mains_1hz.dtype, "|", mains_1hz.nbytes, "bytes"
	print " "
	
	print "POWER 1HZ"
	print power_1hz
	print power_1hz.shape, power_1hz.dtype, "|", power_1hz.nbytes, "bytes"
	print " "
	print "START:", power_1hz[0,0]
	print "END:", power_1hz[-1,0]
	print " "

# GRAFICI
def graph(mains_1hz, power_1hz):
	print("STAMP A VIDEO GRAFICI")

	graph_mains, ax = plt.subplots(3, 1, sharex=True, sharey=False)
	plt.xlabel("TIME")

	ax[0].plot(mains_1hz[:,0], mains_1hz[:,1], c='b')
	ax[0].scatter(mains_1hz[:,0], mains_1hz[:,1], c='r', s=20)
	ax[0].set_xlim(mains_1hz[0,0])
	ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax[0].grid(True)
	plt.ylabel("ACTIVE POWER")
	
	ax[1].plot(mains_1hz[:,0], mains_1hz[:,2], c='b')
	ax[1].scatter(mains_1hz[:,0], mains_1hz[:,2], c='r', s=20)
	ax[1].set_xlim(mains_1hz[0,0])
	ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax[1].grid(True)
	plt.ylabel("APPARENT POWER")
		
	ax[2].plot(mains_1hz[:,0], mains_1hz[:,3], c='b')
	ax[2].scatter(mains_1hz[:,0], mains_1hz[:,3], c='r', s=20)
	ax[2].set_xlim(mains_1hz[0,0])
	ax[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax[2].grid(True)
	plt.ylabel("VOLTAGE")
	
	graph_power, ax = plt.subplots()
	plt.xlabel("TIME")

	ax.plot(power_1hz[:,0], power_1hz[:,1], c='b')
	ax.scatter(power_1hz[:,0], power_1hz[:,1], c='r', s=20)
	ax.set_xlim(power_1hz[0,0])
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
	ax.grid(True)
	plt.ylabel("APPARENT POWER")

	plt.show(graph)

	print " "

# MAIN
def uk_dale(filename, train_test):
	print "IMPORTAZIONE DATASET..."
	
	mains_1hz = importazione(filename, train_test)
	
	time_1hz = mains_1hz[:,0]*10
	time_1hz = time_1hz.astype(int)
	time_1hz = time_1hz*100
	
	power_1hz = np.column_stack((time_1hz, mains_1hz[:,1]))

	print "... DONE"
	print " "

	#debug(mains_1hz, power_1hz)
	#graph(mains_1hz, power_1hz)

	return power_1hz
