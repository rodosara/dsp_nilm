#!/usr/bin/python

import numpy as np

import dataset
import event_detection
import calc_features
import decision_algorithm
import combined_ground_truth
import output_plot

# MAIN
print " "
train_test = raw_input("VUOI LAVORARE CON DATASET DI TRAIN O TEST? (train|test) ")
print " "

print "SCEGLI UN OPZIONE (1|2):"
choice = eval(raw_input(" 1- SALVARE IL RISULTATO DELL'ALGORITMO \n 2- PLOTTARE A VIDEO IL RISULTATO DELL'ULTIMO SALVATAGGIO \n "))
print " "

filename = str('./'+train_test+'/mains_1hz_h2_'+train_test+'.dat')

power_1hz = dataset.uk_dale(filename, train_test)

if choice == 1:
	events = event_detection.main(power_1hz, 'event_detection')

	decision = decision_algorithm.main(power_1hz, events)

	folder = str('./'+train_test+'/appliances_events_'+train_test+'/')
	combined_ground_truth.main(power_1hz, decision, events, folder, train_test)

if choice == 2:
	output_plot.main(power_1hz, train_test)

print " "
