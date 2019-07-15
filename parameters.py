#!/usr/bin/python

import numpy as np

# SCRIPT CONTENTE TUTTI I PARAMETRI UTILIZZATI NELL'ALGORITMO

# PARAMETRI CONVERSIONE DATI HOUSE
class House:
	number_of_ADC_steps = 0
	volts_per_adc_step = 0
	amps_per_adc_step = 0

def house(number_house):
	# PARAMETRI CONVERSIONE DATI HOUSE 1
	if number_house == "1":
		h = House()
		h.number_of_ADC_steps = 2147483648
		h.volts_per_adc_step = 0.000000190101491444
		h.amps_per_adc_step = 0.000000049224284384
		return h

	# PARAMETRI CONVERSIONE DATI HOUSE 2
	elif number_house == "2":
		h = House()
		h.number_of_ADC_steps = 2147483648
		h.volts_per_adc_step = 0.000000188296904357
		h.amps_per_adc_step = 0.0000000477518864497
		return h

# PARAMETRI PER DETECTION EVENT
class Event:
	window = 0 #SAMPLE
	epsilon = 0 #VA
	threshold_mean_on = 0 #VA
	threshold_mean_off = 0 #VA
	wavefront_threshold = 0 #VA
	event_interval = 0 #SAMPLE

def event(appliance_name):
	# PARAMETERS FOR THE DETECTION EVENT
	if appliance_name == 'event_detection':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 0 #VA
		e.threshold_mean_on = 60 #VA
		e.threshold_mean_off = 40 #VA
		e.wavefront_threshold = 5 #VA
		e.event_interval_on = 5 #SAMPLE
		e.event_interval_off = 0 #SAMPLE

	if appliance_name == 'kettle':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 20 # VA
		e.threshold_mean_on =1000 #VA
		e.threshold_mean_off =1000 #VA
		e.wavefront_threshold = 0 #VA
		e.event_interval_on = 120 #SAMPLE
		e.event_interval_off = 0 #SAMPLE

	if appliance_name == 'washing_machine':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 0 # VA
		e.threshold_mean_on = 500 #VA
		e.threshold_mean_off = 50 #VA
		e.wavefront_threshold = 0 #VA
		e.event_interval_on = 0 #SAMPLE
		e.event_interval_off = 0 #SAMPLE

	if appliance_name == 'dish_washer':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 0 # VA
		e.threshold_mean_on = 450 #VA
		e.threshold_mean_off = 100 #VA
		e.wavefront_threshold = 0 #VA
		e.event_interval_on = 0 #SAMPLE
		e.event_interval_off = 0 #SAMPLE

	if appliance_name == 'fridge':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 0 # VA
		e.threshold_mean_on = 30 #VA
		e.threshold_mean_off = 30 #VA
		e.wavefront_threshold = 0 #VA
		e.event_interval_on = 650 #SAMPLE
		e.event_interval_off = 0 #SAMPLE

	if appliance_name == 'microwave':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 0 # VA
		e.threshold_mean_on = 1000 #VA
		e.threshold_mean_off = 1000 #VA
		e.wavefront_threshold = 0 #VA
		e.event_interval_on = 35 #SAMPLE
		e.event_interval_off = 35 #SAMPLE

	if appliance_name == 'toaster':
		e = Event()

		e.window = 1 #SAMPLE (whitout the sample consider)
		e.epsilon = 0 # VA
		e.threshold_mean_on = 100 #VA
		e.threshold_mean_off = 100 #VA
		e.wavefront_threshold = 0 #VA
		e.event_interval_on = 10 #SAMPLE
		e.event_interval_off = 0 #SAMPLE

	return e

# PARAMETRI PER LE FEATURES
class Features:
	ss_interval = 0 #SAMPLES
	number_features_event = 0
	name_features_event = 0
	statistica = 0

def features():
	f = Features()

	# NAME OF FEATURES
	f.name_features_event = np.array(["1 - SSAC","2 - STEADY-STATE_SD", "3 - TSAC", "4 - TRANIENT-STATE_SD", "5 - RISE_TIME"])

	# NUMBER OF FEATURES BASED ON EVENT
	f.number_features_event = 5

	# STEADY STATE INTERVAL
	f.ss_interval = 30 #SAMPLES

	# ARRAY PER LA SCELTA TRA MEDIA E MODA PER LA CREAZIONE DEL DATABASE APPLIANCES
	f.statistica = np.loadtxt(str('./statistica_database.dat'), delimiter=' ', dtype=str)

	return f

# PARAMETRI PER L'OUTPUT PLOT
class Output_plot:
	combined_interval = 0 #SAMPLES
	label_confusion_matrix = 0 #TEXT
	label_confusion_matrix_nan = 0 #TEXT
	label_index = 0 #TEXT

def appliance(number):
		if number == 0:
			return "kettle"
		elif number == 1:
			return "dish_washer"
		elif number == 2:
			return "microwave"
		elif number == 3:
			return "fridge"
		elif number == 4:
			return "washing_machine"

def output_plot():
	out = Output_plot()

	# LABEL CONFUSION MATRIX
	out.label_confusion_matrix = ['kettle_ON', 'kettle_OFF', 'dish_washer_ON', 'dish_washer_OFF', 'microwave_ON', 'microwave_OFF', 'fridge_ON', 'fridge_OFF', 'washing_machine_ON', 'washing_machine_OFF']#,'NAN']
	out.label_confusion_matrix_nan = ['kettle_ON', 'kettle_OFF', 'dish_washer_ON', 'dish_washer_OFF', 'microwave_ON', 'microwave_OFF', 'fridge_ON', 'fridge_OFF', 'washing_machine_ON', 'washing_machine_OFF','NAN_NAN']
	out.label_index = ['TP', 'TN', 'FP', 'FN', 'PPV', 'TPR', 'F1']

	return out
