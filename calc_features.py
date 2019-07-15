#!/usr/bin/python

# SCRIPT DI DEFINIZIONE DELLE FEATURES UTILIZZATE NELLO SVILUPPO DELL'ALGORITMO

import numpy as np
import math

import parameters

# STEADY-STATE AVERAGE CONSUMPTION [VA]
def ss_average(data_input, pos_event, end_trans):
	global f1
	summ = 0

	for d in range(0, f.ss_interval, 1):
		summ += data_input[end_trans+d,1] - data_input[pos_event,1]
	f1 = summ / f.ss_interval

	return f1

# STEADY-STATE STANDARD DEVIATION CONSUMPTION [VA] -- BASE EVENTO
def ss_deviation(data_input, pos_event, end_trans):
	summ, mean = 0, 0

	for d in range(0, f.ss_interval, 1):
		square = (data_input[end_trans+d,1] - data_input[pos_event,1] + f1) ** 2
		summ += square
	mean = summ / f.ss_interval
	f2 = math.sqrt(mean)

	return f2

# TRANSIENT-STATE AVERAGE CONSUMPTION [VA] -- BASE EVENTO
def ts_average(data_input, pos_event, end_trans):
	global f3
	summ = 0
	ts_interval = end_trans - pos_event + 1

	for d in range(0, ts_interval, 1):
		summ += data_input[pos_event+d,1] - data_input[pos_event,1]
	f3 = summ / ts_interval

	return f3

# TRANSIENT-STATE STANDARD DEVIATION CONSUMPTION [VA] -- BASE EVENTO
def ts_deviation(data_input, pos_event, end_trans):
	summ, mean = 0, 0
	ts_interval = end_trans - pos_event + 1

	for d in range(0, ts_interval, 1):
		square = (data_input[pos_event+d,1] - data_input[pos_event,1] + f3) ** 2
		summ += square
	mean = summ / ts_interval
	f4 = math.sqrt(mean)

	return f4

# RISE TIME [SAMPLES] -- BASE EVENTO
def rise_time(data_input, pos_event, end_trans):
	ten_pcent = abs((10 * f1) / 100)
	nine_pcent = abs((90 * f1) / 100)

	pos = pos_event

	#print "FASE_1_PREC:", data_input[pos,1] - data_input[pos_event,1], "POS:", pos
	while abs(data_input[pos,1] - data_input[pos_event,1]) < ten_pcent or pos > len(data_input)-1:
		#print "FASE_1 -->", "POWER:", data_input[pos,1], "DIFF:", abs(data_input[pos,1] - data_input[pos_event,1]), "POS:", pos, "TEN_PCENT:", ten_pcent
		pos += 1

	start = pos

	#print "FASE_2_PREC:", data_input[pos,1] - data_input[pos_event,1], "POS:", pos
	while abs(data_input[pos,1] - data_input[pos_event,1]) < nine_pcent:# or pos > len(data_input)-1:
		#print "FASE_2 -->", "POWER:", data_input[pos,1], "DIFF:", abs(data_input[pos,1] - data_input[pos_event,1]), "POS:", pos, "NINE_PCENT:", nine_pcent
		pos += 1
	#print " "

	f5 = pos - start

	return f5

# CALCOLO FEATURES SU BASE EVENTO
def f_evento(data_input, pos_event, end_trans):
	global f
	f = parameters.features()
	features_evento = np.arange(5, dtype=float).reshape(1,5)

	features_evento[0,0] = ss_average(data_input, pos_event, end_trans)
	features_evento[0,1] = ss_deviation(data_input, pos_event, end_trans)
	features_evento[0,2] = ts_average(data_input, pos_event, end_trans)
	features_evento[0,3] = ts_deviation(data_input, pos_event, end_trans)
	features_evento[0,4] = rise_time(data_input, pos_event, end_trans)

	return features_evento
