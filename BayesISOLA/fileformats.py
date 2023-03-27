#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function for reading synthetic waveforms and response files in various fileformats.

"""

import re # RegExp
from scipy.io import FortranFile # fortran unformated (binary) files
import numpy as np
import os.path
from obspy import Trace, Stream, read_inventory
from obspy.core import read, AttribDict
from BayesISOLA.helpers import prefilter_data

def read_elemse_from_files(nr, path, stations, origin_time, samprate, npts_elemse, invert_displacement=False):
	"""
	Reads elementary seismograms file generated by any external code from SAC or miniSEED files.
	
	:param nr: number of receivers contained
	:type nr: integer
	:param path: path to the directory with 6 subdirectories 1..6 for elementary seismograms
	:type path: string
	:param stations: ``BayesISOLA.stations`` metadata of inverted stations
	:type stations: list of dictionaries
	:param origin_time: Event origin time in UTC
	:type origin_time: :class:`~obspy.core.utcdatetime.UTCDateTime`
	:param samprate: Resample elementary seismogram to this sampling rate
	:type samprate: float
	:param npts_elemse: number of points of inverted elementary seismograms
	:type npts_elemse: integer
	:param invert_displacement: if `True`, integrate elementary seismograms to displacement, otherwise keep it in velocity
	:type invert_displacement: bool, optional

	:return: elementary seismograms in form of list of lists of streams
	"""
	elemse_all=[]
	for r in range(nr):
		elemse=[]
		for j in range(6):
			f = {}
			dir_file = os.path.join(path, str(j+1), stations[r]['network']+'.'+stations[r]['code']+'.'+stations[r]['location']+'.'+stations[r]['channelcode'])
			DIR_FILE = os.path.join(path, 'GFs'+str(j+1), stations[r]['network']+'.'+stations[r]['code']+'.SE.MX')
			if os.path.isfile(dir_file+'Z.SAC'):
				st = Stream(traces=[read(dir_file+'Z.SAC')[0], read(dir_file+'N.SAC')[0], read(dir_file+'E.SAC')[0]])
			elif os.path.isfile(dir_file+'Z'):
				st = Stream(traces=[read(dir_file+'Z')[0], read(dir_file+'N')[0], read(dir_file+'E')[0]])
			else:
				st = Stream(traces=[read(DIR_FILE+'Z')[0], read(DIR_FILE+'N')[0], read(DIR_FILE+'E')[0]])
			st.trim(origin_time)
			if st[0].stats.sampling_rate != samprate:
				f = (int(npts_elemse/2)+1) * samprate / npts_elemse
				prefilter_data(st, f)
				st.interpolate(samprate)
			if invert_displacement:
				st.detrend('linear')
				st.integrate()
			elemse.append(st)
		elemse_all.append(elemse)
	return elemse_all
	
def read_elemse(nr, npts, filename, stations, invert_displacement=False):
	"""
	Reads elementary seismograms file generated by code ``Axitra``.
	
	:param nr: number of receivers contained
	:type nr: integer
	:param npts: number of points of each component
	:type npts: integer
	:param filename: path to the file
	:type filename: string
	:param stations: ``BayesISOLA.stations`` metadata of inverted stations
	:type stations: list of dictionaries
	:param invert_displacement: if `True`, integrate elementary seismograms to displacement, otherwise keep it in velocity
	:type invert_displacement: bool, optional

	:return: elementary seismograms in form of list of lists of streams
	"""
	ff = {}
	tr = Trace(data=np.empty(npts))
	tr.stats.npts = npts
	elemse_all=[]
	for r in range(nr):
		model = stations[r]['model']
		if model not in ff:
			if model:
				f = filename[0:filename.rfind('.')] + '-' + model + filename[filename.rfind('.'):]
			else:
				f = filename
			ff[model] = FortranFile(f)
		elemse=[]
		for j in range(6):
			f = {}
			f['N'] = tr.copy()
			f['E'] = tr.copy()
			f['Z'] = tr.copy()

			for i in range(npts):
				t,N,E,Z = ff[model].read_reals('f')
				f['N'].data[i] = N
				f['E'].data[i] = E
				f['Z'].data[i] = Z
				if i == 0:
					t1 = t
				elif i == npts-1:
					t2 = t
			tl = t2-t1
			samprate = (npts-1)/tl
			delta = tl/(npts-1)
			for comp in ['N', 'E', 'Z']:
				f[comp].stats.channel = comp;
				f[comp].stats.sampling_rate = samprate
				f[comp].stats.delta = delta
			st = Stream(traces=[f['Z'], f['N'], f['E']])
			if invert_displacement:
				st.detrend('linear')
				st.integrate()
			elemse.append(st)
		elemse_all.append(elemse)
	del ff
	return elemse_all

def attach_xml_paz(st, paz_file=None, inventory=None):
	"""
	Attaches an XML response file to a stream.
	
	:param st: Stream
	:type tr: :class:`~obspy.core.stream.Stream`
	:param paz_file: path to XML response file (if ``None``, just copy values from ``tr.stats.response`` to ``tr.stats.paz``)
	:type paz_file: string, optional
	:param inventory: inventory object
	:type inventory: :class:`~obspy.core.inventory.inventory.Inventory`, optional
	"""
	if paz_file:
		inv = read_inventory(paz_file)
		st.attach_response(inv)
	elif inventory:
		st.attach_response(inventory)		
	for tr in st:
		tr.stats.paz = tr.stats.response.get_paz()
		tr.stats.paz.gain = tr.stats.paz.normalization_factor # VERIFY
		tr.stats.paz.sensitivity           = tr.stats.response.instrument_sensitivity.value
		tr.stats.paz.sensitivity_frequency = tr.stats.response.instrument_sensitivity.frequency
		tr.stats.paz.sensitivity_unit      = tr.stats.response.instrument_sensitivity.input_units

def attach_ISOLA_paz(tr, paz_file):
	"""
	Attaches an ISOLA poles&zeros file to a trace as a paz AttribDict containing poles zeros and gain.
	
	:param tr: Trace
	:type tr: :class:`~obspy.core.trace.Trace`
	:param paz_file: path to pazfile in ISOLA format
	:type paz_file: string
	"""
	f = open(paz_file, 'r')
	f.readline() # comment line: A0
	A0 = float(f.readline())
	f.readline() # comment line: count-->m/sec
	count2ms = float(f.readline())
	f.readline() # comment line: zeros
	n_zeros = int(f.readline())
	zeros = []
	for i in range(n_zeros):
		line = f.readline()
		search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
		(r, i) = search.groups()
		zeros.append(complex(float(r), float(i)))
	f.readline() # comment line: poles
	n_poles = int(f.readline())
	poles = []
	for i in range(n_poles):
		line = f.readline()
		search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
		try:
			(r, i) = search.groups()
		except:
			print(line)
		poles.append(complex(float(r), float(i)))
	tr.stats.paz = AttribDict({
		'sensitivity': A0,
		'poles': poles,
		'gain': 1./count2ms,
		'zeros': zeros
		})
	f.close()

