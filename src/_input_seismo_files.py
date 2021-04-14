#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
from obspy import read, Trace, Stream

from fileformats import attach_ISOLA_paz, attach_xml_paz

def add_NEZ(self, filename, network, station, starttime, channelcode='LH', location='', accelerograph=False):
	"""
	Read stream from four column file (t, N, E, Z) (format used in ISOLA).
	Append the stream to ``self.data``.
	If its sampling is not contained in ``self.data_deltas``, add it there.
	
	:param filename: path to input file
	:type filename: string
	:param network: network code (for ``trace.stats``)
	:type network: string
	:param station: station code (for ``trace.stats``)
	:type station: string
	:param starttime: traces start time
	:type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
	:param channelcode: component names of all traces start with these letters (if channelcode is `LH`, component names will be `LHZ`, `LHN`, and `LHE`)
	:type channelcode: string, optional
	:param location: location code
	:type location: string, optional
	:param accelerograph: set ``True`` when the recorded quantity is acceleration
	:type accelerograph: bool, optional
	"""
	inp  = open(filename, 'r')
	lines = inp.readlines()
	inp.close()
	npts = len(lines)

	tr = Trace(data=np.empty(npts))
	tr.stats.network = network
	tr.stats.location = location
	tr.stats.station = station
	tr.stats.starttime = starttime
	tr.stats.npts = npts

	f = {}
	f['N'] = tr
	f['E'] = tr.copy()
	f['Z'] = tr.copy()
	for i in range(npts):
		a = lines[i].split()
		if i == 0:
			t1 = float(a[0])
		elif i == npts-1:
			t2 = float(a[0])
		f['N'].data[i] = float(a[1])
		f['E'].data[i] = float(a[2])
		f['Z'].data[i] = float(a[3])
	tl = t2-t1
	samprate = (npts-1)/tl
	delta = tl/(npts-1)
	for comp in ['N', 'E', 'Z']:
		f[comp].stats.channel = channelcode+comp;
		f[comp].stats.sampling_rate = samprate
		f[comp].stats.delta = delta
	st = Stream(traces=[f['Z'], f['N'], f['E']])
	self.data.append(st)
	if not delta in self.data_deltas:
		self.data_deltas.append(delta)
	# set flag "use in inversion" for all components
	stn = self.stations_index['_'.join([network, station, location, channelcode])]
	stn['useN'] = stn['useE'] = stn['useZ'] = True
	stn['accelerograph'] = accelerograph

def add_SAC(self, filename, filename2=None, filename3=None, accelerograph=False):
	"""
	Reads data from SAC. Can read either one SAC file, or three SAC files simultaneously to produce three component stream.
	Append the stream to ``self.data_raw``.
	If its sampling is not contained in ``self.data_deltas``, add it there.
	
	:param accelerograph: set ``True`` when the recorded quantity is acceleration
	:type accelerograph: bool, optional
	"""

	if filename3:
		st1 = read(filename); st2 = read(filename2); st3 = read(filename3)
		st = Stream(traces=[st1[0], st2[0], st3[0]])
		# set flag "use in inversion" for all components
		stn = self.stations_index['_'.join([st1[0].stats.network, st1[0].stats.station, st1[0].stats.location, st1[0].stats.channel[0:2]])]
		stn['useN'] = stn['useE'] = stn['useZ'] = True
		stn['accelerograph'] = accelerograph
	elif filename2:
		raise ValueError('Read either three files (Z, N, E components) or only one file (Z)')
		#st1 = read(filename); st2 = read(filename2)
		#st = Stream(traces=[st1[0], st2[0]])
	else:
		st1 = read(filename)
		st = Stream(traces=[st1[0]])
		stn = self.stations_index['_'.join([st1[0].stats.network, st1[0].stats.station, st1[0].stats.location, st1[0].stats.channel[0:2]])]
		stn['useZ'] = True
		stn['accelerograph'] = accelerograph
	self.data_raw.append(st)
	self.data_are_corrected = False
	if not st1[0].stats.delta in self.data_deltas:
		self.data_deltas.append(st1[0].stats.delta)

def add_NIED(self, filename, filename2=None, filename3=None, accelerograph=False):
	"""
	Reads data from NIED ASCII files. Can read either one file, or three NIED files simultaneously to produce three component stream.
	Append the stream to ``self.data_raw``.
	If its sampling is not contained in ``self.data_deltas``, add it there.

	:param accelerograph: set ``True`` when the recorded quantity is acceleration
	:type accelerograph: bool, optional
	"""
	
	if filename3:
		st1 = read(filename); st2 = read(filename2); st3 = read(filename3)
		st1[0].stats.network = self.stations[0]['network']
		st1[0].stats.location = self.stations[0]['location']
		st1[0].stats.channel = self.stations[0]['channelcode'] + st1[0].stats.channel[0:2]
		st2[0].stats.network = self.stations[0]['network']
		st2[0].stats.location = self.stations[0]['location']
		st2[0].stats.channel = self.stations[0]['channelcode'] + st2[0].stats.channel[0:2]
		st3[0].stats.network = self.stations[0]['network']
		st3[0].stats.location = self.stations[0]['location']
		st3[0].stats.channel = self.stations[0]['channelcode'] + st3[0].stats.channel[0:2]
		# sensitivity (convert to m/s^2)
		st1[0].data = st1[0].data * st1[0].stats.calib
		st1[0].stats.calib = 100
		st2[0].data = st2[0].data * st2[0].stats.calib
		st2[0].stats.calib = 100
		st3[0].data = st3[0].data * st3[0].stats.calib
		st3[0].stats.calib = 100
		st = Stream(traces=[st1[0], st2[0], st3[0]])
		# set flag "use in inversion" for all components
		stn = self.stations_index['_'.join([st1[0].stats.network, st1[0].stats.station, st1[0].stats.location, st1[0].stats.channel[0:2]])]
		stn['useN'] = stn['useE'] = stn['useZ'] = True
		stn['accelerograph'] = accelerograph
	elif filename2:
		raise ValueError('Read either three files (Z, N, E components) or only one file (Z)')
		#st1 = read(filename); st2 = read(filename2)
		#st = Stream(traces=[st1[0], st2[0]])
	else:
		st1 = read(filename)
		st1[0].stats.network = self.stations[0]['network']
		st1[0].stats.location = self.stations[0]['location']
		st1[0].stats.channel = self.stations[0]['channelcode'] + st1[0].stats.channel[0:2]
		st = Stream(traces=[st1[0]])
		# sensitivity (convert to m/s^2)
		st1[0].data = st1[0].data * st1[0].stats.calib
		st1[0].stats.calib = 1.0
		stn = self.stations_index['_'.join([st1[0].stats.network, st1[0].stats.station, st1[0].stats.location, st1[0].stats.channel[0:2]])]
		stn['useZ'] = True
		stn['accelerograph'] = accelerograph
	self.data_raw.append(st)
	self.data_are_corrected = False
	if not st1[0].stats.delta in self.data_deltas:
		self.data_deltas.append(st1[0].stats.delta)

def load_files(self, dir='.', prefix='', suffix='.sac', separator='.', pz_dir='.', pz_prefix='', pz_suffix='', pz_separator='.', xml_dir=None, xml_prefix='', xml_suffix='.xml', xml_separator='.'):
	"""
	2DO: UPDATE INFO
	
	Loads SAC files from specified directory. Filenames must correspond with sensor names in self.stations. Add traces to ``self.data_raw``. Attach to them instrument response from specified files. Uses functions :func:`add_SAC` and :func:`attach_ISOLA_paz`.
	
	:param dir: directory containing SAC files
	:type dir: string, optional
	:param prefix: data files prefix
	:type prefix: string, optional
	:param suffix: data files suffix; default '.sac'
	:type suffix: string, optional
	:param separator: data files separator between station code and component code etc.
	:type separator: string, optional
	:param pz_dir: directory containing poles & zeros files
	:type pz_dir: string, optional
	:param pz_prefix: poles & zeros files prefix
	:type pz_prefix: string, optional
	:param pz_suffix: poles & zeros files suffix
	:type pz_suffix: string, optional
	:param pz_separator: poles & zeros files separator
	:type pz_separator: string, optional
	
	It expectes the data files to be named like some of the following:
	
		* <dir>/<prefix><net>.<sta>.<loc>.<channel><suffix>
		* <dir>/<prefix><net>.<sta>.<channel><suffix>
		* <dir>/<prefix><sta>.<channel><suffix>
		
	where "." stands for the ``separator``.
	
	The instrument response files should be named:
	
		* <pz_dir>/<pz_preffix><net>.<sta>.<loc>.<channel><pz_suffix>
		* <pz_dir>/<pz_preffix><net>.<sta>.<channel><pz_suffix>
		* <pz_dir>/<pz_preffix><sta>.<channel><pz_suffix>
	
	where "." stands for the ``pz_separator``.
	
	.. note::
	
		It should work also with any other file formats (supported by ObsPy), where are the components stored in separate files
	"""
	self.logtext['data'] = s = '\nLoading data from files.\n\tdata dir: {0:s}\n\tp&z dir:  {1:s}'.format(dir, [pz_dir,xml_dir][bool(xml_dir)])
	self.log('\n'+s)
	loaded = len(self.data)+len(self.data_raw)
	#for i in range(self.nr):
	i = 0
	while i < len(self.stations):
		if i < loaded: # the data for the station already loaded from another source, it will be probably used rarely
			i += 1
			continue
		#if i >= self.nr: # some station removed inside the cycle
			#break
		sta = self.stations[i]['code']
		net = self.stations[i]['network']
		loc = self.stations[i]['location']
		ch  = self.stations[i]['channelcode']
		acc = self.stations[i]['accelerograph']
		# load data
		files = []
		for comp in ['Z', 'N', 'E']:
			names = [prefix + separator.join([sta,net,loc,ch+comp]) + suffix,
				prefix + separator.join([net,sta,loc,ch+comp]) + suffix,
				prefix + separator.join([sta,net,ch+comp]) + suffix,
				prefix + separator.join([net,sta,ch+comp]) + suffix,
				prefix + separator.join([sta,ch+comp]) + suffix,
				prefix + separator.join([sta,ch+comp,net]) + suffix]
			for name in names:
				#print(os.path.join(dir, name)) # DEBUG
				if os.path.isfile(os.path.join(dir, name)):
					files.append(os.path.join(dir, name))
					break
		if len(files) == 3:
			self.add_SAC(files[0], files[1], files[2], acc)
		else:
			self.stations.pop(i)
			self.create_station_index()
			self.log('Cannot find data file(s) for station {0:s}:{1:s}. Removing station from further processing.'.format(net, sta), printcopy=True)
			self.log('\tExpected file location: ' + os.path.join(dir, names[1]), printcopy=True)
			continue
		if xml_dir:
			# load poles and zeros - station XML
			names = [xml_prefix + xml_separator.join([net,sta,loc]) + xml_suffix,
				xml_prefix + xml_separator.join([sta,net,loc]) + xml_suffix,
				xml_prefix + xml_separator.join([sta,net]) + xml_suffix,
				xml_prefix + xml_separator.join([net,sta]) + xml_suffix,
				xml_prefix + xml_separator.join([sta]) + xml_suffix]
			for name in names:
				if os.path.isfile(os.path.join(xml_dir, name)):
					attach_xml_paz(self.data_raw[-1], os.path.join(xml_dir, name))
					break
			else: # poles&zeros file not found
				self.stations.pop(i)
				self.data_raw.pop()
				self.create_station_index()
				self.log('Cannot find xml response file for station {0:s}:{1:s}.  Removing station from further processing.'.format(net, sta), printcopy=True)
				self.log('\tExpected file location: ' + os.path.join(xml_dir, names[0]), printcopy=True)
				continue
			i += 1 # station not removed
		else:
			# load poles and zeros - ISOLA format
			for tr in self.data_raw[-1]:
				comp = tr.stats.channel[2]
				names = [pz_prefix + pz_separator.join([net,sta,loc,ch+comp]) + pz_suffix,
					pz_prefix + pz_separator.join([sta,net,loc,ch+comp]) + pz_suffix,
					pz_prefix + pz_separator.join([sta,net,ch+comp]) + pz_suffix,
					pz_prefix + pz_separator.join([sta,ch+comp]) + pz_suffix]
				for name in names:
					if os.path.isfile(os.path.join(pz_dir, name)):
						attach_ISOLA_paz(tr, os.path.join(pz_dir, name))
						break
				else: # poles&zeros file not found
					self.stations.pop(i)
					self.data_raw.pop()
					self.create_station_index()
					self.log('Cannot find poles and zeros file(s) for station {0:s}:{1:s}.  Removing station from further processing.'.format(net, sta), printcopy=True)
					self.log('\tExpected file location: ' + os.path.join(pz_dir, names[0]), printcopy=True)
					break
			else:
				i += 1 # station not removed
	self.check_a_station_present()

def load_NIED_files(self, dir='.', prefix='', dateString='', suffix='1', separator='.'):
	"""
	Loads NIED ASCII files from specified directory. Filenames must correspond with sensor names in self.stations. Add traces to ``self.data_raw``. Uses function :func:`add_SAC`
	
	:param dir: directory containing NIED ASCII files
	:type dir: string, optional
	:param prefix: data files prefix
	:type prefix: string, optional
	:param dateString: date + time string (e.g. YYMMDDhhmm)
	:type dateString: string, optional
	:param suffix: data files suffix; default '1'
	:type suffix: string, optional
	:param separator: data files separator between station code and component code; default '.'
	:type separator: string, optional
	
	It expectes the data files to be named like some of the following:
	
		* <dir>/<prefix><sta><dateString>.<channel><suffix>
		* <dir>/<prefix><sta><dateString>.<channel>
		
	where "." stands for the ``separator``.
	
	"""
	self.log('\nLoading data from files.\n\tdata dir: {0:s}'.format(dir))
	loaded = len(self.data)+len(self.data_raw)
	#for i in range(self.nr):
	i = 0
	while i < len(self.stations):
		if i < loaded: # the data for the station already loaded from another source, it will be probably used rarely
			i += 1
			continue
		sta = self.stations[i]['code']
		net = self.stations[i]['network']
		loc = self.stations[i]['location']
		ch  = self.stations[i]['channelcode']
		if len(sta) > 5: #stations with 6 char in name (K-net and KiK-net) are accelographs
			self.stations[i]['accelerograph'] = True
		acc = self.stations[i]['accelerograph']
		# load data
		files = []
		for comp in ['UD', 'NS', 'EW']:
			names = [prefix + sta + separator.join([dateString,comp]) + suffix,
				prefix + sta + separator.join([dateString,comp])]
			for name in names:
				#print(os.path.join(dir, name)) # DEBUG
				if os.path.isfile(os.path.join(dir, name)):
					files.append(os.path.join(dir, name))
					#print(os.path.join(dir, name)) # DEBUG
					break
		if len(files) == 3:
			self.add_NIED(files[0], files[1], files[2], acc)
			self.log('Staion {0:s}:{1:s} successfully loaded.'.format(net, sta), printcopy=True)
		else:
			self.stations.pop(i)
			self.create_station_index()
			self.log('Cannot find data file(s) for station {0:s}:{1:s}. Removing station from further processing.'.format(net, sta), printcopy=True)
			self.log('\tExpected file location: ' + os.path.join(dir, names[1]), printcopy=True)
			continue
		i += 1 # station not removed
	self.check_a_station_present()

def check_a_station_present(self):
	"""
	Checks whether at least one station is present, otherwise raises error.
	
	Called from :func:`load_streams_ArcLink` and :func:`load_files`.
	"""
	if not len(self.stations):
		self.log('No station present. Exiting...')
		raise ValueError('No station present.')
