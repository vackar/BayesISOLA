#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import urllib.request as urllib
from obspy import read, Stream
#from obspy.clients.arclink import Client # The ArcLink protocol is deprecated

from fileformats import attach_xml_paz
from _input_seismo_files import check_a_station_present

def load_streams_fdsnws(self, hosts, t_before=90, t_after=360, save_to=None):
	"""
	Loads waveform from fdsnws server for stations listed in ``self.stations``.
	
	:param hosts: Host name(s) of the remote fdsnws server(s)
	:type hosts: string or list of strings
	:param t_before: length of the record before the event origin time
	:type t_before: float, optional
	:param t_after: length of the record after the event origin time
	:type t_after: float, optional
	:param save_to: save downloaded streams and responses to a specified directory
	:type save_to: string, optional
	"""
	t = self.event['t']
	i = 0
	if type(hosts) == str:
		hosts = [hosts]
	s = 'Loading data from fdsnws server(s):\n'
	for host in hosts:
		s += '\thost: {0:s}\n'.format(host)
	self.logtext['data'] = s
	self.log(s)
	while i < len(self.stations):
		sta = self.stations[i]
		st = None
		for host in hosts:
			try:
				url_data = "{host}dataselect/1/query?net={network}&sta={code}&loc={location}&cha={channelcode}*&starttime={start}&endtime={end}&format=miniseed".format(host=host, start=(t-t_before).isoformat(), end=(t+t_after).isoformat(), **sta)
				url_resp = "{host}station/1/query?network={network}&station={code}&level=response&starttime={start}&endtime={end}".format(host=host, start=(t-t_before).isoformat(), end=(t+t_after).isoformat(), **sta)
				if save_to:
					filename = os.path.join(save_to, "{network}.{code}.{location}.{channelcode}.mseed".format(**sta))
					urllib.urlretrieve(url_data, filename)
					st = read(filename)
					filename = os.path.join(save_to, "{network}.{code}.{location}.{channelcode}.xml".format(**sta))
					urllib.urlretrieve(url_resp, filename)
					attach_xml_paz(st, filename)
				else:
					st = read(url_data)
					inv = read_inventory(url_resp)
					st.attach_response(inv)
					attach_xml_paz(st)
			except:
				print(sta['network'], sta['code'], host, 'exception')
			else:
				if st:
					print(sta['network'], sta['code'], host, 'ok')
					break
		if not st:
			self.log('{0:s}:{1:s}: Downloading unsuccessful. Removing station from further processing.'.format(sta['network'], sta['code']))
			self.stations.remove(sta)
			self.create_station_index()
			continue
		if (st.__len__() != 3):
			self.log('{0:s}:{1:s}: Gap in data / wrong number of components. Removing station from further processing.'.format(sta['network'], sta['code']))
			self.stations.remove(sta)
			self.create_station_index()
			continue
		ch = {}
		for comp in range(3):
			ch[st[comp].stats.channel[2]] = st[comp]
		if (sorted(ch.keys()) != ['E', 'N', 'Z']):
			self.log('{0:s}:{1:s}: Unoriented components. Removing station from further processing.'.format(sta['network'], sta['code']))
			self.stations.remove(sta)
			self.create_station_index()
			continue
		st = Stream(traces=[ch['Z'], ch['N'], ch['E']])
		self.data_raw.append(st)
		sta['useZ'] = sta['useN'] = sta['useE'] = True
		if not st[0].stats.delta in self.data_deltas:
			self.data_deltas.append(st[0].stats.delta)
		i += 1
	self.data_are_corrected = False
	self.check_a_station_present()
	self.write_stations()

def load_streams_ArcLink(self, host, user='', t_before=90, t_after=360):
	"""
	Loads waveform from ArcLink server for stations listed in ``self.stations``.
	
	:param host: Host name of the remote ArcLink server
	:param user: The user name is used for identification with the ArcLink server. This entry is also used for usage statistics within the data centers, so please provide a meaningful user id such as your email address.
	:param t_before: length of the record before the event origin time
	:type t_before: float, optional
	:param t_after: length of the record after the event origin time
	:type t_after: float, optional
	"""
	self.logtext['data'] = s = 'Loading data from ArcLink server.\n\thost: {0:s}'.format(host)
	self.log('\n'+s)
	client = Client(host=host, user=user)
	t = self.event['t']
	i = 0
	while i < len(self.stations):
		sta = self.stations[i]
		try:
			st = client.getWaveform(sta['network'], sta['code'], sta['location'], sta['channelcode']+'*', t-t_before, t + t_after, metadata=True)
			#st.write('_'.join([sta['network'], sta['code'], sta['location'], sta['channelcode']]), 'MSEED') # DEBUG
		except:
			self.log('{0:s}:{1:s}: Downloading unsuccessful. Removing station from further processing.'.format(sta['network'], sta['code']))
			self.stations.remove(sta)
			self.create_station_index()
			continue
		if (st.__len__() != 3):
			self.log('{0:s}:{1:s}: Gap in data / wrong number of components. Removing station from further processing.'.format(sta['network'], sta['code']))
			self.stations.remove(sta)
			self.create_station_index()
			continue
		ch = {}
		for comp in range(3):
			ch[st[comp].stats.channel[2]] = st[comp]
		if (sorted(ch.keys()) != ['E', 'N', 'Z']):
			self.log('{0:s}:{1:s}: Unoriented components. Removing station from further processing.'.format(sta['network'], sta['code']))
			self.stations.remove(sta)
			self.create_station_index()
			continue
		st = Stream(traces=[ch['Z'], ch['N'], ch['E']])
		self.data_raw.append(st)
		sta['useN'] = sta['useE'] = sta['useZ'] = True
		if not st[0].stats.delta in self.data_deltas:
			self.data_deltas.append(st[0].stats.delta)
		i += 1
	self.data_are_corrected = False
	self.check_a_station_present()
	self.write_stations()

