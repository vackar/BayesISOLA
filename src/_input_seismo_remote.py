#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from obspy.clients.arclink import Client

from _input_seismo_files import check_a_station_present

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
		#try: # DEBUG
			#st = read('_'.join([sta['network'], sta['code'], sta['location'], sta['channelcode']])) # DEBUG
		#except: # DEBUG
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

