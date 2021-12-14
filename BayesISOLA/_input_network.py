#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#import psycopg2 as pg
from math import sin,cos,radians
from pyproj import Geod
import obspy
from obspy.geodetics.base import gps2dist_azimuth

def read_network_info_DB(self, db, host, port=-1, user=None, password=None, min_distance=None, max_distance=None):
	"""
	Reads station coordinates from `SeisComp3` database.
	Calculate their distances and azimuthes using WGS84 elipsoid.
	Create data structure ``self.stations``. Sorts it according to station epicentral distance.
	
	:param db: the database name
	:param host: database host address
	:param port: connection port number (defaults to 5432 if not provided)
	:parem user: user name used to authenticate
	:param password: password used to authenticate
	:param min_distance: minimal epicentral distance in meters
	:param min_distance: float or None
	:param min_distance: maximal epicentral distance in meters
	:param max_distance: float or None
	
	Default value for ``min_distance`` is 2*self.rupture_length. Default value for ``max_distance`` is :math:`1000 \cdot 2^{2M}`.
	"""
	self.logtext['network'] = s = 'Network info: '+host
	self.log('\n'+s)
	conn = pg.connect(database=db, host=host, port=port, user=user, password=password)
	mag = self.event['mag']
	if min_distance==None:
		min_distance = 2*self.rupture_length
	if max_distance==None:
		max_distance = 1000 * 2**(mag*2.)
	query = """SELECT
		network.m_code AS net,
		station.m_code AS sta,
		sensorlocation.m_code AS loc,
		substring(stream.m_code FROM 1 FOR 2) AS channels,
		station.m_longitude AS lon,
		station.m_latitude AS lat
		FROM network
		INNER JOIN station ON station._parent_oid = network._oid
		INNER JOIN sensorlocation ON sensorlocation._parent_oid = station._oid
		INNER JOIN stream ON stream._parent_oid = sensorlocation._oid
		WHERE
		st_distance(ST_GeographyFromText('SRID=4326;POINT({lon:s} {lat:s})'), 
		ST_GeographyFromText('SRID=4326;POINT(' || station.m_longitude || ' ' || station.m_latitude ||')') ) < {dist:14.8f}
		AND
		st_distance(ST_GeographyFromText('SRID=4326;POINT({lon:s} {lat:s})'), 
		ST_GeographyFromText('SRID=4326;POINT(' || station.m_longitude || ' ' || station.m_latitude ||')') ) > {mindist:14.8f}
		AND
		stream.m_start < '{t}'::timestamp
		AND
		(stream.m_end > '{t}'::timestamp OR stream.m_end IS NULL)
		GROUP BY net, sta, loc, channels, lon, lat
		""".format(lon=self.event['lon'], lat=self.event['lat'], dist=max_distance, mindist=min_distance, t=self.event['t'])
	cur = conn.cursor()
	cur.execute(query)
	records = cur.fetchall()
	if obspy.__version__[0] == '0':
		g = Geod(ellps='WGS84')
	stats = []
	for rec in records:
		if not rec[3] in ['HH', 'BH']: # jen vybrane kanaly (+EH, HG?)
			continue
		stn = {'code':rec[1], 'lat':rec[5], 'lon':rec[4], 'network':rec[0], 'location':rec[2], 'channelcode':rec[3], 'model':''}
		if obspy.__version__[0] == '0':
			az,baz,dist = g.inv(self.event['lon'], self.event['lat'], rec[4], rec[5])
		else:
			dist,az,baz = gps2dist_azimuth(self.event['lat'], self.event['lon'], rec[5], rec[4])
		stn['az'] = az; stn['dist'] = dist
		stn['useN'] = stn['useE'] = stn['useZ'] = False
		stn['accelerograph'] = False
		stn['weightN'] = stn['weightE'] = stn['weightZ'] = 1.
		stats.append(stn)
	stats = sorted(stats, key=lambda stn: stn['dist']) # seradit podle vzdalenosti
	if len(stats) > 21:
		stats = stats[0:21] # BECAUSE OF GREENS FUNCTIONS CALCULATION
	self.stations = stats
	self.create_station_index()
	self.models[''] = 0

def read_network_coordinates(self, filename, network='', location='', channelcode='LH', min_distance=None, max_distance=None, max_n_of_stations=None):
	"""
	Read informations about stations from file in ISOLA format.
	Calculate their distances and azimuthes using WGS84 elipsoid.
	Create data structure ``self.stations``. Sorts it according to station epicentral distance.
	
	:param filename: path to file with network coordinates
	:type filename: string
	:param network: all station are from specified network
	:type network: string, optional
	:param location: all stations has specified location
	:type location: string, optional
	:param channelcode: component names of all stations start with these letters (if channelcode is `LH`, component names will be `LHZ`, `LHN`, and `LHE`)
	:type channelcode: string, optional
	:param min_distance: minimal epicentral distance in meters
	:param min_distance: float or None
	:param min_distance: maximal epicentral distance in meters
	:param max_distance: float or None
	:param min_distance: maximal number of stations used in inversion
	:param max_distance: int or None
	
	If ``min_distance`` is ``None``, value is calculated as 2*self.rupture_length. If ``max_distance`` is ``None``, value is calculated as :math:`1000 \cdot 2^{2M}`.
	"""
	# 2DO: osetreni chyby, pokud neni event['lat'] a ['lon']
	if min_distance==None:
		min_distance = 2*self.rupture_length
	if max_distance==None:
		max_distance = 1000 * 2**(mag*2.)
	self.logtext['network'] = s = 'Station coordinates: '+filename
	self.log(s)
	inp  = open(filename, 'r')
	lines = inp.readlines()
	inp.close()
	stats = []
	for line in lines:
		if line == '\n': # skip empty lines
			continue
		# 2DO: souradnice stanic dle UTM
		items = line.split()
		sta,lat,lon = items[0:3]
		if len(items) > 3:
			model = items[3]
		else:
			model = ''
		if model not in self.models:
			self.models[model] = 0
		net = network; loc = location; ch = channelcode # default values given by function parameters
		if ":" in sta or "." in sta:
			l = sta.replace(':', '.').split('.')
			net = l[0]; sta = l[1]
			if len(l) > 2: loc = l[2]
			if len(l) > 3: ch = l[3]
		stn = {'code':sta, 'lat':lat, 'lon':lon, 'network':net, 'location':loc, 'channelcode':ch, 'model':model}
		if obspy.__version__[0] == '0':
			g = Geod(ellps='WGS84')
			az,baz,dist = g.inv(self.event['lon'], self.event['lat'], lon, lat)
		else:
			dist,az,baz = gps2dist_azimuth(float(self.event['lat']), float(self.event['lon']), float(lat), float(lon))
		stn['az'] = az
		stn['dist'] = dist
		stn['useN'] = stn['useE'] = stn['useZ'] = False
		stn['accelerograph'] = False
		stn['weightN'] = stn['weightE'] = stn['weightZ'] = 1.
		if dist > min_distance and dist < max_distance:
			stats.append(stn)
	stats = sorted(stats, key=lambda stn: stn['dist']) # sort by distance
	if max_n_of_stations and len(stats) > max_n_of_stations:
		stats = stats[0:max_n_of_stations]
	self.stations = stats
	self.create_station_index()
	
def create_station_index(self):
	"""
	Creates ``self.stations_index`` which serves for accesing ``self.stations`` items by the station name.
	It is called from :func:`read_network_coordinates`.
	"""
	stats = self.stations
	self.nr = len(stats)
	self.stations_index = {}
	for i in range(self.nr):
		self.stations_index['_'.join([stats[i]['network'], stats[i]['code'], stats[i]['location'], stats[i]['channelcode']])] = stats[i]

def write_stations(self, filename='green/station.dat'):
	"""
	Write file with carthesian coordinates of stations. The file is necessary for Axitra code.
	
	This function is usually called from some of functions related to reading seismograms.
	
	:param filename: name (with path) to created file
	:type filename: string, optional
	"""
	for model in self.models:
		if model:
			f = filename[0:filename.rfind('.')] + '-' + model + filename[filename.rfind('.'):]
		else:
			f = filename
		outp = open(f, 'w')
		outp.write(' Station co-ordinates\n x(N>0,km),y(E>0,km),z(km),azim.,dist.,stat.\n')
		self.models[model] = 0
		for s in self.stations:
			if s['model'] != model:
				continue
			az = radians(s['az'])
			dist = s['dist']/1000 # from meter to kilometer
			outp.write('{N:10.4f} {E:10.4f} {z:10.4f} {az:10.4f} {d:10.4f} {code:4s} ?\n'.format(N=cos(az)*dist, E=sin(az)*dist, z=0, az=s['az'], d=dist, code=s['code']))
			self.models[model] += 1
		outp.close()

