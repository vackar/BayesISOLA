#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from BayesISOLA.MT_comps import a2mt, decompose, decompose_mopad

def print_solution(self, precision=2):
	"""
	Write into log the best solution ``self.centroid``.
	
	:param precision: number of decimal digits of moment tensor components (default ``2``)
	:type precision: int, optional
	"""
	C = self.centroid
	t = self.event['t'] + C['shift']
	self.log('\nCentroid location:\n  Centroid time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth {d:5.1f} km'.format(t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=C['lat'], lon=C['lon'], d=C['z']/1e3))
	self.log('  ({0:5.0f} m to the north and {1:5.0f} m to the east with respect to epicenter)'.format(C['x'], C['y']))
	if C['edge']:
		self.log('  Warning: the solution lies on the edge of the grid!')
	mt2 = a2mt(C['a'], system='USE')
	c = max(abs(min(mt2)), max(mt2))
	c = 10**np.floor(np.log10(c))
	MT2 = mt2 / c
	if C['shift'] >= 0:
		self.log('  time: {0:5.2f} s after origin time\n'.format(C['shift']))
	else:
		self.log('  time: {0:5.2f} s before origin time\n'.format(-C['shift']))
	if C['shift'] in (self.d.shifts[0], self.d.shifts[-1]):
		self.log('  Warning: the solution lies on the edge of the time-grid!')
	self.log('  VR: {0:4.0f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN']))
	#self.log('  VR: {0:8.4f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN'])) # DEBUG
	self.log('  MT [ {1:{0}}  {2:{0}}  {3:{0}}  {4:{0}}  {5:{0}}  {6:{0}}]:'.format(precision+3, 'Mrr','Mtt','Mpp','Mrt','Mrp','Mtp'))
	self.log('     [{1:{7}.{8}f}  {2:{7}.{8}f}  {3:{7}.{8}f}  {4:{7}.{8}f}  {5:{7}.{8}f}  {6:{7}.{8}f} ] * {0:5.0e}'.format(c, *MT2, precision+3, precision))

def print_fault_planes(self, precision='3.0', tool=''):
	"""
	Decompose the moment tensor of the best grid point by :func:`decompose` and writes the result to the log.
	
	:param precision: precision of writing floats, like ``5.1`` for 5 letters width and 1 decimal place (default ``3.0``)
	:type precision: string, optional
	:param tool: tool for the decomposition, `mopad` for :func:`decompose_mopad`, otherwise :func:`decompose` is used
	"""
	mt = a2mt(self.centroid['a'])
	if tool == 'mopad':
		self.mt_decomp = decompose_mopad(mt)
	else:
		self.mt_decomp = decompose(mt)
	self.log('''\nScalar Moment: M0 = {{mom:5.2e}} Nm (Mw = {{Mw:3.1f}})
  DC component: {{dc_perc:{0:s}f}} %,   CLVD component: {{clvd_perc:{0:s}f}} %,   ISOtropic component: {{iso_perc:{0:s}f}} %
  Fault plane 1: strike = {{s1:{0:s}f}}, dip = {{d1:{0:s}f}}, slip-rake = {{r1:{0:s}f}}
  Fault plane 2: strike = {{s2:{0:s}f}}, dip = {{d2:{0:s}f}}, slip-rake = {{r2:{0:s}f}}'''.format(precision).format(**self.mt_decomp))

 
