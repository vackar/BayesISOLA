#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moment tensor decomposition and conversion of its components.

"""

import subprocess
import numpy as np

def a2mt(a, system='NEZ'):
	"""
	Convert the coefficients of elementary seismograms to moment tensor components.
	
	:param a: coefficients of elementary seismograms
	:type a: list of 6 floats
	:param system: coordinate system: 'NEZ' = coordinate positive to north-east-down in given order, 'USE' = up-south-east
	:type system: string
	:return: list of 6 components of the moment tensor
	"""
	mt = [-a[3,0]+a[5,0], -a[4,0]+a[5,0], a[3,0]+a[4,0]+a[5,0], a[0,0], a[1,0], -a[2,0]] # [M11, M22, M33, M12, M13, M23] in NEZ system
	if system == 'NEZ':
		return mt
	elif system == 'USE':
		return [mt[2], mt[0], mt[1], mt[4], -mt[5], -mt[3]] # convert to USE system

def decompose_mopad(mt):
	"""
	Decomposition of the moment tensor using ``obspy-mopad``.
	
	:param mt: moment tensor in system 'NEZ'
	:type mt: list of 6 floats
	:return: dictionary {'dc_perc': double couple percentage, 'clvd_perc': compensated linear vector dipole percentage, 'iso_perc': isotropic component percentage, 'faultplanes': list of fault planes parameters, 'moment': scalar seismic moment, 'Mw': moment magnitude :math:`M_W`, 's1': strike (fault plane 1), 'd1': dip (fault plane 1), 'r1': slip rake (fault plane 1), 's2': strike (fault plane 2), 'd2': dip (fault plane 2), 'r2': slip rake (fault plane 2)}
	"""
	process = subprocess.Popen(['obspy-mopad', 'decompose', '-t 20', '-c', '--', '{0:f},{1:f},{2:f},{3:f},{4:f},{5:f}'.format(*mt)], stdout=subprocess.PIPE)
	out, err = process.communicate()
	out = eval(out)
	f = out[23]
	return {
		'iso_perc':out[5], 
		'dc_perc':out[9], 
		'clvd_perc':out[15], 
		'mom':out[16], 
		'Mw':out[17],
		'eigvecs':out[18],
		'eigvals':out[19],
		'p':out[20],
		't':out[22],
		'faultplanes':out[23], 
		's1':f[0][0], 'd1':f[0][1], 'r1':f[0][2], 's2':f[1][0], 'd2':f[1][1], 'r2':f[1][2]}

def decompose(mt):
	"""
	Decomposition of the moment tensor using eigenvalues and eigenvectors according to paper Vavrycuk, JoSE.
	
	:param mt: moment tensor in system 'NEZ'
	:type mt: list of 6 floats
	:return: dictionary {'dc_perc': double couple percentage, 'clvd_perc': compensated linear vector dipole percentage, 'iso_perc': isotropic component percentage, 'faultplanes': list of fault planes parameters, 'moment': scalar seismic moment, 'Mw': moment magnitude :math:`M_W`, 's1': strike (fault plane 1), 'd1': dip (fault plane 1), 'r1': slip rake (fault plane 1), 's2': strike (fault plane 2), 'd2': dip (fault plane 2), 'r2': slip rake (fault plane 2)}
	"""
	M = np.array([
		[mt[0], mt[3], mt[4]],
		[mt[3], mt[1], mt[5]],
		[mt[4], mt[5], mt[2]]])
	m,v = np.linalg.eig(M)
	idx = m.argsort()[::-1]   
	m = m[idx]
	v = v[:,idx]
	
	iso  = 1./3. * m.sum()
	clvd = 2./3. * (m[0] + m[2] - 2*m[1])
	dc   = 1./2. * (m[0] - m[2] - np.abs(m[0] + m[2] - 2*m[1]))
	moment = np.abs(iso) + np.abs(clvd) + dc
	iso_perc  = 100 * iso/moment
	clvd_perc = 100 * clvd/moment
	dc_perc   = 100 * dc/moment
	Mw = 2./3. * np.log10(moment) - 18.1/3.
	p = v[:,0]
	n = v[:,1]
	t = v[:,2]
	c1_4 = 0.5 * np.sqrt(2)
	n1 = (p+t) * c1_4 # normals to fault planes
	n2 = (p-t) * c1_4
	
	if iso_perc < 99.9:
		s1, d1, r1 = angles(n2, n1)
		s2, d2, r2 = angles(n1, n2)
	else:
		s1, d1, r1 = (None, None, None)
		s2, d2, r2 = (None, None, None)
	return {'dc_perc':dc_perc, 'clvd_perc':clvd_perc, 'iso_perc':iso_perc, 'mom':moment, 'Mw':Mw, 'eigvecs':v, 'eigvals':m,
		'p':p, 't':t, 'n':n, 
		's1':s1, 'd1':d1, 'r1':r1, 's2':s2, 'd2':d2, 'r2':r2, 
		'faultplanes':[(s1, d1, r1), (s2, d2, r2)]}

def angles(n1, n2):
	"""
	Calculate strike, dip, and rake from normals to the fault planes.
	
	:param n1, n2: normals to the fault planes
	:type n1, n2: list or 1-D array of 3 floats
	:return: return a tuple of the strike, dip, and rake (one of two possible solutions; the second can be obtained by switching parameters ``n1`` and ``n2``)
	
	Written according to the fortran program sile4_6acek.for by J. Sileny
	"""
	eps = 1e-3
	if n1[2] > 0:
		n2 *= -1
		n1 *= -1
	if -n1[2] < 1:
		dip = np.arccos(-n1[2])
	else:
		dip = 0.
	if abs(abs(n1[2])-1) < eps: # n1[2] close to +-1
		rake = 0.
		strike = np.arctan2(n2[1], n2[0])
		if strike < 0: strike += 2*np.pi
	else:
		strike = np.arctan2(-n1[0], n1[1])
		if strike < 0: strike += 2*np.pi
		cf = np.cos(strike)
		sf = np.sin(strike)
		if abs(n1[2]) < eps:
			if abs(strike) < eps:
				rake = np.arctan2(-n2[2], n2[0])
			elif abs(abs(strike)-np.pi/2) < eps:
				rake = np.arctan2(-n2[2], n2[1])
			else:
				if abs(cf) > abs(sf):
					rake = np.arctan2(-n2[2], n2[0]/cf)
				else:
					rake = np.arctan2(-n2[2], n2[1]/sf)
		else:
			rake = np.arctan2((n2[0]*sf-n2[1]*cf)/-n1[2], n2[0]*cf+n2[1]*sf)
	strike, dip, rake = np.rad2deg((strike, dip, rake))
	return strike, dip, rake
