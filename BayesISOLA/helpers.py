#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Various data manipulation / arithmetic / waveform filtering functions.

"""

import numpy as np
import fractions

def rename_keys(somedict, prefix='', suffix=''):
	"""
	Returns a dictionary with keys renamed by adding some prefix and/or suffix
	:param somedict: dictionary, whose keys will be remapped
	:type somedict: dictionary
	:param prefix: new keys starts with
	:type prefix: string, optional
	:param suffix: new keys ends with
	:type suffix: string, optional
	:returns : dictionary with keys renamed
	"""
	return dict(map(lambda key, value: (prefix+str(key)+suffix, value), somedict.items()))

def next_power_of_2(n):
	"""
	Return next power of 2 greater than or equal to ``n``
	
	:type n: integer
	"""
	return 2**(n-1).bit_length()

def lcmm(b, *args):
	"""
	Returns generelized least common multiple.
	
	:param b,args: numbers to compute least common multiple of them 
	:type b,args: float, which is a multiple of 0.00033333
	:returns: the least multiple of ``a`` and ``b``
	"""
	b = 3/b
	if b - round(b) < 1e6:
		b = round(b)
	for a in args:
		a = 3/a
		if a - round(a) < 1e6:
			a = round(a)
		b = fractions.gcd(a, b)
	return 3/b

def my_filter(data, fmin, fmax):
	"""
	Filter used for filtering both elementary and observed seismograms
	"""
	if fmax:
		data.filter('lowpass', freq=fmax)
	if fmin:
		data.filter('highpass', freq=fmin, corners=2)
		data.filter('highpass', freq=fmin, corners=2)

def decimate(a, n=2):
	"""
	Decimates given sequence.
	
	:param data: data
	:type data: 1-D array
	:param n: decimation factor
	:type n: integer, optional
	
	Before decimating, filter out frequencies over Nyquist frequency using :func:`numpy.fft.fft`
	"""
	npts = len(a)
	#NPTS = npts # next_power_of_2(npts)
	NPTS = npts
	A = np.fft.fft(a, NPTS)
	idx = int(np.round(npts/n/2))
	A[idx:NPTS-idx+1] = 0+0j
	a = np.fft.ifft(A)
	if npts % (2*n) == 1 or n!=2: # keep odd length for decimation factor 2
		return a[:npts:n].real
	else:
		return a[1:npts:n].real
