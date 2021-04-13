#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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

#def lcm(a,b):
	#"""
	#Returns generelized least common multiple.
	
	#:param a,b: numbers to compute least common multiple of them 
	#:type a,b: float, which is a multiple of 0.00033333
	#:returns: the least multiple of ``a`` and ``b``
	#"""
	#return abs(a * b * 9e6) / fractions.gcd(a*3e3,b*3e3) / 3e3 if a and b else 0
#def lcmm(*args):
    #"""Return generalized least common multiple of args."""
    #return reduce(lcm, args)

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

