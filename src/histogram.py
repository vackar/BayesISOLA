#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
 
def histogram(data, outfile=None, bins=100, range=None, xlabel='', multiply=1, reference=None, reference2=None, fontsize=None):
	"""
	Plots a histogram of a given data.
	
	:param data: input values
	:type data: array
	:param outfile: filename of the output. If ``None``, plots to the screen.
	:type outfile: string or None, optional
	:param bins: number of bins of the histogram
	:type bins: integer, optional
	:param range: The lower and upper range of the bins. Lower and upper outliers are ignored. If not provided, range is (data.min(), data.max()).
	:type range: tuple of 2 floats, optional
	:param xlabel: x-axis label
	:type xlabel: string, optional
	:param multiply: Normalize the sum of histogram to a given value. If not set, normalize to 1.
	:type multiply: float, optional
	:param reference: plots a line at the given value as a reference
	:type reference: array_like, scalar, or None, optional
	:param reference2: plots a line at the given value as a reference
	:type reference2: array_like, scalar, or None, optional
	:param fontsize: size of the font of tics, labels, etc.
	:type fontsize: scalar, optional
	
	Uses `matplotlib.pyplot.hist <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`_
	"""
	if fontsize:
		plt.rcParams.update({'font.size': fontsize})
	weights = np.ones_like(data) / float(len(data)) * multiply
	if type(bins) == tuple:
		try:
			n = 1 + 3.32 * np.log10(len(data))	# Sturgesovo pravidlo
		except:
			n = 10
		if range:
			n *= (range[1]-range[0])/(max(data)-min(data))
		bins = max(min(int(n), bins[1]), bins[0])
	plt.hist(data, weights=weights, bins=bins, range=range)
	ax = plt.gca()
	ymin, ymax = ax.get_yaxis().get_view_interval()
	if reference != None:
		try:
			iter(reference)
		except:
			reference = (reference,)
		for ref in reference:
			ax.add_artist(Line2D((ref, ref), (0, ymax), color='r', linewidth=5))
	if reference2 != None:
		try:
			iter(reference2)
		except:
			reference2 = (reference2,)
		for ref in reference2:
			ax.add_artist(Line2D((ref, ref), (0, ymax), color=(0.,1.,0.2), linewidth=5, linestyle='--'))
	if range:
		plt.xlim(range[0], range[1])
	if xlabel:
		plt.xlabel(xlabel)
	plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))
	plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,3))
	if outfile:
		plt.savefig(outfile, bbox_inches='tight')
	else:
		plt.show()
	plt.clf()
	plt.close()
def to_percent(y, position):
	"""
	Something with tics positioning used by :func:`histogram`
	# Ignore the passed in position. This has the effect of scaling the default tick locations.
	"""
	s = "{0:2.0f}".format(100 * y)
	if mpl.rcParams['text.usetex'] is True:
		return s + r' $\%$'
	else:
		return s + ' %'
