#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import textwrap
import re # RegExp

from BayesISOLA.MT_comps import a2mt

def html_log(self, outfile='$outdir/index.html', reference=None, h1='ISOLA-ObsPy automated solution', backlink=False, plot_MT=None, plot_uncertainty=None, plot_stations=None, plot_seismo_cova=None, plot_seismo_sharey=None, mouse_figures=None, plot_spectra=None, plot_noise=None, plot_covariance_function=None, plot_covariance_matrix=None, plot_maps=None, plot_slices=None, plot_maps_sum=None):
	"""
	Generates an HTML page containing informations about the calculation and the result together with figures
	
	:param outfile: filename of the created HTML file
	:type outfile: string, optional
	:param reference: reference solution which is shown for comparison
	:type reference: dict or none, optional
	:param h1: main header of the html page
	:type h1: string, optional
	:param backlink: show a link to an list of events located as `index.html` in the parent directory
	:type backlink: bool, optional
	:param plot_MT: path to figure of moment tensor (product of :func:`plot_MT`)
	:type plot_MT: string, optional
	:param plot_uncertainty: path to figures of uncertainty plotted by :func:`plot_uncertainty` (the common part of filename)
	:type plot_uncertainty: string, optional
	:param plot_stations: path to map of inverted stations (product of :func:`plot_stations`)
	:type plot_stations: string, optional
	:param plot_seismo_cova: path to figure of waveform match shown as standardized seismograms (product of :func:`plot_seismo`)
	:type plot_seismo_cova: string, optional
	:param plot_seismo_sharey: path to figure of waveform match shown as original (non-standardized) seismograms  (product of :func:`plot_seismo`)
	:type plot_seismo_sharey: string, optional
	:param mouse_figures: path to figure of detected mouse disturbances (product of :func:`detect_mouse`)
	:type mouse_figures: string, optional
	:param plot_spectra: path to figure of spectra (product of :func:`plot_spectra`)
	:type plot_spectra: string, optional
	:param plot_noise: path to figure of noise (product of :func:`plot_noise`)
	:type plot_noise: string, optional
	:param plot_covariance_function: path to figure of the covariance function (product of :func:`plot_covariance_function`)
	:type plot_covariance_function: string, optional
	:param plot_covariance_matrix: path to figure of the data covariance matrix (product of :func:`plot_covariance_matrix`)
	:type plot_covariance_matrix: string, optional
	:param plot_maps: path to figures of solutions across the grid (top view) plotted by :func:`plot_maps` (the common part of filename)
	:type plot_maps: string, optional
	:param plot_slices: path to figures of solutions across the grid (side view) plotted by :func:`plot_slices` (the common part of filename)
	:type plot_slices: string, optional
	:param plot_maps_sum: path to figures of solutions across the grid plotted by :func:`plot_maps_sum` (the common part of filename)
	:type plot_maps_sum: string, optional
	"""
	out = open(outfile.replace('$outdir', self.outdir), 'w')
	e = self.event
	C = self.centroid
	decomp = self.mt_decomp.copy()
	out.write(textwrap.dedent("""\
		<!DOCTYPE html>
		<html lang="en" dir="ltr">
		<head>
		<meta charset="UTF-8">
		<title>{0:s}</title>
		<link rel="stylesheet" href="../html/style.css" />
		<link rel="stylesheet" href="../html/css/lightbox.min.css">
		</head>
		<body>
		""".format(h1)))
	out.write('<h1>'+h1+'</h1>\n')
	if backlink:
		out.write('<p><a href="../index.html">back to event list</a></p>\n')
	out.write('<dl>  <dt>Method</dt>\n  <dd>Waveform inversion for <strong>' + 
		{1:'deviatoric part of', 0:'full'}[self.deviatoric] + 
		'</strong> moment tensor (' + 
		{1:'5', 0:'6'}[self.deviatoric] + 
		' components)<br />\n    ' + 
		{1:'with the <strong>data covariance matrix</strong> based on real noise', 0:'without the covariance matrix'}[bool(self.Cd_inv)] + 
		{1:'<br />\n    with <strong>crosscovariance</strong> between components', 0:''}[bool(self.LT3)] + 
		'.</dd>\n  <dt>Reference</dt>\n  <dd>Vackář, Gallovič, Burjánek, Zahradník, and Clinton. Bayesian ISOLA: new tool for automated centroid moment tensor inversion, <em>in preparation</em>, <a href="http://geo.mff.cuni.cz/~vackar/papers/isola-obspy.pdf">PDF</a></dd>\n</dl>\n\n')
	out.write(textwrap.dedent('''\
		<h2>Hypocenter location</h2>
		
		<dl>
		<dt>Agency</dt>
		<dd>{agency:s}</dd>
		<dt>Origin time</dt>
		<dd>{t:s}</dd>
		<dt>Latitude</dt>
		<dd>{lat:8.3f}° N</dd>
		<dt>Longitude</dt>
		<dd>{lon:8.3f}° E</dd>
		<dt>Depth</dt>
		<dd>{d:3.1f} km</dd>
		<dt>Magnitude</dt>
		<dd>{m:3.1f}</dd>
		</dl>
		'''.format(
			t=e['t'].strftime('%Y-%m-%d %H:%M:%S'), 
			lat=float(e['lat']), 
			lon=float(e['lon']), 
			d=e['depth']/1e3, 
			agency=e['agency'], 
			m=e['mag'])))
	out.write('\n\n<h2>Results</h2>\n\n')
	if plot_MT:
		out.write(textwrap.dedent('''\
			<div class="thumb tright">
			<a href="{0:s}" data-lightbox="MT" data-title="moment tensor best solution">
				<img alt="MT" src="{0:s}" width="199" height="200" class="thumbimage" />
			</a>
			<div class="thumbcaption">
				moment tensor best solution
			</div>
			</div>
			'''.format(plot_MT)))
	if plot_uncertainty:
		k = plot_uncertainty.rfind(".")
		s1 = plot_uncertainty[:k]+'_'; s2 = plot_uncertainty[k:]
		out.write(textwrap.dedent('''\
			<div class="thumb tright">
			<a href="{MT_full:s}" data-lightbox="MT" data-title="moment tensor uncertainty">
				<img alt="MT" src="{MT_full:s}" width="199" height="200" class="thumbimage" />
			</a>
			<div class="thumbcaption">
				moment tensor uncertainty
			</div>
			</div>

			<div class="thumb tright">
			<a href="{MT_DC:s}" data-lightbox="MT" data-title="moment tensor DC-part uncertainty">
				<img alt="MT" src="{MT_DC:s}" width="199" height="200" class="thumbimage" />
			</a>
			<div class="thumbcaption">
				DC-part uncertainty
			</div>
			</div>
			'''.format(MT_full=s1+'MT'+s2, MT_DC=s1+'MT_DC'+s2)))
	t = self.event['t'] + C['shift']
	out.write(textwrap.dedent('''\
		<h3>Centroid location</h3>

		<table>
			<tr>
			<th></th>
			<th>absolute</th>
			<th>relative</th>
			</tr>
			<tr>
			<th>Time</th>
			<td>{t:s}</td>
			<td>{shift:5.2f} s {sgn_shift:s} origin time</td>
			</tr>
			<tr>
			<th>Latitude</th>
			<td>{lat:8.3f}° {sgn_lat:s}</td>
			<td>{x:5.0f} m {dir_x:s} of the epicenter</td>
			</tr>
			<tr>
			<th>Longitude</th>
			<td>{lon:8.3f}° {sgn_lon:s}</td>
			<td>{y:5.0f} m {dir_y:s} of the epicenter</td>
			</tr>
			<tr>
			<th>Depth</th>
			<td>{d:5.1f} km</td>
			<td>{dd:5.1f} km {sgn_dd:s} than location</td>
			</tr>
		</table>

		'''.format(
			t = 	t.strftime('%Y-%m-%d %H:%M:%S'), 
			lat = 	abs(C['lat']),
			sgn_lat = {1:'N', 0:'', -1:'S'}[int(np.sign(C['lat']))], 
			lon = 	abs(C['lon']),
			sgn_lon = {1:'E', 0:'', -1:'W'}[int(np.sign(C['lon']))],
			d = 	C['z']/1e3,
			x = 	abs(C['x']),
			dir_x = 	{1:'north', 0:'', -1:'south'}[int(np.sign(C['x']))], 
			y = 	abs(C['y']),
			dir_y = 	{1:'east', 0:'', -1:'west'}[int(np.sign(C['y']))],
			shift = 	abs(C['shift']), 
			sgn_shift={1:'after', 0:'after', -1:'before'}[int(np.sign(C['shift']))],
			dd = 	abs(C['z']-self.event['depth'])/1e3,
			sgn_dd = 	{1:'deeper', 0:'deeper', -1:'shallower'}[int(np.sign(C['z']-self.event['depth']))]
		)))
	if C['edge']:
		out.write('<p class="warning">Warning: the solution lies on the edge of the grid!</p>')
	if C['shift'] in (self.shifts[0], self.shifts[-1]):
		out.write('<p class="warning">Warning: the solution lies on the edge of the time-grid!</p>')

	mt2 = a2mt(C['a'], system='USE')
	c = max(abs(min(mt2)), max(mt2))
	c = 10**np.floor(np.log10(c))
	MT2 = mt2 / c

	out.write('\n\n<h3>Moment tensor and its quality</h3>\n\n')
	if self.mt_decomp and reference:
		decomp.update(rename_keys(reference, 'ref_'))
		out.write('''
<table>
  <tr><th>&nbsp;</th><th>ISOLA-ObsPy</th><th>SeisComP</th></tr>
  <tr><th colspan="3" class="center">Centroid position</th></tr>
  <tr><th>depth</th>	<td>{depth:3.1f} km</td>	<td>{ref_depth:3.1f} km</td></tr>
  <tr><th colspan="3" class="center">Seismic moment</th></tr>
  <tr><th>scalar seismic moment M<sub>0</sub></th>	<td>{mom:5.2e} Nm</td>	<td></td></tr>
  <tr><th>moment magnitude M<sub>w</sub></th>	<td>{Mw:3.1f}</td>	<td>{ref_Mw:3.1f}</td></tr>
  <tr><th colspan="3" class="center">Moment tensor components</th></tr>
  <tr><th>M<sub>rr</sub></th>			<td>{1:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>&theta;&theta;</sub></th>	<td>{2:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>&#981;&#981;</sub></th>		<td>{3:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>r&theta;</sub></th>		<td>{4:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>r&#981;</sub></th>		<td>{5:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>&theta;&#981;</sub></th>	<td>{6:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th colspan="3" class="center">Moment tensor decomposition</th></tr>
  <tr><th>DC component</th>	<td>{dc_perc:3.0f} %</td>	<td>{ref_dc_perc:3.0f} %</td></tr>
  <tr><th>CLVD component</th>	<td>{clvd_perc:3.0f} %</td>	<td>{ref_clvd_perc:3.0f} %</td></tr>
'''.format(c, *MT2, depth=C['z']/1e3, **decomp))
		if not self.deviatoric:
			out.write('''
  <tr><th>isotropic component</th>	<td>{iso_perc:3.0f} %</td>	<td>{ref_iso_perc:3.0f} %</td></tr>
'''.format(**decomp))
		out.write('''
  <tr><th>strike</th>	<td>{s1:3.0f} / {s2:3.0f}</td>	<td>{ref_s1:3.0f} / {ref_s2:3.0f}</td></tr>
  <tr><th>dip</th>  	<td>{d1:3.0f} / {d2:3.0f}</td>	<td>{ref_d1:3.0f} / {ref_d2:3.0f}</td></tr>
  <tr><th>slip-rake</th>	<td>{r1:3.0f} / {r2:3.0f}</td>	<td>{ref_r1:3.0f} / {ref_r2:3.0f}</td></tr>
  <tr><th colspan="3" class="center">Result quality</th></tr>
  <tr><th>condition number</th>	<td>{CN:2.0f}</td>	<td></td></tr>
  <tr><th>variance reduction</th>	<td>{VR:2.0f} %</td>	<td></td></tr>
'''.format(VR=C['VR']*100, CN=C['CN'], **decomp))
	elif self.mt_decomp:
		out.write('''
<table>
  <tr><th colspan="2" class="center">Centroid position</th></tr>
  <tr><th>depth</th>	<td>{depth:3.1f} km</td></tr>
  <tr><th colspan="2" class="center">Seismic moment</th></tr>
  <tr><th>scalar seismic moment M<sub>0</sub></th>	<td>{mom:5.2e} Nm</td></tr>
  <tr><th>moment magnitude M<sub>w</sub></th>	<td>{Mw:3.1f}</td></tr>
  <tr><th colspan="2" class="center">Moment tensor components</th></tr>
  <tr><th>M<sub>rr</sub></th>			<td>{1:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&theta;</sub></th>	<td>{2:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&#981;&#981;</sub></th>		<td>{3:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&theta;</sub></th>		<td>{4:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&#981;</sub></th>		<td>{5:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&#981;</sub></th>	<td>{6:5.2f} * {0:5.0e}</td></tr>
  <tr><th colspan="2" class="center">Moment tensor decomposition</th></tr>
  <tr><th>DC</th>	<td>{dc_perc:3.0f} %</td></tr>
  <tr><th>CLVD</th>	<td>{clvd_perc:3.0f} %</td></tr>
'''.format(c, *MT2, depth=C['z']/1e3, **decomp))
		if not self.deviatoric:
			out.write('''
  <tr><th>ISO</th>	<td>{iso_perc:3.0f} %</td></tr>
'''.format(**decomp))
		out.write('''
  <tr><th>strike</th>	<td>{s1:3.0f} / {s2:3.0f}</td></tr>
  <tr><th>dip</th>  	<td>{d1:3.0f} / {d2:3.0f}</td></tr>
  <tr><th>rake</th>	<td>{r1:3.0f} / {r2:3.0f}</td></tr>
  <tr><th colspan="2" class="center">Quality measures</th></tr>
  <tr><th>condition number</th>	<td>{CN:2.0f}</td></tr>
  <tr><th>variance reduction</th>	<td>{VR:2.0f} %</td></tr>
'''.format(VR=C['VR']*100, CN=C['CN'], **decomp))
	else:
		out.write('''
<table>
  <tr><th colspan="2" class="center">Centroid position</th></tr>
  <tr><th>depth</th>	<td>{depth:3.1f} km</td></tr>
  <tr><th colspan="2" class="center">Moment tensor components</th></tr>
  <tr><th>M<sub>rr</sub></th>			<td>{1:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&theta;</sub></th>	<td>{2:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&#981;&#981;</sub></th>		<td>{3:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&theta;</sub></th>		<td>{4:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&#981;</sub></th>		<td>{5:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&#981;</sub></th>	<td>{6:5.2f} * {0:5.0e}</td></tr>
  <tr><th colspan="2" class="center">Result quality</th></tr>
  <tr><th>condition number</th>	<td>{CN:2.0f}</td></tr>
  <tr><th>variance reduction</th>	<td>{VR:2.0f} %</td></tr>
'''.format(c, *MT2, depth=C['z']/1e3, VR=C['VR']*100, CN=C['CN']))
	if self.max_VR:
		out.write('  <tr><th>VR ({2:d} closest components)</th>	<td>{1:2.0f} %</td>{0:s}</tr>'.format(('', '<td></td>')[bool(reference)], self.max_VR[0]*100, self.max_VR[1]))
	if reference and 'kagan' in reference:
		out.write('<tr><th>Kagan angle</th>	<td colspan="2" class="center">{0:3.1f}°</td></tr>\n'.format(reference['kagan']))
	out.write('</table>\n\n')
		
	if plot_uncertainty:
		out.write('''
<h3>Histograms&mdash;uncertainty of MT parameters</h3>

<div class="thumb tleft">
  <a href="{DC:s}" data-lightbox="histogram" data-title="DC-part uncertainty">
    <img alt="" src="{DC:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    DC-part
  </div>
</div>

<div class="thumb tleft">
  <a href="{CLVD:s}" data-lightbox="histogram" data-title="CLVD-part uncertainty">
    <img alt="" src="{CLVD:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    CLVD-part
  </div>
</div>
'''.format(DC=s1+'comp-1-DC'+s2, CLVD=s1+'comp-2-CLVD'+s2))
		if not self.deviatoric:
			out.write('''
<div class="thumb tleft">
  <a href="{ISO:s}" data-lightbox="histogram" data-title="isotropic part uncertainty">
    <img alt="" src="{ISO:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    CLVD-part
  </div>
</div>
'''.format(ISO=s1+'comp-3-ISO'+s2))
		out.write('''
<div class="thumb tleft">
  <a href="{Mw:s}" data-lightbox="histogram" data-title="moment magnitude uncertainty">
    <img alt="" src="{Mw:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    moment magnitude
  </div>
</div>

'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW'+s2, NS=s1+'place-NS'+s2, time=s1+'time-shift'+s2))
		if len(self.shifts) > 1 or len(self.grid) > 1:
			out.write('\n\n<h3>Histograms&mdash;uncertainty of centroid position and time</h3>\n\n')
		if len (self.depths) > 1:
			out.write('''
<div class="thumb tleft">
  <a href="{depth:s}" data-lightbox="histogram" data-title="centroid depth uncertainty">
    <img alt="" src="{depth:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    centroid depth
  </div>
</div>

'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW.png'+s2, NS=s1+'place-NS.png'+s2, time=s1+'time-shift'+s2))
		if len(self.grid) > len(self.depths):
			out.write('''
<div class="thumb tleft">
  <a href="{EW:s}" data-lightbox="histogram" data-title="centroid position east-west uncertainty">
    <img alt="" src="{EW:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    position east-west
  </div>
</div>

<div class="thumb tleft">
  <a href="{NS:s}" data-lightbox="histogram" data-title="centroid position north-south uncertainty">
    <img alt="" src="{NS:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    position north-south
  </div>
</div>

'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW'+s2, NS=s1+'place-NS'+s2, time=s1+'time-shift'+s2))
		if len(self.shifts) > 1:
			out.write('''
<div class="thumb tleft">
  <a href="{time:s}" data-lightbox="histogram" data-title="centroid time uncertainty">
    <img alt="" src="{time:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    centroid time
  </div>
</div>
'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW'+s2, NS=s1+'place-NS'+s2, time=s1+'time-shift'+s2))
	out.write('\n\n<h2>Data used</h2>\n\n')
	if plot_stations:
		out.write('''
<div class="thumb tright">
  <a href="{0:s}" data-lightbox="stations">
    <img alt="" src="{0:s}" width="200" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    stations used
  </div>
</div>
'''.format(plot_stations))
	if 'components' in self.logtext:
		s = self.logtext['components']
		i = s.find('(Hz)\t(Hz)\n')
		s = s[i+10:]
		out.write('\n\n<h3>Components used in inversion and their weights</h3>\n\n<table>\n  <tr><th colspan="2">station</th>	<th colspan="3">component</th>		<th><abbr title="epicentral distance">distance *</abbr></th>	<th>azimuth</th>	<th>fmin</th>	<th>fmax</th></tr>\n  <tr><th>code</th>	<th>channel</th>	<th>Z</th>	<th>N</th>	<th>E</th>	<th>(km)</th>	<th>(deg)</th>	<th>(Hz)</th>	<th>(Hz)</th></tr>\n')
		s = s.replace('\t', '</td>\t<td>').replace('\n', '</td></tr>\n<tr><td>')[:-8]
		s = '<tr><td>' + s + '</table>\n\n'
		out.write(s)
	if 'mouse' in self.logtext:
		out.write('<h3>Mouse detection</h3>\n<p>\n')
		s = self.logtext['mouse']
		lines = s.split('\n')
		if mouse_figures:
			p = re.compile('  ([0-9A-Z]+) +([A-Z]{2})([ZNE]{1}).* (MOUSE detected.*)')
		for line in lines:
			if mouse_figures:
				m = p.match(line)
				if m:
					line = '  <a href="{fig:s}mouse_YES_{0:s}{comp:s}.png" data-lightbox="mouse">\n    {0:s} {1:s}{2:s}</a>: {3:s}'.format(*m.groups(), fig=mouse_figures, comp={'Z':'0', 'N':'1', 'E':'2'}[m.groups()[2]])
			out.write(line+'<br />\n')
	out.write('<h3>Data source</h3>\n<p>\n')
	if 'network' in self.logtext:
		out.write(self.logtext['network'] + '<br />\n')
	if 'data' in self.logtext:
		out.write(self.logtext['data'] + '\n')
	out.write('</p>\n\n')
	if plot_seismo_cova:
		out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="seismo" data-title="waveform fit: filtered by C<sub>D</sub>">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    waveform fit <br />(filtered)
  </div>
</div>
'''.format(plot_seismo_cova))
	if plot_seismo_sharey:
		out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="seismo" data-title="waveform fit: original data">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    waveform fit <br />(non-filtered)
  </div>
</div>
'''.format(plot_seismo_sharey))
	if plot_spectra:
		out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="spectra" data-title="waveform spectra">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    waveform spectra<br />&nbsp;
  </div>
</div>
'''.format(plot_spectra))
	if plot_noise:
		out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="noise" data-title="before-event noise">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    before-event noise<br />&nbsp;
  </div>
</div>
'''.format(plot_noise))
	if plot_covariance_function:
		out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="cova_func" data-title="data covariance function">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    data covariance function<br />&nbsp;
  </div>
</div>
'''.format(plot_covariance_function))
	if plot_covariance_matrix:
		out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="Cd" data-title="data covariance matrix">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    data covariance matrix<br />&nbsp;
  </div>
</div>
'''.format(plot_covariance_matrix))
	if plot_maps or plot_slices or plot_maps_sum:
		out.write('\n\n\n<h2>Stability and uncertainty of the solution</h2>')
	if plot_maps_sum:
		out.write('\n\n<h3>Posterior probability density function (PPD)</h3>\n\n')
		k = plot_maps_sum.rfind(".")
		s1 = plot_maps_sum[:k] + '_'
		s2 = plot_maps_sum[k:]
		out.write('''
<div class="thumb tleft">
  <a href="{top:s}" data-lightbox="PPD">
    <img alt="" src="{top:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    PPD: top view
  </div>
</div>

<div class="thumb tleft">
  <a href="{NS:s}" data-lightbox="PPD">
    <img alt="" src="{NS:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    PPD: north-south view
  </div>
</div>

<div class="thumb tleft">
  <a href="{WE:s}" data-lightbox="PPD">
    <img alt="" src="{WE:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    PPD: west-east view
  </div>
</div>
'''.format(top=s1+'top'+s2, NS=s1+'N-S'+s2, WE=s1+'W-E'+s2))
	if plot_maps:
		out.write('\n\n<h3>Stability in space (top view)</h3>\n\n<div class="thumb tleft">\n')
		k = plot_maps.rfind(".")
		for z in self.depths:
			filename = plot_maps[:k] + "_{0:0>5.0f}".format(z) + plot_maps[k:]
			out.write('  <a href="{0:s}" data-lightbox="map">\n    <img alt="" src="{0:s}" height="100" class="thumbimage" />\n  </a>\n'.format(filename))
		out.write('  <div class="thumbcaption">\n    click to compare different depths\n  </div>\n</div>\n')
	if plot_slices:
		k = plot_slices.rfind(".")
		s1 = plot_slices[:k] + '_'
		s2 = plot_slices[k:]
		out.write('\n\n<h3>Stability in space (side view)</h3>\n\n<div class="thumb tleft">\n')
		for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE'):
			out.write('  <a href="{0:s}" data-lightbox="slice">\n    <img alt="" src="{0:s}" height="150" class="thumbimage" />\n  </a>\n'.format(s1+slice+s2))
		out.write('  <div class="thumbcaption">\n    click to compare different points of view\n  </div>\n</div>\n')
	out.write('''

<h2>Calculation parameters</h2>

<h3>Grid-search over space</h3>
<dl>
  <dt>number of points</dt>
  <dd>{points:4d}</dd>
  <dt>horizontal step</dt>
  <dd>{x:5.0f} m</dd>
  <dt>vertical step</dt>
  <dd>{z:5.0f} m</dd>
  <dt>grid radius</dt>
  <dd>{radius:6.3f} km</dd>
  <dt>minimal depth</dt>
  <dd>{dmin:6.3f} km</dd>
  <dt>maximal depth</dt>
  <dd>{dmax:6.3f} km</dd>
  <dt>rupture length (estimated)</dt>
  <dd>{rlen:6.3f} km</dd>
</dl>

<h3>Grid-search over time</h3>
<dl>
  <dt>min</dt>
  <dd>{sn:5.2f} s ({Sn:3d} samples)</dd>
  <dt>max</dt>
  <dd>{sx:5.2f} s ({Sx:3d} samples)</dd>
  <dt>step</dt>
  <dd>{step:4.2f} s ({STEP:3d} samples)</dd>
</dl>

<h3>Green's function calculation</h3>
<dl>
  <dt>Crustal model</dt>
  <dd>{crust:s}</dd>
  <dt>npts</dt>
  <dd>{npts_elemse:4d}</dd>
  <dt>tl</dt>
  <dd>{tl:4.2f}</dd>
  <dt>freq</dt>
  <dd>{freq:4d}</dd>
  <dt>npts for inversion</dt>
  <dd>{npts_slice:4d}</dd>
</dl>

<h3>Sampling frequencies</h3>
<dl>
  <dt>Data sampling</dt>
  <dd>{samplings:s}</dd>
  <dt>Common sampling</dt>
  <dd>{SAMPRATE:5.1f} Hz</dd>
  <dt>Decimation factor</dt>
  <dd>{decimate:3.0f} x</dd>
  <dt>Sampling used</dt>
  <dd>{samprate:5.1f} Hz</dd>
</dl>
'''.format(
	points = len(self.grid), 
	x = self.step_x, 
	z = self.step_z, 
	radius = self.radius/1e3, 
	dmin = self.depth_min/1e3, 
	dmax = self.depth_max/1e3,
	rlen = self.rupture_length/1e3,
	sn = self.shift_min, 
	Sn = self.SHIFT_min, 
	sx = self.shift_max, 
	Sx = self.SHIFT_max, 
	step = self.shift_step, 
	STEP = self.SHIFT_step,
	npts_elemse = self.npts_elemse, 
	tl = self.tl, 
	freq = self.freq, 
	npts_slice = self.npts_slice,
	samplings = self.logtext['samplings'], 
	decimate = self.max_samprate / self.samprate, 
	samprate = self.samprate, 
	SAMPRATE = self.max_samprate,
	crust = self.logtext['crust']
))


	out.write("""
<script src="../html/js/lightbox-plus-jquery.min.js"></script>
<script>
lightbox.option({
'resizeDuration': 0
})
</script>
</body>
</html>
""")
	out.close()
