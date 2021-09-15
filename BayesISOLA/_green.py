#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
import os.path
import hashlib

from BayesISOLA.axitra import Axitra_wrapper

def set_Greens_parameters(self):
	"""
	Sets parameters for Green's function calculation:
		- time window length ``self.tl``
		- number of frequencies ``self.freq``
		- spatial periodicity ``self.xl``
		
	Writes used parameters to the log file.
	"""
	self.tl = self.npts_elemse/self.samprate
	#freq = int(math.ceil(fmax*tl))
	#self.freq = min(int(math.ceil(self.fmax*self.tl))*2, self.npts_elemse/2) # pocitame 2x vic frekvenci, nez pak proleze filtrem, je to pak lepe srovnatelne se signalem, ktery je kauzalne filtrovany
	self.freq = int(self.npts_elemse/2)+1
	self.xl = max(np.ceil(self.d.stations[self.d.nr-1]['dist']/1000), 100)*1e3*20 # `xl` 20x vetsi nez nejvetsi epicentralni vzdalenost, zaokrouhlena nahoru na kilometry, minimalne 2000 km
	self.log("\nGreen's function calculation:\n  npts: {0:4d}\n  tl: {1:4.2f}\n  freq: {2:4d}\n  npts for inversion: {3:4d}\n  source time function: {4:s}".format(self.npts_elemse, self.tl, self.freq, self.npts_slice, self.d.stf_description))

def write_Greens_parameters(self):
	"""
	Writes file grdat.hed - parameters for gr_xyz (Axitra)
	"""
	for model in self.d.models:
		if model:
			f = 'green/grdat' + '-' + model + '.hed'
		else:
			f = 'green/grdat.hed'
		grdat = open(f, 'w')
		grdat.write("&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.d.models[model], xl=self.xl)) # 'nc' is probably ignored in the current version of gr_xyz???
		grdat.close()

def verify_Greens_parameters(self):
	"""
	Check whetrer parameters in file grdat.hed (probably used in Green's function calculation) are the same as used now.
	If it agrees, return True, otherwise returns False, print error description, and writes it into log.
	"""
	try:
		grdat = open('green/grdat.hed', 'r')
	except:
		readable = False
	else:
		readable = True
	if not readable or grdat.read() != "&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.d.nr, xl=self.xl):
		desc = 'Pre-calculated Green\'s functions calculated with different parameters (e.g. sampling) than used now, calculate Green\'s functions again. Exiting...'
		self.log(desc)
		print(desc)
		print ("Expected content of green/grdat.hed:\n&input\nnc=99\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(freq=self.freq,tl=self.tl,nr=self.d.nr, xl=self.xl))
		return False
	grdat.close()
	return True

def verify_Greens_headers(self):
	"""
	Checked whether elementary-seismogram-metadata files (created when the Green's functions were calculated) agree with curent grid points positions.
	Used to verify whether pre-calculated Green's functions were calculated on the same grid as used now.
	"""
	md5_crustal = hashlib.md5(open('green/crustal.dat', 'rb').read()).hexdigest()
	md5_station = hashlib.md5(open('green/station.dat', 'rb').read()).hexdigest()
	txt_soutype = open('green/soutype.dat').read().strip().replace('\n', '_')
	for g in range(len(self.grid.grid)):
		gp = self.grid.grid[g]
		point_id = str(g).zfill(4)
		try:
			meta  = open('green/elemse'+point_id+'.txt', 'r')
			lines = meta.readlines()
			meta.close()
		except:
			problem = True
		else:
			problem = False
			if len(lines)==0:
				self.grid.grid[g]['err'] = 1
				self.grid.grid[g]['VR'] = -10
			elif lines[0] != '{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(gp['x']/1e3, gp['y']/1e3, gp['z']/1e3, md5_crustal, md5_station, txt_soutype):
				problem = True
		if problem:
			l = lines[0].split()
			desc = 'Pre-calculated grid point {0:d} was calculated with different parameters. '.format(g)
			if l[0:3] != '{0:1.3f} {1:1.3f} {2:1.3f}'.format(gp['x']/1e3, gp['y']/1e3, gp['z']/1e3).split():
				desc += 'Its coordinates differs, probably the shape of the grid was changed. '
			if l[3] != md5_crustal:
				desc += 'File green/crustal.dat has different hash, probably crustal model was changed. '
			if l[4] != md5_station:
				desc += 'File green/station.dat has different hash, probably station set was different. '
			if l[5] != txt_soutype:
				desc += 'Source time function (file soutype.txt) was different. '
			self.log(desc)
			print(desc)
			return False
	return True

def calculate_or_verify_Green(self):
	"""
	If ``self.use_precalculated_Green`` is True, verifies whether the pre-calculated Green's functions were calculated on the same grid and with the same parameters (:func:`verify_Greens_headers` and :func:`verify_Greens_parameters`)
	Otherwise calculates Green's function (:func:`write_Greens_parameters` and :func:`calculate_Green`).
	"""
	
	if not self.use_precalculated_Green: # calculate Green's functions in all grid points
		self.write_Greens_parameters()
		self.calculate_Green()
	else: # verify whether the pre-calculated Green's functions are calculated on the same grid and with the same parameters
		differs = False
		if not self.verify_Greens_parameters():
			differs = True
		if not self.verify_Greens_headers():
			differs = True
		if differs:
			if self.use_precalculated_Green == 'auto':
				self.log('Shape or the grid or some parameters changed, calculating Gren\'s functions again...')
				self.write_Greens_parameters()
				self.calculate_Green()
			else:
				raise ValueError('Metadata of pre-calculated Green\'s functions differs from actual calculation. More details are shown above and in the log file.')

def calculate_Green(self):
	"""
	Runs :func:`Axitra_wrapper` (Green's function calculation) in parallel.
	"""
	grid = self.grid.grid
	logfile = self.d.outdir+'/log_green.txt'
	open(logfile, "w").close() # erase file contents
	# run `gr_xyz` aand `elemse`
	for model in self.d.models:
		if self.threads > 1: # parallel
			pool = mp.Pool(processes=self.threads)
			results = [pool.apply_async(Axitra_wrapper, args=(i, model, grid[i]['x'], grid[i]['y'], grid[i]['z'], self.npts_exp, self.elemse_start_origin, logfile)) for i in range(len(grid))]
			output = [p.get() for p in results]
			for i in range (len(grid)):
				if output[i] == False:
					grid[i]['err'] = 1
					grid[i]['VR'] = -10
		else: # serial
			for i in range (len(grid)):
				gp = grid[i]
				Axitra_wrapper(i, model, gp['x'], gp['y'], gp['z'], self.npts_exp, self.elemse_start_origin, logfile)

