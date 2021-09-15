#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import hashlib

def Axitra_wrapper(i, model, x, y, z, npts_exp, elemse_start_origin, logfile='output/log_green.txt'):
	"""
	Evaluate Green's function using code ``Axitra`` (programs ``gr_xyz`` and ``elemse``) in a given grid point.
	
	:param i: number (identifier) of grid point
	:type i: integer
	:param model: identifier of crust model
	:type model: string
	:param x: source coordinate in N-S direction [m] (positive to the north)
	:type x: float
	:param y: source coordinate in E-W direction [m] (positive to the east)
	:type y: float
	:param z: source depth [m] (positive down)
	:type z: float
	:param npts_exp: the number of samples in the computation is :math:`2^{\mathrm{npts\_exp}}`
	:type npts_exp: integer
	:param elemse_start_origin: time between elementary seismogram start and elementary seismogram origin time
	:type elemse_start_origin: float
	:param logfile: path to text file, where are details about computation logged
	:type logfile: string, optional
	
	Remark: because of paralelisation, this wrapper cannot be part of class :class:`ISOLA`.
	"""
	iter_max = 10
	point_id = str(i).zfill(4)
	if model:
		point_id += '-' + model

	log = open(logfile, 'a')
	for iter in range(iter_max):
		process = subprocess.Popen(['./gr_xyz', '{0:1.3f}'.format(x/1e3), '{0:1.3f}'.format(y/1e3), '{0:1.3f}'.format(z/1e3), point_id, model], stdout=subprocess.PIPE, cwd='green') # spustit GR_XYZ
		out, err = process.communicate()
		if not out and not err:
			break
		else:
			if iter == iter_max-1:
				log.write('grid point {0:3d}, gr_xyz failed {1:2d} times, POINT SKIPPED\n'.format(i, iter))
				return False
	log.write('grid point {0:3d}, {1:2d} calculation(s)\n'.format(i, iter+1))
	process = subprocess.Popen(['./elemse', str(npts_exp), point_id, "{0:8.3f}".format(elemse_start_origin)], stdout=subprocess.PIPE, cwd='green') # spustit CONSHIFT
	out, err = process.communicate()
	if out or err:
		log.write('grid point {0:3d}: elemse FAILED\n'.format(i, iter))
		return False
	log.close()

	meta = open('green/elemse'+point_id+'.txt', 'w')
	# TODO add md5 sum of green/crustal.dat and green/station.dat
	# TODO add type and parameters of source time function
	md5_crustal = hashlib.md5(open('green/crustal.dat', 'rb').read()).hexdigest()
	md5_station = hashlib.md5(open('green/station.dat', 'rb').read()).hexdigest()
	txt_soutype = open('green/soutype.dat').read().strip().replace('\n', '_')
	meta.write('{0:1.3f} {1:1.3f} {2:1.3f} {3:s} {4:s} {5:s}'.format(x/1e3, y/1e3, z/1e3, md5_crustal, md5_station, txt_soutype))
	meta.close()

	return True

 
