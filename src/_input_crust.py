#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil

def read_crust(self, source, output='green/crustal.dat'):
	"""
	Copy a file or files with crustal model definition to location where code ``Axitra`` expects it
	
	:param source: path to crust file
	:type source: string
	:param output: path to copy target
	:type output: string, optional
	"""
	inputs = []
	for model in self.models:
		if model:
			inp  = source[0:source.rfind('.')] + '-' + model + source[source.rfind('.'):]
			outp = output[0:output.rfind('.')] + '-' + model + output[output.rfind('.'):]
		else:
			inp  = source
			outp = output
		shutil.copyfile(inp, outp)
		inputs.append(inp)
	self.log('Crustal model(s): '+', '.join(inputs))
	self.logtext['crust'] = ', '.join(inputs)
