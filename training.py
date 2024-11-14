import datetime
import os
import sys
import math
import numpy as np
import pandas as pd
from itertools import product

from algo_common import *
from utill import *



class Trainer:
	def __init__(self,config):
		self.config = config
		self.years = config.years
		self.feature_list = config.feature_list
		
	def prepare(self,years,pid,algs):
		year = 0
		pass
	
    def train(self, algos, pid, instance):
        start_time = datetime.datetime.now()
        last_report_timestamp = datetime.datetime.now()





class Config:
	"""
	Class for parsing configuration file
	"""

	def __init__(self, fn):
		self.options = { "algorithms":[] }

		fd = open(fn)
		category = "DEFAULT"
		for line in fd:
			line = line.strip()
			if line.startswith("#"):
				continue
			elif line.endswith(":"):
				category = line[:-1].strip()
			elif line.startswith("algorithm"):
				self.options[ "algorithms" ].append( ( category, line.split("=")[1].strip() ) )
			elif line.strip() == "":
				continue
			else:
				line = line.split("=")
				self.options[ line[0].strip() ] = eval( line[1] )
		fd.close()

	def __getattr__(self, key):
		if key in self.options:
			return self.options[key]
		return None


def find_option(key, default=None):
	for a in sys.argv:
		if a.startswith(key+"="):
			return a[ len(key)+1 : ]
	return default

if __name__ == "__main__":
    # 1. load configuration file
    device = find_option("cuda","0")
    config_name = find_option("config")
    if config_name.endswith(".config"):
        config_name = config_name[:-len(".config")]

    config = Config('configs/'+config_name+'.config')

    
    # 2. setup algorithms
    algs = []
    for cat, cmd in config.algorithms:
        exec( "import " + cmd[ :cmd.find(".") ] )
        module = cmd[:-1] + ", config, device )"
        alg = eval( module )
        algs.append( alg )

    if len(algs) == 0:
        raise Exception( "no algorithm is configured" )
    
    
	# 3. simulate
    pid = find_option( "pid", "0" )
    
    train = Trainer( config )

    # Training
    if "training" in config_name:
        model_dir = config.instance
        if model_dir not in os.listdir( "models" ):
            os.mkdir( "models/"+model_dir )
        if algs[0].name not in os.listdir("models/"+model_dir):
            os.mkdir( "models/"+model_dir+'/'+algs[0].name )
        train.train(algs,pid,config.years)