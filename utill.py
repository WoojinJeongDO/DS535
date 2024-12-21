import sys
import random
import numpy as np
import torch
import os

def find_option(key, default=None):
	for a in sys.argv:
		if a.startswith(key+"="):
			return a[ len(key)+1 : ]
	return default


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
	
def set_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_dataset(dataset, file_path):
    """Dataset 저장 함수"""
    torch.save(dataset, file_path)

def load_dataset(file_path):
    """Dataset 로드 함수"""
    return torch.load(file_path)

def dataset_exists(file_path):
    """Dataset 파일 존재 여부 확인"""
    return os.path.exists(file_path)
