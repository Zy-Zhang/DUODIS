import torch
import os
import pickle

from extraction.extracting import nms_info_dig
from utils.bbxs_generator import NMS_with_centrality, gnd_generator

def NMS(dataset, bbxs_init, gpu, iteration):
	file_nms = os.path.join('./data/nms', dataset, 'nms_info_{}.pkl'.format(iteration))
	if os.path.exists(file_nms):
		print('We have nms_info already.')
		nms_info = pickle.load(open(file_nms,'rb'))
	else:
		nms_info = nms_info_dig(dataset = dataset, bbxs_init = bbxs_init, gpu = gpu, iteration = iteration)
	bbxs_nms = NMS_with_centrality(nms_info)
	gnd_generator(dataset, bbxs_nms, iteration)
