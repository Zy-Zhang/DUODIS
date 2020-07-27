import argparse
import numpy as np 
import os
import pickle

import torch
import torchvision

from extraction.extracting import feature_extract, nms_info_dig
from utils.pseudo_clustering import pclass_definition, pclass_definition_kr, pclass_update
from networks.camnet import cam_train
from discovery.cam_generation import camgenerator
from utils.nms_process import NMS


parser = argparse.ArgumentParser(description='PyTorch implementation for DUODIS')

parser.add_argument('--dataset', '-d', metavar='DATASETS', default='roxford5k', help='dataset')
parser.add_argument('--iter', metavar='INTER',default=5, help='interation for self-boosting')
parser.add_argument('--gpu-id', '-g', default='0', metavar='N', help="gpu id used for testing (default: '0')")

def main():
	args = parser.parse_args()
	for iteration in range(args.iter):
		print('Now we start to process {}-th iteration.........................................................'.format(iteration))
		# extract the features with a pre-defined model:
		file_fea = os.path.join('./data','features',args.dataset,'feas_{}.pkl'.format(iteration))
		if os.path.exists(file_fea):
			feas = pickle.load(open(file_fea,'rb'))
		else:
			print('Extracting features...')
			feas = feature_extract(dataset = args.dataset, gpu = args.gpu_id, iteration = iteration)
			if iteration == args.iter-1:
				break
		# generate / update the definition of pseudo-classes
		# pclass_definition(feas['vecs'], 0.7)
		file_pc = os.path.join('./data', 'pseudo_clusters', args.dataset, 'pc_{}.pkl'.format(iteration))
		if os.path.exists(file_pc):
			print('The pseudo_clusters has been done already.')
		else:
			if iteration > 0:
				print('Pseudo_class updating...')
				pclass_update(dataset = args.dataset, iteration = iteration)
			else:
				print('Pseudo_class defining...')
				pclass_definition_kr(vecs = feas['vecs'], dataset = args.dataset, iteration = iteration)
		# train the CAM model for further object discovery
		file_mod = os.path.join('./models','model_cam_{}_{}.pth'.format(args.dataset, iteration))
		if os.path.exists(file_mod):
			print('The cam model has been pre-trained.')
		else:
			cam_train(dataset = args.dataset, iteration = iteration, gpu = args.gpu_id)
		# generate cam heatmap and discover objects
		file_bbx_init = os.path.join('./data/bbxs', args.dataset, 'bbxs_{}.pkl'.format(iteration))
		if os.path.exists(file_bbx_init):
			print('We have stored bbxs_init already.')
			bbxs_init = pickle.load(open(file_bbx_init,'rb'))
		else:
			bbxs_init = camgenerator(dataset = args.dataset, iteration = iteration, num_pc = 50, gpu = args.gpu_id)
		# do the NMS to obtain final bbxs
		file_bbx_nms = os.path.join('./data/test', args.dataset, 'gnd_{}_{}.pkl'.format(args.dataset, iteration+1))
		if os.path.exists(file_bbx_nms):
			print('We have obtained the final bbxs and generated new gnd.')
		else:
			NMS(dataset = args.dataset, bbxs_init = bbxs_init, gpu = args.gpu_id, iteration = iteration)

if __name__ == '__main__':
    main()
