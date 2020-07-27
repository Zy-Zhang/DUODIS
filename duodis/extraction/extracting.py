import torch
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
import time

from torch.utils.model_zoo import load_url
from torchvision_ import transforms

from networks.imageretrievalnet import init_network, extract_vectors
from datasets.testdataset import configdataset
from utils.general import get_data_root
from utils.evaluate import compute_map_and_print, rank_trans
from utils.whiten import whitenlearn, whitenapply
from utils.diffusion import *

PRETRAINED = {
    'vgg16'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'resnet101'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

def feature_extract(dataset, gpu, iteration):
	# load the pre-trained model fine-tuned by Radenovic
	state = torch.load('./models/retrievalSfM120k-vgg16-gem-b4dcdc6.pth')
	# initialize all the settings of model
	net_params = {}
	net_params['architecture'] = state['meta']['architecture']
	net_params['pooling'] = state['meta']['pooling']
	net_params['whitening'] = state['meta']['whitening']
	net_params['mean'] = state['meta']['mean']
	net_params['std'] = state['meta']['std']
	net_params['pretrained'] = False
	# initialize the model
	net = init_network(net_params)
	net.load_state_dict(state['state_dict'])

	# if whitening is precomputed
	if 'Lw' in state['meta']:
		net.meta['Lw'] = state['meta']['Lw']

	print(">>>> loaded network: ")
	if iteration == 0:
		print(net.meta_repr())

	# setting up the multi-scale parameters
	ms = list(eval('[1, 1/2**(1/2), 1/2]'))
	if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['whitening']:
		msp = net.pool.p.data.tolist()[0]
		# msp = net.pool.p.item()
		print(">> Set-up multiscale:")
		print(">>>> ms: {}".format(ms))            
		print(">>>> msp: {}".format(msp))
	else:
		msp = 1

    # moving network to gpu and eval mode
	net.cuda()
	net.eval()

    # set up the transform
	normalize = transforms.Normalize(
		mean=net.meta['mean'],
		std=net.meta['std']
	)
	transform = transforms.Compose([
		transforms.ToTensor(),
		normalize
	])

	print('>> {}: Extracting...'.format(dataset))
    # prepare config structure for the test dataset
	cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'), iteration)
	images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
	qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
	try:
		bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
	except:
		bbxs = None  # for holidaysmanrot and copydays
	if iteration > 0:
		bbxs_db = cfg['bbxs_db']
	else:
		bbxs_db = None
    # extract database and query vectors
	print('>> {}: database images...'.format(dataset))
	vecs = extract_vectors(net, images, 1024, transform, bbxs = bbxs_db, ms=ms, msp=msp, print_freq=5000)
	print('>> {}: query images...'.format(dataset))
	qvecs = extract_vectors(net, qimages, 1024, transform, bbxs=bbxs, ms=ms, msp=msp)
    
	print('>> {}: Evaluating...'.format(dataset))

	vecs = vecs.numpy()
	qvecs = qvecs.numpy()
	# whitening
	Lw = net.meta['Lw']['retrieval-SfM-120k']['ms']
	vecs  = whitenapply(vecs, Lw['m'], Lw['P'])
	qvecs = whitenapply(qvecs, Lw['m'], Lw['P'])
	scores = np.dot(vecs.T, qvecs)
	# search, rank, and print
	if iteration > 0:
		ranks = rank_trans(scores,cfg['vec_labels'])
	else:
		ranks = np.argsort(-scores, axis=0)
	compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'])
	feas = {}
	feas.update({'vecs':vecs})
	feas.update({'qvecs':qvecs})
	file_fea = os.path.join('data/features',dataset,'feas_{}.pkl'.format(iteration))
	pickle.dump(feas,open(file_fea,'wb'))
	return feas

def nms_info_dig(dataset, bbxs_init, gpu, iteration):
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'), 0)
	imlist_ = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
	imlist = []
	bbxs = []
	ind_bbx_to_img = []
	for idx in bbxs_init.keys():
		for num in range(bbxs_init[idx].shape[0]):
			imlist.append(imlist_[int(idx)])
			bbxs.append(bbxs_init[idx][num,:])
			ind_bbx_to_img.append(int(idx))
	file_fea = os.path.join('./data/features',dataset,'vecs_nms_{}.pkl'.format(iteration))
	if os.path.exists(file_fea):
		vecs = pickle.load(open(file_fea,'rb'))
	else:
		state = torch.load('./models/retrievalSfM120k-vgg16-gem-b4dcdc6.pth')
		# initialize all the settings of model
		net_params = {}
		net_params['architecture'] = state['meta']['architecture']
		net_params['pooling'] = state['meta']['pooling']
		net_params['whitening'] = state['meta']['whitening']
		net_params['mean'] = state['meta']['mean']
		net_params['std'] = state['meta']['std']
		net_params['pretrained'] = False
		# initialize the model
		net = init_network(net_params)
		net.load_state_dict(state['state_dict'])

		# if whitening is precomputed
		if 'Lw' in state['meta']:
			net.meta['Lw'] = state['meta']['Lw']

		# setting up the multi-scale parameters
		ms = list(eval('[1, 1/2**(1/2), 1/2]'))
		msp = net.pool.p.data.tolist()[0]

    	# moving network to gpu and eval mode
		net.cuda()
		net.eval()

    	# set up the transform
		normalize = transforms.Normalize(
			mean=net.meta['mean'],
			std=net.meta['std']
		)
		transform = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])
		vecs = extract_vectors(net, imlist, 1024, transform, bbxs = bbxs, ms=ms, msp=msp, print_freq=10000)
		vecs = vecs.numpy()
		Lw = net.meta['Lw']['retrieval-SfM-120k']['ms']
		vecs  = whitenapply(vecs, Lw['m'], Lw['P'])
		vecs = vecs.astype(np.float16)
		pickle.dump(vecs,open(file_fea,'wb'))

	centrality = np.zeros((vecs.shape[1],1))
	K = 100 # approx 50 mutual nns
	QUERYKNN = 10
	R = 2000
	alpha = 0.9
	KM = 20
	print('Do KMeans to separate the whole data into {} partitions...'.format(KM))
	since = time.time()
	kmeans = KMeans(KM,random_state=0).fit(vecs.T)
	time_elapsed = time.time() - since
	print('Clustering is done in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	for num_c in range(KM):
		since = time.time()
		vecs_ = vecs[:,np.argwhere(kmeans.labels_ == num_c)[:,0]]
		# vecs_ = vecs[:,index]
		Y = np.ones((1, vecs_.shape[1]))
		# sortidxs = np.argsort(-Y, axis = 1)
		# for i in range(len(Y)):
			# Y[i,sortidxs[i,QUERYKNN:]] = 0
		# Y = sim_kernel(Y)
		A = np.dot(vecs_.T, vecs_)
		W = sim_kernel(A).T
		W = topK_W(W, K)
		Wn = normalize_connection_graph(W)
		_, scores_ = cg_diffusion(Y, Wn, alpha)
		# _, scores_ = fsr_rankR(Y, Wn, alpha, R)
		centrality[np.argwhere(kmeans.labels_ == num_c)[:,0],:] = scores_# .reshape(scores_.shape[0],1)
		# centrality[index,:] = scores_
		time_elapsed = time.time() - since
		# print('The {}-th centraltiy is done in {:.0f}m {:.0f}s ...'.format(num_c, time_elapsed // 60, time_elapsed % 60))
	nms_info = []
	for idx in bbxs_init.keys():
		info = {}
		info.update({'bbxs':bbxs_init[idx]})
		info.update({'vecs':vecs[:,np.argwhere(np.asarray(ind_bbx_to_img) == int(idx))[:,0]]})
		info.update({'centrality':centrality[np.argwhere(np.asarray(ind_bbx_to_img) == int(idx))[:,0],0]})
		nms_info.append(info)
	file_nms = os.path.join('./data/nms',dataset,'nms_info_{}.pkl'.format(iteration))
	pickle.dump(nms_info,open(file_nms,'wb'))
	return nms_info