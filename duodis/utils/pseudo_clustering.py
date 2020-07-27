import numpy as np
import pickle
import os

from datasets.testdataset import configdataset
from utils.general import get_data_root
from utils.diffusion import *

def pclass_definition(vecs, threshold, knn = 50, power = 3):
	# generate initial clusters with alpha-query expansion with knn x pow
	scores = np.dot(vecs.T, vecs)
	ranks = np.argsort(-scores, axis = 0)
	matrix = np.zeros((scores.shape))
	for idx in range(vecs.shape[1]):
		matrix[ranks[:knn,idx],idx] = scores[ranks[:knn,idx],idx]**power
	vecs_alpha = np.dot(vecs, matrix)
	scores_alpha = np.dot(vecs.T, vecs_alpha)
	ranks_alpha = np.argsort(-scores_alpha, axis = 0)
	clusters_init = {}
	for idx in range(ranks_alpha.shape[1]):
		index = np.argwhere(scores_alpha[:,idx] > threshold)
		if len(index) == 0:
			clusters_init.update({str(idx):[ranks_alpha[0,idx].tolist()]})
		else:
			clusters_init.update({str(idx):ranks_alpha[index,idx][:,0].tolist()})
		if (idx + 1) % 1000 == 0:
			print('\r pseudo_product: %d / %d ...' % (idx+1,ranks_alpha.shape[1]), end='')
	print('\r')
	# merge the overlapped clusters

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]

def pclass_definition_kr(vecs, dataset, iteration, knn = 20):
	# generate initial clusters with k-reciprocal nearest neighbors
	scores = np.dot(vecs.T, vecs)
	ranks = np.argsort(-scores, axis = 0)
	clusters_init = {}
	centre_ = np.zeros(vecs.shape)
	for idx in range(ranks.shape[1]):
		kr = []
		for N in ranks[:knn,idx]:
			if idx in ranks[:knn,N]:
				kr.append(N)
		clusters_init.update({str(idx):kr})
		vec = np.mean(vecs[:,kr], axis = 1);
		centre_[:,idx] = vec/(np.linalg.norm(vec) + 1e-6)
		if (idx + 1) % 1000 == 0:
			print('\r pseudo_product: %d / %d ...' % (idx+1,ranks.shape[1]), end='')
	print('\r')
	# merge clusters with enough overlapping
	sub_clusters = cluster_merge(centre_, 0.9)
	pseudo_classes = {}
	N = 0
	for key in sub_clusters.keys():
		First = True
		for X in sub_clusters[key]:
			if First:
				elements = np.asarray(clusters_init[str(X)])
				First = False
			else:
				elements = np.concatenate((elements, np.asarray(clusters_init[str(X)])),axis = 0)
		if len(elements) > 1:
			pseudo_classes.update({str(N):np.unique(elements).tolist()})
			N += 1
	print('We generate {} pseudo-classes in total.'.format(len(pseudo_classes.keys())))
	# calculate the centres of the clusters and assign relative pseudo-classes to each image
	centres = np.zeros((vecs.shape[0],len(pseudo_classes.keys())))
	for idx in range(len(pseudo_classes.keys())):
		vec = np.mean(vecs[:,pseudo_classes[str(idx)]], axis = 1)
		centres[:,idx] = vec / (np.linalg.norm(vec) + 1e-6)
	scores = np.dot(centres.T, vecs)
	ranks = np.argsort(-scores, axis = 0)
	pseudo_labels = {}
	for idx in range(scores.shape[1]):
		pseudo_labels.update({str(idx):ranks[:,idx].tolist()})
	print('We just finish assigning the pseudo-labels for each image.')
	pseudo_clusters = {}
	pseudo_clusters.update({'classes':pseudo_classes})
	pseudo_clusters.update({'labels':pseudo_labels})
	file_pc = os.path.join('data/pseudo_clusters',dataset,'pc_{}.pkl'.format(iteration))
	pickle.dump(pseudo_clusters,open(file_pc,'wb'))
	# return pseudo_clusters



def cluster_merge(centre, threshold, scores = None):
	if scores is None:
		scores = np.dot(centre.T, centre)
	matrix = np.zeros(scores.shape)
	anchors = np.argwhere(scores > threshold)
	for idx in range(anchors.shape[0]):
		matrix[anchors[idx,0],anchors[idx,1]] = 1
	pool_list = np.asarray(range(scores.shape[0])).tolist()
	merged_c = {}
	N = 0
	while pool_list:
		merge_list_ori = [pool_list[0]]
		Loop = True
		while Loop:
			idx = np.argwhere(matrix[:,merge_list_ori] == 1)
			merge_list_new = np.unique(idx[:,0]).tolist()
			if len(merge_list_ori) < len(merge_list_new):
				merge_list_ori = merge_list_new
			else:
				for num in merge_list_ori:
					pool_list.remove(num)
				merged_c.update({str(N):merge_list_ori})
				N += 1
				Loop = False
	return merged_c

def pclass_update(dataset, iteration, threshold = 0.6):
	file_pc = os.path.join('./data/pseudo_clusters', dataset, 'pc_0.pkl')
	pseudo_clusters = pickle.load(open(file_pc,'rb'))
	file_fe = os.path.join('./data/features', dataset, 'feas_{}.pkl'.format(iteration))
	feas = pickle.load(open(file_fe,'rb'))
	cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'), iteration)
	vec_labels = cfg['vec_labels']
	K = 100 # approx 50 mutual nns
	alpha = 0.9
	# find out all the regions belonging to a pclass
	classes = {}
	for key in pseudo_clusters['classes'].keys():
		anchors = []
		ind_region = np.asarray([]).astype(np.int32)
		for idx in pseudo_clusters['classes'][key]:
			index = np.argwhere(np.asarray(vec_labels) == idx)[:,0]
			ind_region = np.concatenate((ind_region,index), axis = 0)
		vecs_ = feas['vecs'][:,ind_region]
		Y = np.ones((1, vecs_.shape[1]))
		A = np.dot(vecs_.T, vecs_)
		W = sim_kernel(A).T
		W = topK_W(W, K)
		Wn = normalize_connection_graph(W)
		_, scores_ = cg_diffusion(Y, Wn, alpha)
		ranks_ = np.argsort(-scores_, axis = 0)
		scores = np.dot(feas['vecs'].T,feas['vecs'][:,ind_region[ranks_[0]]])
		ranks_ = np.argsort(-scores, axis = 0)
		for num in range(ranks_.shape[0]):
			if scores[ranks_[num]] >= threshold:
				anchors.append(ranks_[num])
			else:
				break
		classes.update({key:anchors})
		if (int(key) + 1) % 1000 == 0:
			print('\r>>>> {}/{} done...'.format((int(key)+1), len(pseudo_clusters['classes'].keys())), end=' ')
	print('')
	pseudo_clusters.update({'classes':classes})
	file_pc_new = os.path.join('./data/pseudo_clusters', dataset, 'pc_{}.pkl'.format(iteration))
	pickle.dump(pseudo_clusters,open(file_pc_new,'wb'))