import numpy as np
import os
import pickle

from utils.pseudo_clustering import cluster_merge
from datasets.testdataset import configdataset
from utils.general import get_data_root

def BBXs(cam_, density_thresh = 0.4, num_region = 1):
	# cam: a array with w x h x 1
	cam = cam_.copy()
	bbxs = np.array([0,0,0,0])
	bbxs_count = 0
	for num in range(num_region):
		index_max = np.argmax(cam_)
		start_y = np.ceil((index_max + 1) / cam.shape[1]).astype(int)-1
		start_x = (index_max + 1) % cam.shape[1] - 1
		start_x.astype(int)
		if start_x == -1:
			start_x = cam.shape[1] - 1
		bbx = np.array([start_x ,start_y ,start_x ,start_y ]).astype(int)
		loop = True
		Th = 0.01;
		while loop:
			if bbx[0] < 1 or np.dot(cam[bbx[1]:bbx[3]+1,bbx[0]].T,cam[bbx[1]:bbx[3]+1,bbx[0]-1]) < Th:
				L = -1
			else:
				L = np.mean(cam[bbx[1]:bbx[3]+1,bbx[0]-1])
			if bbx[2] > cam.shape[1]-2 or np.dot(cam[bbx[1]:bbx[3]+1,bbx[2]].T,cam[bbx[1]:bbx[3]+1,bbx[2]+1]) < Th:
				R = -1
			else:
				R = np.mean(cam[bbx[1]:bbx[3]+1,bbx[2]+1])
			if bbx[1] < 1 or np.dot(cam[bbx[1],bbx[0]:bbx[2]+1].T,cam[bbx[1]-1,bbx[0]:bbx[2]+1]) < Th:
				T = -1
			else:
				T = np.mean(cam[bbx[1]-1,bbx[0]:bbx[2]+1])
			if bbx[3] > cam.shape[0]-2 or np.dot(cam[bbx[3],bbx[0]:bbx[2]+1].T,cam[bbx[3]+1,bbx[0]:bbx[2]+1]) < Th:
				B = -1
			else:
				B = np.mean(cam[bbx[3]+1,bbx[0]:bbx[2]+1])
			side_power = np.array([L,T,R,B])
			side_choice = np.argmax(side_power)
			if side_power[side_choice] == -1:
				break
			if side_choice == 0:
				bbx[0] = max(0,bbx[0]-1)
			elif side_choice == 1:
				bbx[1] = max(0,bbx[1]-1)
			elif side_choice == 2:
				bbx[2] = min(cam.shape[1]-1,bbx[2]+1)
			else:
				bbx[3] = min(cam.shape[0]-1,bbx[3]+1)		
			x = np.where(cam[bbx[1]:bbx[3]+1,bbx[0]:bbx[2]+1] == 0)
			num_zeros = len(x[0])
			area_bbx = (bbx[3]-bbx[1]+1)*(bbx[2]-bbx[0]+1)
			if np.sum(cam[bbx[1]:bbx[3]+1,bbx[0]:bbx[2]+1])/(area_bbx - num_zeros) < density_thresh:
				# print(np.mean(cam[bbx[1]:bbx[3]+1,bbx[0]:bbx[2]+1]))
				loop = False
		if num_region > 1:
			cam_[bbx[1]:bbx[3]+1, bbx[0]:bbx[2]+1] = 0
			# cam[start_y:start_y+1, start_x:start_x+1] = 0.0
		if bbx[3]-bbx[1] > 2 and bbx[2]-bbx[0] > 2 and (bbx[3]-bbx[1]+1)*(bbx[2]-bbx[0]+1) <= cam.shape[0]*cam.shape[1] and max(bbx[3]-bbx[1]+1,bbx[2]-bbx[0]+1)/min(bbx[3]-bbx[1]+1,bbx[2]-bbx[0]+1) <= 4: 
			if bbxs_count == 0:
				bbxs = bbx
			else:
				bbxs = np.concatenate((bbxs,bbx),axis=0)
			bbxs_count += 1
	bbxs.astype(int)
	return bbxs

def nms(bbxs, overlap_thresh=0.9):
	BBX = bbxs[1:,:]
	IoU_M = IoU_Matrix(bbxs[1:,:])
	clusters = cluster_merge(centre = None, threshold = overlap_thresh, scores = IoU_M)
	bbxs_nms = bbxs[0,:]
	for idx in clusters.keys():
		bbx = np.round(np.mean(BBX[clusters[idx],:], axis = 0))
		if bbx[2] - bbx[0] > 20 and bbx[3] - bbx[1] > 20:
			bbxs_nms = np.concatenate((bbxs_nms, bbx), axis = 0)
	return bbxs_nms.reshape((-1,4)).astype(int)

def IoU_Matrix(bbxs):
	IoU_M = np.zeros((bbxs.shape[0],bbxs.shape[0]))
	for idx in range(bbxs.shape[0]):
		anchor = bbxs[idx,:]
		W = np.maximum(0,np.minimum(anchor[2],bbxs[:,2]) - np.maximum(anchor[0],bbxs[:,0]))
		H = np.maximum(0,np.minimum(anchor[3],bbxs[:,3]) - np.maximum(anchor[1],bbxs[:,1]))
		area_anchor = (anchor[2]-anchor[0])*(anchor[3]-anchor[1])
		area_pool = np.multiply(bbxs[:,2]-bbxs[:,0],bbxs[:,3]-bbxs[:,1])
		area = np.multiply(W,H)
		overlap = area / (area_anchor + area_pool - area)
		IoU_M[:,idx] = overlap
	return IoU_M

def NMS_with_centrality(nms_info, threshold_0 = 0.95, threshold_1 = 0.90):
	bbxs_ms = []
	for num_img in range(len(nms_info)):
		bbxs = nms_info[num_img]['bbxs']
		if bbxs.shape[0] == 1:
			bbxs_ = bbxs
		else:
			centrality = nms_info[num_img]['centrality'][1:]
			vecs = nms_info[num_img]['vecs']
			bbxs_ = bbxs[0,:].reshape(-1,4)
			bbxs_m = bbxs[1:,:]
			M_iou = IoU_Matrix(bbxs_m)
			M_sim = np.dot(vecs[:,1:].T, vecs[:,1:])
			rank_c = np.argsort(-centrality,axis = 0)
			index = rank_c[0]
			keep_search = True
			search_scale = np.asarray(range(bbxs_m.shape[0])).tolist()
			while keep_search:
				A = np.argwhere(M_iou[index,search_scale] >= threshold_1)[:,0]
				B = np.argwhere(M_sim[index, search_scale] >= 0.8)[:,0]
				ind_cut = np.unique(np.concatenate((np.asarray(search_scale)[A.tolist()], np.asarray(search_scale)[B.tolist()]), axis = 0)).tolist()
				ind_merge = np.argwhere(M_iou[index, search_scale] >= threshold_0)[:,0].tolist()
				if len(ind_merge) == 0:
					print('No ind_merge is found!!!')
				BBX = np.round(np.mean(bbxs_m[np.asarray(search_scale)[ind_merge].tolist(),:], axis = 0)).reshape(-1,4)
				if BBX[0,2] - BBX[0,0] > 20 and BBX[0,3] - BBX[0,1] > 20:
					bbxs_ = np.concatenate((bbxs_,BBX),axis = 0)
				for num in ind_cut:
					search_scale.remove(num)
				if len(search_scale) == 1:
					bbxs_ = np.concatenate((bbxs_, bbxs_m[search_scale,:]), axis = 0)
					break
				elif len(search_scale) == 0:
					break
				else:
					rank_c = np.argsort(-centrality[search_scale], axis = 0)
					index = search_scale[rank_c[0]]
		bbxs_ms.append(bbxs_)
		if (num_img + 1) % 1000 == 0:
			print('\r>>>> {}/{} done...'.format((num_img+1), len(nms_info)), end=' ')
	print('')
	return bbxs_ms

def gnd_generator(dataset, bbxs_ms, iteration):
	cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'), 0)
	imlist = []
	bbxs_db = []
	vec_labels = []
	for num in range(len(bbxs_ms)):
		for X in range(bbxs_ms[num].shape[0]):
			imlist.append(cfg['imlist'][num])
			bbxs_db.append(bbxs_ms[num][X,:])
			vec_labels.append(num)
	cfg.update({'imlist':imlist})
	cfg.update({'bbxs_db':bbxs_db})
	cfg.update({'vec_labels':vec_labels})
	file_gnd = os.path.join('./data/test',dataset, 'gnd_{}_{}.pkl'.format(dataset, iteration+1))
	pickle.dump(cfg,open(file_gnd,'wb'))
