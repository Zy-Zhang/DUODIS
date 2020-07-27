import pickle
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from utils.general import get_data_root
from utils.bbxs_generator import BBXs, nms
from networks.camnet import model_trans, CamGen_trans
from datasets.genericdataset import ImagesFromList
from datasets.testdataset import configdataset

def camgenerator(dataset, iteration, num_pc, gpu):
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu
	# generate the cam-maps, bbxs
	print('Now the camgenerator starts to work ...')
	# set up the dataset list
	cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'), 0)
	images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
	cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'), iteration)
	imlist = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
	bbxs = None
	# set up the pre-process and loader
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
	loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=None, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )
    # initialize the model
	net = model_trans(dataset, iteration, len(imlist), pretrained = False)
	file_mod = os.path.join('./models','model_cam_{}_{}.pth'.format(dataset,iteration))
	# state = torch.load(file_mod)
	net.load_state_dict(torch.load(file_mod))
	model = CamGen_trans(net)
	model.cuda()
	model.eval()

	# start to 
	bbxs_ = {}

	################from here are all for the test
	file_pc = os.path.join('./data/pseudo_clusters', dataset, 'pc_{}.pkl'.format(iteration))
	pseudo_clusters = pickle.load(open(file_pc,'rb'))
	P_classes = pseudo_clusters['classes']
	P_labels = pseudo_clusters['labels']
	if num_pc > len(P_labels['0']):
		print('The chosen number of pseudo-label is larger than the initialized length!')
		num_pl = len(P_labels['0'])
	for image_idx, (inputs, labels) in enumerate(loader):
		idx_anchor = P_labels[str(image_idx)]
		inputs = inputs.cuda()
		heat_maps = model(inputs)
		bbxs_final = np.array([0,0,heat_maps.size(3)-1,heat_maps.size(2)-1])
		# num_bbx = [0]
		for index_cls in range(num_pc):
			idx_weight_choice = P_classes[str(idx_anchor[index_cls])]
			if len(idx_weight_choice) < 3:
				continue
			mask = heat_maps[:,idx_weight_choice,:,:].sum(1).squeeze_(0)
			mask = mask.cpu().data.numpy().reshape(heat_maps.size(2),heat_maps.size(3))
			if np.max(mask) == 0.0:
				continue
			mask = mask - np.min(mask)
			mask = mask / np.max(mask)
			cam = mask.copy()
			bbxs = BBXs(cam, density_thresh = 0.4)
			if bbxs[2] > 0:
				# for XX in range(np.floor_divide(len(bbxs),4)):
					# num_bbx.append(idx_anchor[index_cls] + 1)
				bbxs_final = np.concatenate((bbxs_final,bbxs),axis=0)
		bbxs_final_ = bbxs_final.reshape((-1,4))
		# num_bbx = np.asarray(num_bbx)
		# num_bbx = num_bbx.reshape((-1,1))

		bbxs_final_[:,2] = bbxs_final_[:,2] + 1
		bbxs_final_[:,3] = bbxs_final_[:,3] + 1
		bbxs_final_ = ((bbxs_final_*16)).astype('int')
		# bbxs_final_ = np.concatenate((bbxs_final_,num_bbx),axis = 1)
		bbxs_final_ = nms(bbxs_final_)
		bbxs_.update({str(image_idx):bbxs_final_})
		if (image_idx + 1) % 1000 == 0:
			print('\r >>>> {}/{}'.format((image_idx+1),len(images)) + '...',end='')
		del heat_maps

	pickle.dump(bbxs_,open(os.path.join('./data/bbxs',dataset,'bbxs_{}.pkl'.format(iteration)),'wb'))
	print('\r The cam generation is done.')
	return bbxs_
