import torch
import pickle
import os
import numpy as np
import time
import math

import torchvision
import torchvision.transforms as transforms

from datasets.genericdataset import ImagesFromList, FeasFromData
from networks.imageretrievalnet import CamNet, init_network, CamGenNet
from datasets.testdataset import configdataset


def cam_train(dataset, iteration, gpu):
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	# train the cam-model
	cfg = configdataset(dataset, os.path.join('data', 'test'), iteration)
	imlist = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
	if iteration > 0:
		bbxs = cfg['bbxs_db']
	else:
		bbxs = None
	labels = np.asarray(range(len(imlist)))
	num_class = len(np.unique(labels))
	print('We have {} samples in total.'.format(num_class))
	model_cam = model_trans(dataset = dataset, iteration = iteration, num_class = num_class)
	# print(model_cam)

	# load imlist and lables
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
	loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=imlist, imsize=1024, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )
	fea_gap = torch.zeros(512,len(loader))
	model_cam.cuda()
	file_gap = os.path.join('data', 'features', dataset, 'fea_gap_{}.pkl'.format(iteration))
	if os.path.exists(file_gap):
		print('We have pre-extracted the fea_gap...')
		fea_gap = pickle.load(open(file_gap,'rb'))
	else:
		for i, (inputs, _) in enumerate(loader):
			inputs = inputs.cuda()
			fea = model_cam.norm(model_cam.pool(model_cam.features(inputs))).cpu().data.squeeze()
			fea_gap[:,i] = fea
			# print('\r>>>> {}/{} done...'.format((i+1), len(loader)), end='')
		fea_gap = fea_gap.numpy().T
		pickle.dump(fea_gap, open(file_gap,'wb'))

	# load the pre-extracted gap-features to train cam-model
	loader_gap = torch.utils.data.DataLoader(FeasFromData(fea_gap, labels), batch_size = 4, shuffle = True)
	criterion = torch.nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	optimizer = torch.optim.SGD(model_cam.parameters(), lr = 0.3, momentum = 0.9)
	# optimizer = optim.Adam(model_cam.parameters(), lr=0.05, betas = (0.9,0.999))
	# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.3)

	model_cam = train_model(model_cam, loader_gap, criterion, optimizer, num_epochs = 1000, data_volume = len(imlist))
	file_mod = os.path.join('./models','model_cam_{}_{}.pth'.format(dataset,iteration))
	torch.save(model_cam.state_dict(), file_mod)


def model_trans(dataset, iteration, num_class, pretrained = True):
	state = torch.load('./models/retrievalSfM120k-vgg16-gem-b4dcdc6.pth')
	net_params = {}
	net_params['architecture'] = state['meta']['architecture']
	net_params['pooling'] = state['meta']['pooling']
	net_params['whitening'] = state['meta']['whitening']
	net_params['mean'] = state['meta']['mean']
	net_params['std'] = state['meta']['std']
	net_params['pretrained'] = False
	# initialize the model
	net = init_network(net_params)
	if pretrained:
		net.load_state_dict(state['state_dict'])
	net = init_network_cam(net, num_class)
	return net

def init_network_cam(net_init, num_class):
	features = list(net_init.features.children())
	pool = torch.nn.AdaptiveAvgPool2d(1)
	net = CamNet(features, pool, num_class)
	return net

def CamGen_trans(model):
	features = list(model.features.children())
	output_dim = model.fc.out_features
	convs = torch.nn.Conv2d(512,output_dim,kernel_size=(1,1),stride=(1,1))
	w = model.fc.weight.data
	w = w.view([w.size(0),w.size(1),1,1])
	convs.weight.data = w
	net = CamGenNet(features,convs)
	return net

def train_model(model, data_loader, criterion, optimizer, num_epochs, data_volume):
	since = time.time()
	for epoch in range(num_epochs):
		adjust_learning_rate(optimizer, epoch, num_epochs, 0.3)
		# scheduler.step()
		model.train()
		running_loss = 0.0
		running_corrects = 0
		# Iterate over data
		for inputs, labels in data_loader:
			inputs = inputs.cuda()
			labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model.fc(inputs)
			_, preds = torch.max(outputs,1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels)
		epoch_loss = running_loss / data_volume
		epoch_acc = running_corrects.double() / data_volume
		if (epoch + 1) % 300 == 0:
			print('\r Epoch: {}/{} '.format(epoch, num_epochs-1) + 'Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc),end='')
	time_elapsed = time.time() - since
	print(' ')
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	return model

def adjust_learning_rate(optimizer, epoch, epochs, lr, cos = True):
	"""Decay the learning rate based on schedule"""
	if cos:  # cosine lr schedule
		lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
	else:  # stepwise lr schedule
		for milestone in [300,600,900]:
			lr *= 0.1 if epoch >= milestone else 1.
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr