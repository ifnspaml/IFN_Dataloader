#+++++++++++++++++++++++++++++++++++++++++++++++
#
# Author: Philipp Donn
#
# Date: 10/10/2019
#
# Supervisor: Marvin Klingner
# 
#+++++++++++++++++++++++++++++++++++++++++++++++
#
# Convert 3D colour ground truths to 2D id
# ground truths. Do yourself a favour and run it
# on a GPU.
#
#+++++++++++++++++++++++++++++++++++++++++++++++

import cv2
import torch
import os

from dataloader.definitions.labels_file import labels_mapillary_seg_cityscapes_def
import dataloader.file_io.get_path as gp

labels = labels_mapillary_seg_cityscapes_def.getlabels()
path_getter = gp.GetPath()
path = path_getter.get_data_path()
path_in = os.path.join(path, 'mapillary', 'train', 'Segmentation')
path_out = os.path.join(path, 'mapillary', 'segmentation_trainid', 'train', 'Segmentation')

if not os.path.isdir(path_out):
	os.makedirs(path_out)

device = torch.device("cuda")

files = os.listdir(path_in)

n_labels = len(labels)
label_color = torch.zeros(size=(n_labels, 3), dtype=torch.uint8)
label_id = torch.zeros(size=(n_labels, 1), dtype=torch.uint8)
i = 0

for label in labels:
	label_color[i, :] = torch.tensor(label.color, dtype=torch.uint8)
	label_id[i, 0] = label.trainId
	i += 1

label_color = label_color.to(device)
label_id = label_id.to(device)

for file in files:
	img = cv2.imread(os.path.join(path_in, file), -1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = torch.tensor(img).to(device)
	img_label = torch.zeros(size=(img.shape[:2]), dtype=torch.uint8).to(device)
	
	for i in range(n_labels):
		img_label = torch.where((img == label_color[i, :]).all(dim=2), label_id[i, 0], img_label)
	cv2.imwrite(os.path.join(path_out, file), img_label.cpu().numpy())
