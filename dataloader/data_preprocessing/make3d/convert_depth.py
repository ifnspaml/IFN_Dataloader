import os
import sys
import numpy as np
import cv2
import PIL.Image as pil
import scipy.io as sio

sys.path.append("../../../")
import dataloader.file_io.dir_lister as dl
import dataloader.file_io.get_path as gp
"""
The depth information for the make3d dataset is stored in .mat files. This script will convert them into PNG files
containing the depth in cm as a uint16 number. The dimensions will be scaled to match the dimensions of the color image.
"""

# Information from the dataset readme:
# 1) Train400Depth.tgz
# 	Laser Range data with Ray Position
# 	Data Format: Position3DGrid (55x305x4)
# 		Position3DGrid(:,:,1) is Vertical axis in meters (Y)
# 		Position3DGrid(:,:,2) is Horizontal axis in meters (X)
# 		Position3DGrid(:,:,3) is Projective Depths in meters (Z)
# 		Position3DGrid(:,:,4) is Depths in meters (d)
#
# 2) Train400Img.tar.gz
# 	Images all in resolution 2272x1704

for split in ('train', 'test'):
    path_getter = gp.GetPath()
    path = path_getter.get_data_path()
    data_path = os.path.join(path, 'make3d', split, 'Depth')
    out_path = os.path.join(path, 'make3d', split, 'Depth_PNG')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    filelist = dl.DirLister.get_files_by_ending(data_path, '.mat')
    print("{}: {} files found".format(split, len(filelist)))
    for file in filelist:
        filename = os.path.split(file)[-1]
        filename = os.path.splitext(filename)[0]
        print(filename)
        depth_data = sio.loadmat(file, verify_compressed_data_integrity=False)['Position3DGrid']
        # We use the 4th channel where the depth in meters is stored
        depth_data = depth_data[:, :, 3]
        mini = np.amin(depth_data)
        maxi = np.amax(depth_data)
        print('Minimum: {}, Maximum: {}'.format(mini, maxi))
        for i in range(depth_data.shape[0]):
            # Set invalid pixels to zero
            for j in range(depth_data.shape[1]):
                if depth_data[i, j] > 81.9:
                    depth_data[i, j] = 0
        depth_data = pil.fromarray(depth_data)
        depth_data = depth_data.resize((1704, 2272), pil.NEAREST)
        depth_data = (np.array(depth_data) * 256).astype(np.uint16)
        cv2.imwrite(os.path.join(out_path, filename+'.png'), depth_data)