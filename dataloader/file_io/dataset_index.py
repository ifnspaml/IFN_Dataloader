import os
import sys
import numpy as np
import json

import dataloader.file_io.get_path as gp

"""
The following paramters have to be present in every dataset definition:
- 'K': Extrinsic camera matrix as a Numpy array. If not available, take None
- 'stereo_T': Distance between the two cameras (see e.g. http://www.cvlibs.net/datasets/kitti/setup.php, 0.54m)
- 'labels_mode': 'fromid' or 'fromrgb', depending on which format the segmentation images have
- 'depth_mode': 'uint_16' or 'uint_16_subtract_one' depending on which format the depth images have
- 'splits': List of splits that are available for this dataset
"""

dataset_index = {
    'a2d2':
        {'K': None,
         'stereo_T': None,
         'labels': 'a2d2',
         'labels_mode': 'fromrgb',
         'depth_mode': None,
         'flow_mode': None,
         'splits': ('andreas_split',)
         },
    'bdd100k':
        {'K': None,
         'stereo_T': None,
         'labels': 'bdd100k',
         'labels_mode': 'fromtrainid',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'camvid':
        {'K': None,
         'stereo_T': None,
         'labels': 'camvid',
         'labels_mode': 'fromrgb',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'cityscapes':
        {'K': [[1.10, 0, 0.5, 0],
               [0, 2.21, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.22,
         'labels': 'cityscapes',
         'labels_mode': 'fromid',
         'depth_mode': 'uint_16_subtract_one',
         'flow_mode': None,
         'splits': None
         },
    'cityscapes_demo_video':
        {'K': [[1.10, 0, 0.5, 0],
               [0, 2.21, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.22,
         'labels': 'cityscapes',
         'labels_mode': 'fromid',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'cityscapes_extra':
        {'K': [[1.10, 0, 0.5, 0],
               [0, 2.21, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.22,
         'labels': 'cityscapes',
         'labels_mode': 'fromid',
         'depth_mode': 'uint_16_subtract_one',
         'flow_mode': None,
         'splits': None
         },
    'cityscapes_sequence':
        {'K': [[1.10, 0, 0.5, 0],
               [0, 2.21, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.22,
         'labels': 'cityscapes',
         'labels_mode': 'fromid',
         'depth_mode': 'uint_16_subtract_one',
         'flow_mode': None,
         'splits': None
         },
    'cityscapes_video':
        {'K': [[1.10, 0, 0.5, 0],
               [0, 2.21, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.22,
         'labels': 'cityscapes',
         'labels_mode': None,
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'gta5':
        {'K': None,
         'stereo_T': None,
         'labels': 'gta5',
         'labels_mode': 'fromrgb',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'kitti':
        {'K': [[0.58, 0, 0.5, 0],
               [0, 1.92, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.54,
         'labels': 'kitti',
         'labels_mode': 'fromid',
         'depth_mode': 'uint_16',
         'flow_mode': None,
         'splits': ('eigen_split', 'zhou_split', 'zhou_split_left', 'zhou_split_right', 'benchmark_split',
                    'kitti_split', 'odom10_split', 'odom09_split', 'video_prediction_split', 'optical_flow_split'),
         },
    'kitti_2012':
        {'K': [[0.58, 0, 0.5, 0],
               [0, 1.92, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.54,
         'labels': 'kitti',
         'labels_mode': 'fromid',
         'depth_mode': 'uint_16',
         'flow_mode': None,
         'splits': None
         },
    'kitti_2015':
        {'K': [[0.58, 0, 0.5, 0],
               [0, 1.92, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.54,
         'labels': 'kitti',
         'labels_mode': 'fromid',
         'depth_mode': 'uint_16',
         'flow_mode': 'kitti',
         'splits': None
         },
    'lostandfound':
        {'K': None,
         'stereo_T': None,
         'labels': 'lostandfound',
         'labels_mode': 'fromid',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'make3d':
        {'K': None,
         'stereo_T': None,
         'labels': None,
         'labels_mode': None,
         'depth_mode': 'uint_16',
         'flow_mode': None,
         'splits': None
         },
    'mapillary':
        {'K': None,
         'stereo_T': None,
         'labels': 'mapillary',
         'labels_mode': 'fromrgb',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
    'synthia':
        {'K': None,
         'stereo_T': None,
         'labels': 'synthia',
         'labels_mode': 'fromid',
         'depth_mode': 'normalized_100',
         'flow_mode': None,
         'splits': None
         },
    'virtual_kitti':
        {'K': [[0.58, 0, 0.5, 0],
               [0, 1.92, 0.5, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]],
         'stereo_T': 0.54,
         'labels': 'virtual_kitti',
         'labels_mode': 'fromrgb',
         'depth_mode': 'normalized_100',
         'flow_mode': None,
         'splits': ('full_split',)
        },
    'voc2012':
        {'K': None,
         'stereo_T': None,
         'labels': None,
         'labels_mode': 'fromtrainid',
         'depth_mode': None,
         'flow_mode': None,
         'splits': None
         },
}


def create_parameter_files(datasets=None):
    if datasets is None:
        datasets = dataset_index.keys()
        parameters = dataset_index.values()
    else:
        if type(datasets) == str:
            datasets = [datasets]
        parameters = []
        for set in datasets:
            assert set in dataset_index.keys(), '{} is not a valid dataset'.format(set)
            parameters.append(dataset_index[set])
    path_getter = gp.GetPath()
    data_path = path_getter.get_data_path()
    for dataset, param in zip(datasets, parameters):
        dataset_path = os.path.join(data_path, dataset)
        if os.path.isdir(dataset_path):
            dump_location = os.path.join(dataset_path, 'parameters.json')
            with open(dump_location, 'w') as fp:
                json.dump(param, fp)
            print("{}: OK".format(dataset))
        else:
            print("{}: not found".format(dataset))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = None
    create_parameter_files(datasets)


