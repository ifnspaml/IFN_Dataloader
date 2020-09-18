import os
import sys
import json
import random
import pandas as pd
from PIL import Image
import scipy.io as sio

import dataloader.file_io.dir_lister as dl
import dataloader.file_io.get_path as gp

SUPPORTED_DATASETS = ('cityscapes', 'cityscapes_video', 'cityscapes_sequence', 'cityscapes_extra', 'cityscapes_part',
                      'kitti', 'kitti_2012', 'kitti_2015', 'virtual_kitti', 'mapillary', 'mapillary_by_ID', 'gta5',
                      'synthia', 'bdd100k', 'voc2012', 'a2d2', 'lostandfound', 'camvid', 'make3d')

class SplitCreator:
    """Class to create the train, validation and test splits for the single datasets.

    This class can use different reading modes to create the train, validation and test split for a number of datasets.
    It is necessary that a basic_files.json has already been created for the dataset, i.e. the FilelistCreator has to
    be executed first on the dataset.
    """

    def __init__(self, dataset, path=None):
        """Initializes the split creator class, mainly defines which dataset is supposed to be used.

        :param dataset: name of the dataset folder
        :param path: makes it possible to self-define a path (not recommended)
        """
        assert dataset in SUPPORTED_DATASETS, 'Dataset not supported'
        self.dataset = dataset
        if path:
            self.dataset_folder_path = os.path.join(path, dataset)
        else:
            path_getter = gp.GetPath()
            self.dataset_folder_path = os.path.join(path_getter.get_data_path(), dataset)
        assert os.path.isdir(self.dataset_folder_path), 'Path to dataset does not exist'
        self.output_path = None

        self.filename = 'basic_files' + '.json'
        self.json_data = None
        self.new_json_data = {}

    def get_all_items(self):
        """Read the json data which is stored in the basic_files.json inside the dataset folder. """
        json_file = os.path.join(self.dataset_folder_path, self.filename)
        with open(json_file) as file:
            self.json_data = json.load(file)

    def set_split_path(self, split=None):
        """Define a path to the split folder where the split information is stored.

        :param split: Name of the split. It is added as an appendix to the dataset folder name, separated by an
            underscore. If this parameter is not set, the dataset folder will be used.
        """
        if split:
            root, dataset_name = os.path.split(self.dataset_folder_path)
            dataset_name = dataset_name + '_' + split
            self.output_path = os.path.join(root, dataset_name)
        else:
            self.output_path = self.dataset_folder_path

    def get_split_data(self, read_mode='folders', path_to_split_info = None, val_equal_test=False,
                       train_filter = 'train', val_filter='validation', test_filter='test', rand_split=None,
                       rand_seed=None):
        '''Get the data of each split subset (train, validation and test).

        There are several ways in which the split information can be available.
            -'folders': The dataset contains distinct folders for training, validation and test data.
            -'textfile': For each split subset, there is a textfile containing the names of all files belonging to it.
            -'matfile': There is a .mat file containing a mapping of file number to the split subsets.
            -'random_folders': Creates a new random split based on a deterministic random seed.
            -'kittifiles': Similar to 'textfile', adapted to the KITTI dataset.
            -'alltrain': The whole dataset will be used as training data.
        The split information is stored in the dictionary self.new_json_data, a class variable.

        :param read_mode: Defines how the splits are determined. Can be 'folders', 'kittifiles', 'matfile', 'textfile',
            'random_folders' or 'alltrain'
        :param path_to_split_info: For read_mode = 'textfile', location of the text files.
        :param val_equal_test: If True, the validation files will equal the test files.
        :param train_filter: String that is used to recognize the train files.
        :param val_filter: String that is used to recognize the validation files.
        :param test_filter: String that is used to recognize the test files.
        :param rand_split: For random mode: 3-tuple containing the split sizes in the format (train, validation, test).
        :param rand_seed: Seed used for generating the random split.
        '''
        assert read_mode in ('folders', 'kittifiles', 'matfile', 'textfile', 'alltrain', 'random_folders'), \
            'read_mode is not supported'

        if read_mode == 'folders':
            self._create_splits_from_folder(train_filter, val_filter, test_filter, val_equal_test)
        elif read_mode == 'kittifiles':
            self._create_splits_from_kitti_list()
        elif read_mode == 'matfile':
            self._create_splits_from_mat_file()
        elif read_mode == 'textfile':
            assert path_to_split_info is not None, 'Please enter the location of the text files with the split ' \
                                                   'information'
            self._create_splits_from_text_file(train_filter, val_filter, test_filter, path_to_split_info)
        elif read_mode == 'random_folders':
            assert type(rand_split) == tuple and len(rand_split) == 3
            assert rand_seed is not None
            self._create_random_split(rand_split, rand_seed)
        # Mode if the whole dataset is only used for training
        elif read_mode == 'alltrain':
            split = 'train'
            new_data = {}
            new_data.update({'names': self.json_data['names']})
            new_data.update({'types': self.json_data['types']})
            new_data.update({'files': self.json_data['files']})
            new_data.update({'folders': self.json_data['folders']})
            new_data.update({'positions': self.json_data['positions']})
            print(split, len(new_data['files'][0]))
            self.new_json_data.update({split: new_data})

    def _create_splits_from_folder(self, train_filter, val_filter, test_filter, val_equal_test):
        """Splits the data from self.json_data into train, validation and test subsets based on the folder structure.

        :param train_filter: Unique folder name for all training files
        :param val_filter: Unique folder name for all validation files
        :param test_filter: Unique folder name for all test files
        :param val_equal_test: If True, the validation files will equal the test files.
        """
        folders = self.json_data['folders']
        files = self.json_data['files']
        positions = self.json_data['positions']
        numerics = self.json_data['numerical_values']

        splits = ['validation', 'test', 'train']
        for split in splits:
            # new data will contain the new json dict for the split
            new_data = {}
            new_data.update({'names': self.json_data['names']})
            new_data.update({'types': self.json_data['types']})
            new_folders = []
            new_items = []
            new_positions = []
            if split == 'train':
                filter_split = train_filter
            elif split == 'validation' and val_equal_test is True:
                filter_split = test_filter
            elif split == 'validation':
                filter_split = val_filter
            elif split == 'test':
                filter_split = test_filter
            else:
                filter_split = split
            for folder, file, position, numeric in zip(folders, files, positions, numerics):
                # Keep only the folders which have the split in the pathname
                new_folder = dl.DirLister.include_dirs_by_folder(folder, filter_split)
                # If numeric values are already added, save them, otherwise save the filename
                if numeric is not None:
                    filter_files, new_position = dl.DirLister.include_files_by_folder(file, filter_split, position)
                    new_item = [numeric[file.index(f)] for f in filter_files]
                else:
                    new_item, new_position = dl.DirLister.include_files_by_folder(file, filter_split, position)

                new_folders.append(new_folder)
                new_items.append(new_item)
                new_positions.append(new_position)
                print(split, len(new_item))

            new_data.update({'files': new_items})
            new_data.update({'folders': new_folders})
            new_data.update({'positions': new_positions})
            self.new_json_data.update({split: new_data})

    def _create_splits_from_kitti_list(self):
        """Splits the data from self.json_data into train, validation and test subsets as given in the split folder.

        The split folder contains a txt-file for the train, validation and subset, respectively. These files contain
        lists of files that will be  allocated to the subset. The KITTI-specific way of storing stereo images is
        being considered.
        """
        names = self.json_data['names']
        folders = self.json_data['folders']
        files = self.json_data['files']
        positions = self.json_data['positions']
        numerics = self.json_data['numerical_values']

        split_folder = os.path.join(self.output_path, 'splits')
        split_files = os.listdir(split_folder)
        splits = []
        # get the split files from the corresponding files
        for s in split_files:
            if 'train' in s:
                splits.append('train')
            elif 'test' in s:
                splits.append('test')
            elif 'val' in s:
                splits.append('validation')

        for split, split_file in zip(splits, split_files):
            split_file = os.path.join(split_folder, split_file)
            files_to_keep = pd.read_csv(split_file, header=None)[0].values
            for i in range(len(files_to_keep)):
                # compatibility for linux + windows
                files_to_keep[i] = files_to_keep[i].replace('\\', os.sep)
                files_to_keep[i] = files_to_keep[i].replace('/', os.sep)
            # new data contains the new data from the json dict
            new_data = {}
            new_data.update({'names': self.json_data['names']})
            new_data.update({'types': self.json_data['types']})
            new_folders = []
            new_items = []
            new_positions = []

            for file, folder, position, numeric in zip(files, folders, positions, numerics):
                bottom = file[0].split(os.path.sep)[0]
                file = [os.path.join(*(f.split(os.path.sep)[1:])) for f in file]
                new_item = []
                new_position = []
                # go through all the files which are supposed to be kept
                for f in files_to_keep:
                    if f in file:
                        # if numeric values are present save them, otherwise save the filename
                        if numeric is not None:
                            new_item.append(numeric[file.index(f)])
                        else:
                            new_item.append(os.path.join(bottom, f))
                        new_position.append(position[file.index(f)])
                    elif f.replace('image_02', 'image_03') in file:
                        # sometimes the stereo images are saved at a slightly different path not
                        # indicated by the folder
                        if numeric is not None:
                            new_item.append(numeric[file.index(f.replace('image_02', 'image_03'))])
                        else:
                            new_item.append(os.path.join(bottom, f.replace('image_02', 'image_03')))
                        new_position.append(position[file.index(f.replace('image_02', 'image_03'))])
                # get the new folder names for the split
                if numeric is not None:
                    new_folder = [os.path.split(f)[0] for f in file]
                else:
                    new_folder = [os.path.split(f)[0] for f in new_item]
                new_folder = sorted(list(set(new_folder)))

                # save the data
                new_items.append(new_item)
                new_positions.append(new_position)
                new_folders.append(new_folder)
            new_data.update({'files': new_items})
            new_data.update({'folders': new_folders})
            new_data.update({'positions': new_positions})
            self.new_json_data.update({split: new_data})

    def _create_splits_from_mat_file(self):
        """Splits the data from self.json_data into train, validation and test subsets as given by a .mat file.

        The mat file is assumed to have a format like in the GTA5 dataset. All image files have to be in one folder,
        the .mat file contains up to three entries with arrays containing the file numbers, each entry corresponding
        to one split. The file numbers start at 1.
        """
        names = self.json_data['names']
        folders = self.json_data['folders']
        files = self.json_data['files']
        positions = self.json_data['positions']
        numerics = self.json_data['numerical_values']

        matfile = dl.DirLister.get_files_by_ending(self.dataset_folder_path, '.mat')
        assert len(matfile) == 1, 'There must only be one .mat file in the folder'
        split_data = sio.loadmat(matfile[0])
        splits = []
        split_keys = []
        for key in list(split_data.keys()):
            if 'train' in key:
                splits.append('train')
                split_keys.append(key)
            elif 'test' in key:
                splits.append('test')
                split_keys.append(key)
            elif 'val' in key:
                splits.append('validation')
                split_keys.append(key)
        assert splits != [], 'No split data found in .mat file'

        for split, key in zip(splits, split_keys):
            new_data = {}
            new_data.update({'names': self.json_data['names']})
            new_data.update({'types': self.json_data['types']})
            new_folders = folders
            new_items = []
            new_positions = []

            for i in range(len(files)):
                new_item = []
                new_position = []
                for file_number in split_data[key]:
                    index = int(file_number) - 1
                    new_item.append(files[i][index])
                    new_position.append(positions[i][index])
                new_items.append(new_item)
                new_positions.append(new_position)

            new_data.update({'files': new_items})
            new_data.update({'folders': new_folders})
            new_data.update({'positions': new_positions})
            self.new_json_data.update({split: new_data})

    def _create_splits_from_text_file(self, train_filter, val_filter, test_filter, path_to_split_info):
        """Splits the data from self.json_data into train, validation and test subsets as given by txt-files.

        Reading mode for the case that the split information is stored in text files inside the dataset.
        It is assumed that each line of the txt file contains a string that is contained in the filename
        (Optimized for Pascal Voc 2012)
        Warning: At the moment, the folders are just copied from the basic_files!

        :param train_filter: String that is used to recognize the train files.
        :param val_filter: String that is used to recognize the validation files.
        :param test_filter: String that is used to recognize the test files.
        """
        names = self.json_data['names']
        folders = self.json_data['folders']
        files = self.json_data['files']
        positions = self.json_data['positions']
        numerics = self.json_data['numerical_values']

        splits = ('train', 'validation', 'test')
        split_filters = (train_filter, val_filter, test_filter)
        for split, filter in zip(splits, split_filters):
            new_data = {}
            new_data.update({'names': self.json_data['names']})
            new_data.update({'types': self.json_data['types']})
            new_folders = folders
            new_items = []
            new_positions = []
            if filter == None:
                continue
            path = os.path.join(self.dataset_folder_path, path_to_split_info, filter + '.txt')
            with open(path) as textfile:
                split_files = textfile.readlines()
            for folder, file, position, numeric in zip(folders, files, positions, numerics):
                new_item = []
                new_position = []
                for split_file in split_files:
                    split_file = split_file[:-1]
                    matching_files = [f for f in file if split_file in f]
                    if len(matching_files) == 0:
                        continue
                    assert len(matching_files) == 1, 'The file name {0} is not unique'.format(split_file)
                    matching_file = matching_files[0]
                    index = file.index(matching_file)
                    new_item.append(matching_file)
                    new_position.append(position[index])
                new_items.append(new_item)
                new_positions.append(new_position)
            new_data.update({'files': new_items})
            new_data.update({'folders': new_folders})
            new_data.update({'positions': new_positions})
            self.new_json_data.update({split: new_data})

    def _create_random_split(self, split_sizes, seed):
        """Randomly splits the data from self.json_data into train, validation and test subsets with a given size.

        Each folder is randomly mapped to the subsets "train", "validation" and "test". The sizes of there three
        subsets are specified by the parameter split_sizes. A seed has to be specified in order to prevent an existing
        split from being overwritten with a new random mapping of files to subsets.

        :param split_sizes: 3-tuple of the format (training_set_size, validation_set_size, test_set_size), the sizes
            being given as fractions of the whole set.
        :param seed: Seed to initialize the random number generator
        """
        names = self.json_data['names']
        folders = self.json_data['folders']
        files = self.json_data['files']
        positions = self.json_data['positions']
        numerics = self.json_data['numerical_values']

        random.seed(seed)
        pop_folders = folders[0].copy()
        splits = ('train', 'test', 'validation')
        split_sizes = {'test': int(len(pop_folders) * split_sizes[2]),
                       'validation': int(len(pop_folders) * split_sizes[1]),
                       'train': len(pop_folders) - int(len(pop_folders) * split_sizes[1]) -
                                int(len(pop_folders) * split_sizes[2])
                       }
        split_folders = {'train': {}, 'validation': {}, 'test': {}}
        split_index = 0
        split = splits[split_index]
        num_folders = 0

        # Map folders randomly to the splits
        while pop_folders != []:
            if num_folders >= split_sizes[split]:
                split_index += 1
                split = splits[split_index]
                num_folders = 0
            pop_index = random.randint(0, len(pop_folders) - 1)
            pop_folder = pop_folders.pop(pop_index)
            global_index = folders[0].index(pop_folder)
            num_folders += 1
            for i, name in zip(range(len(names)), names):
                if name not in split_folders[split]:
                    split_folders[split][name] = []
                split_folders[split][name].append(folders[i][global_index])
        self.split_folders = split_folders

        # Check that no folder has been assigned twice (Just a safety measure)
        for name in names:
            for split_index in range(3):
                split = splits[split_index]
                for idx in range(2):
                    other_split = splits[(split_index + idx) % 3]
                    assert split_folders[split][name] not in split_folders[other_split][name], \
                        "Program error: {} has been assigned twice".format(split_folders[split][name])

        # Based on the random mapping, the three split subsets are added to the new_json_data.
        for split in splits:
            # new data will contain the new json dict for the split
            new_data = {}
            new_data.update({'names': self.json_data['names']})
            new_data.update({'types': self.json_data['types']})
            new_folders = []
            new_items = []
            new_positions = []
            filter_split = split
            for name, folder, file, position, numeric in zip(names, folders, files, positions, numerics):
                # Take only the folders and files which belong to the current split
                new_folder = sorted(split_folders[split][name])
                filter_files = []
                new_position = []
                for f in new_folder:
                    ff, np = dl.DirLister.include_files_by_folder(file, f, position)
                    filter_files.extend(ff)
                    new_position.extend(np)
                # If numerical values are already added, save them, otherwise save the filename
                if numeric is not None:
                    new_item = [numeric[file.index(f)] for f in filter_files]
                else:
                    new_item = filter_files

                new_folders.append(new_folder)
                new_items.append(new_item)
                new_positions.append(new_position)
                print(split, len(new_item))

            new_data.update({'files': new_items})
            new_data.update({'folders': new_folders})
            new_data.update({'positions': new_positions})
            self.new_json_data.update({split: new_data})

    def dump_to_json(self):
        """Dumps each split to a json file.

         The name of the split file is set automatically as the name of the split, e.g. train.json
         """
        if not os.path.exists(self.output_path):
            print('Output path does not exits!')
            print('Creating output path {}'.format(self.output_path))
            os.makedirs(self.output_path)

        for split in self.new_json_data.keys():
            with open(os.path.join(self.output_path, split + '.json'), 'w') as fp:
                json.dump(self.new_json_data[split], fp)

    def filter_by_resolution(self, resolution):
        """Filter files by minimal resolutions that the image files have to have

        :param resolution: minimal resolution of type tuple(height, width). If one dimension shall not be included
            in the filter, set the corresponding dimension to 0.
        """
        splits = ['validation', 'test', 'train']
        for split in splits:
            print('Filtering split: {}'.format(split))
            folders = self.new_json_data[split]['folders']
            files = self.new_json_data[split]['files']
            positions = self.new_json_data[split]['positions']

            # Find the color folder in the list of folders
            color_folder = None
            for folder in folders:
                if 'color' in folder[0].split('/')[-1].lower():
                    color_folder = folders.index(folder)
                    break
            assert color_folder is not None, "Could not find color folder"

            # Load color images one-by-one and check resolution. If resolution is sufficient, add its index to
            # indices_to_keep
            indices_to_keep = []
            for file, index in zip(files[color_folder], positions[color_folder]):
                img = Image.open(os.path.join(self.dataset_folder_path, file))
                if resolution[0] > 0:
                    if resolution[0] > img.height:
                        continue
                if resolution[1] > 0:
                    if resolution[1] > img.width:
                        continue
                indices_to_keep.append(index)
            indices = indices_to_keep

            n_files_orig = len(files[color_folder])
            n_files_filt = len(indices_to_keep)

            # Apply indices_to_keep to all file_lists and position_lists
            for i in range(len(folders)):
                self.new_json_data[split]['files'][i] = [file for file, pos in zip(self.new_json_data[split]['files'][i], self.new_json_data[split]['positions'][i]) if pos in indices]
                self.new_json_data[split]['positions'][i] = indices_to_keep

            print(' -> Originally: {} images\n -> After resolution filter: {} images\n -> Removed: {} images'.format(
                n_files_orig, n_files_filt, n_files_orig - n_files_filt))

    def save_split_folders(self, mode):
        """Saves three text files containing the folders of each split into the split folder

        This method can be used to obtain the split mapping created by a random split.

        :param mode: 'same_root' means that each split folder contains all image categories, e.g.
            'image_folder_01/color', 'image_folder_01/depth' etc.
        :param split_path: Path where the files will be saved
        """
        assert mode in ('same_root',)
        if mode == 'same_root':
            splits = ('train', 'test', 'validation')
            for split in splits:
                split_list = []
                splitlength = len(self.split_folders[split]['color'])
                for i in range(splitlength):
                    sample_folders = [sorted(self.split_folders[split][key])[i]
                                      for key in self.split_folders[split].keys()]
                    split_list.append(os.path.commonpath(sample_folders))
                with open(os.path.join(self.output_path, split+'_folders.txt'), 'w') as file:
                    for folder in split_list:
                        file.write(folder+'\n')


class DatasetSplitter:
    """Class to create the train, validation and test split filelists for different datasets"""

    def __init__(self, dataset, splits=None):
        """Initializes the dataset name and split

        :param dataset: Name of the dataset (without any suffixes for the split name)
        :param splits: List of split names as suffixes for the dataset name. Only for dataset with different split
            options.
        """
        assert dataset in SUPPORTED_DATASETS, 'Dataset not supported'
        # This dictionary contains a mapping from the dataset name to the corresponding function that has to be
        # called in order to create the splits. Its entries have the format
        #     'dataset_name': (self._create_datasetname_splits, (arguments list))
        # The arguments list contains the arguments for the split function. It can be replaced by None if there are no
        # arguments to be passed.
        self.data_dict = {'cityscapes': (self._create_cityscapes_splits, (dataset,)),
                          'cityscapes_video': (self._create_cityscapes_splits, (dataset,)),
                          'cityscapes_sequence': (self._create_cityscapes_splits, (dataset,)),
                          'cityscapes_extra': (self._create_cityscapes_splits, (dataset,)),
                          'cityscapes_part': (self._create_cityscapes_splits, (dataset,)),
                          'kitti': (self._create_kitti_splits, (splits,)),
                          'kitti_2012': (self._create_kitti2012_splits, None),
                          'kitti_2015': (self._create_kitti2015_splits, None),
                          'virtual_kitti': (self._create_virtual_kitti_splits, None),
                          'mapillary': (self._create_mapillary_splits, None),
                          'mapillary_by_ID': (self._create_mapillary_splits, None),
                          'gta5': (self._create_gta5_splits, None),
                          'synthia': (self._create_synthia_splits, None),
                          'bdd100k': (self._create_bdd100k_splits, None),
                          'voc2012': (self._create_voc2012_splits, None),
                          'a2d2': (self._create_a2d2_splits, (splits,)),
                          'lostandfound': (self._create_lostandfound_splits, None),
                          'camvid': (self._create_camvid_splits, None),
                          'make3d': (self._create_make3d_splits, None),
                          }
        self.dataset = dataset

    def create_splits(self):
        """Creates the train, validation and test json-files for the dataset in all given splits"""
        split_function = self.data_dict[self.dataset][0]
        split_args = self.data_dict[self.dataset][1]
        if split_args is None:
            split_function()
        else:
            split_function(*split_args)

    def _create_kitti_splits(self, splits=None):
        """This method creates all kitti split files.

        :param splits: List of names of splits for which the json files will be generated.
        """
        SPLITS = ('eigen_split', 'zhou_split', 'zhou_split_left', 'zhou_split_right', 'benchmark_split', 'kitti_split',
                  'odom10_split', 'odom09_split', 'video_prediction_split', 'optical_flow_split')
        if splits is None:
            splits = SPLITS
        else:
            for split in splits:
                assert split in SPLITS

        for split in splits:
            print(split)
            split_creator = SplitCreator(self.dataset)
            split_creator.get_all_items()
            split_creator.set_split_path(split)
            split_creator.get_split_data(read_mode='kittifiles')
            split_creator.dump_to_json()

    def _create_kitti2012_splits(self):
        """This method creates the split files for the KITTI 2012 dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='folders', val_equal_test=True, train_filter='training',
                                     test_filter='testing')
        split_creator.dump_to_json()

    def _create_kitti2015_splits(self):
        """This method creates the split files for the KITTI 2015 dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='folders', val_equal_test=True, train_filter='training',
                                     test_filter='testing')
        split_creator.dump_to_json()

    def _create_cityscapes_splits(self, dataset):
        """This method creates the splits for the Cityscapes dataset"""
        split_creator = SplitCreator(dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        if dataset == 'cityscapes_extra':
            split_creator.get_split_data(read_mode='folders', train_filter='train_extra', val_filter='val')
        else:
            split_creator.get_split_data(read_mode='folders', val_filter='val')
        split_creator.dump_to_json()

    def _create_virtual_kitti_splits(self):
        """This method creates all kitti split files"""
        split = 'full_split'
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path(split)
        split_creator.get_split_data(read_mode='alltrain')
        split_creator.dump_to_json()

        splits = ['clone_split']
        for split in splits:
            split_creator = SplitCreator(self.dataset)
            split_creator.get_all_items()
            split_creator.set_split_path(split)
            split_creator.get_split_data(read_mode='folders', train_filter='clone')
            split_creator.dump_to_json()

    def _create_mapillary_splits(self):
        """This method creates the splits for the Mapillary dataset"""
        splits = [None, 'by_ID_res_288x960', 'by_ID_res_512x1024']
        resolutions = [None, (288, 960), (512, 1024)]
        for split, res in zip(splits, resolutions):
            split_creator = SplitCreator(self.dataset)
            split_creator.get_all_items()
            if split is not None:
                split_creator.set_split_path(split)
            else:
                split_creator.set_split_path()
            split_creator.get_split_data(read_mode='folders')
            if res is not None:
                split_creator.filter_by_resolution(res)
            split_creator.dump_to_json()

    def _create_gta5_splits(self):
        """This method creates the splits for the GTA5 dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='matfile')
        split_creator.dump_to_json()

        # Also create a full split with all files being categorized as train
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path('full_split')
        split_creator.get_split_data(read_mode='alltrain')
        split_creator.dump_to_json()

    def _create_synthia_splits(self):
        """This method creates the splits for the Snythia dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='alltrain')
        split_creator.dump_to_json()

    def _create_bdd100k_splits(self):
        """This method creates the splits for the BDD100K dataset"""
        split_creator = SplitCreator(self.dataset)
        print(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='folders', val_filter='val')
        split_creator.dump_to_json()

    def _create_voc2012_splits(self):
        """This method creates the splits for the voc2012 dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='textfile', path_to_split_info=os.path.join('ImageSets', 'Segmentation'),
                                     train_filter='trainaug', val_filter='val', test_filter=None)
        split_creator.dump_to_json()

    def _create_a2d2_splits(self, split_name):
        """This method creates the splits for the a2d2 dataset"""
        split_info = {'andreas_split': {'seed': 53,
                                        'sizes': (0.8, 0.1, 0.1)}
                      }
        assert split_name in split_info, 'No seed is known for this split.'
        seed = split_info[split_name]['seed']
        sizes = split_info[split_name]['sizes']
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path(split_name)
        split_creator.get_split_data(read_mode='random_folders', rand_split=sizes, rand_seed=seed)
        split_creator.dump_to_json()
        split_creator.save_split_folders(mode='same_root')

    def _create_lostandfound_splits(self):
        """This method creates the splits for the Lostandfound dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='folders')
        split_creator.dump_to_json()

    def _create_camvid_splits(self):
        """This method creates the splits for the BDD100K dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='folders', val_filter='val')
        split_creator.dump_to_json()

    def _create_make3d_splits(self):
        """This method creates the splits for the make3d dataset"""
        split_creator = SplitCreator(self.dataset)
        split_creator.get_all_items()
        split_creator.set_split_path()
        split_creator.get_split_data(read_mode='folders')
        split_creator.dump_to_json()


if __name__ == '__main__':
    """ Supported datasets are:
        - a2d2
        - bdd100k
        - camvid
        - cityscapes
        - cityscapes_extra
        - cityscapes_sequence
        - cityscapes_standard
        - cityscapes_video
        - gta5
        - kitti
        - kitti_2012
        - kitti_2015
        - lostandfound
        - mapillary
        - mapillary_by_ID
        - synthia
        - virtual_kitti
        - voc2012
    """

    dataset = 'cityscapes'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    splits = None   # Possibility to pass a list of split names

    data_splitter = DatasetSplitter(dataset, splits)
    data_splitter.create_splits()
