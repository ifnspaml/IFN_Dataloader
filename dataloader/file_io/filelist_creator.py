import os
import sys
import json
import numpy as np
import pandas as pd

import dataloader.file_io.dir_lister as dl
import dataloader.file_io.get_path as gp

SUPPORTED_DATASETS = ('cityscapes', 'cityscapes_video', 'cityscapes_sequence', 'cityscapes_extra', 'cityscapes_part',
                      'kitti', 'kitti_2012', 'kitti_2015', 'virtual_kitti', 'mapillary', 'mapillary_by_ID', 'gta5',
                      'synthia', 'bdd100k', 'voc2012', 'a2d2', 'lostandfound', 'camvid', 'make3d')
# If a directory path contains one one the following strings, it will be ignored
FOLDERS_TO_IGNORE = ('segmentation_trainid')

class FilelistCreator:
    """This class provides helper functions to create a list of all files inside a dataset."""

    def __init__(self, path):
        """Initializes the dataset path and gets a list of all folders in the dataset

        :param path: absolute path to the dataset
        """
        if os.path.isdir(os.path.abspath(path)):
            self.dataset_path = os.path.abspath(path)
        else:
            sys.exit("Der angegebene Dateipfad existiert nicht")
        self.folders = dl.DirLister.get_directories(self.dataset_path)
        self.json_dict = {}

    def preprocess_directories_list(self, filter_names):
        """Removes all directories from the list that contain at least one of the given strings

        :param filter_names: list of names by which the folders should be filtered
        """
        self.folders = dl.DirLister.remove_dirs_by_name(self.folders, filter_names)

    def preprocess_file_list(self, filter_dict):
        """Removes all entries from the file lists that do not contain all of the specified filters.

        Sometimes (e.g. in segmentation masks) there are multiple representations of the data and one only wants to
        keep one. This function will go through the self.json_dict name by name and for each name, it will remove all
        entries from the 'files' entry that to not contain all of the strings in filter_dict[name].

        :param filter_dict: dictionary where the key corresponds to one of the names and the values correspond to the
            names which have to appear in the files
        """
        for key in filter_dict.keys():
            if key in self.json_dict['names']:
                index = self.json_dict['names'].index(key)
                self.json_dict['files'][index] = dl.DirLister.include_dirs_by_name(
                    self.json_dict['files'][index], filter_dict[key])

    def create_filelist(self, filters, ending, ignore=(), ambiguous_names_to_ignore=()):
        """Creates a filtered list of files inside self.folders

        :param filters: names which have to appear in the directory path
        :param ending: file ending of valid files
        :param ignore: list of strings. All files/folders containing these strings will be ignored
        :param ambiguous_names_to_ignore: Sometimes it is inevitable to have a filter name that also appears in every
            path name. In this case, these longer strings can be specified in this parameter and the folders and files
            will only be included if the filters appear outside of any of the ambiguous_strings_to_ignore.
        :return: list of all filtered folders and a list of all filtered files inside these folders.
        """
        self.preprocess_directories_list(FOLDERS_TO_IGNORE)
        folders = dl.DirLister.include_dirs_by_name(self.folders, filters, ignore, ambiguous_names_to_ignore)
        folders = sorted(folders, key=str.lower)
        filelist = []
        for fold in folders:
            files = dl.DirLister.get_files_by_ending(fold, ending, ignore)
            filelist.extend(files)
        filelist = sorted(filelist, key=str.lower)
        return folders, filelist

    def dump_to_json(self, filename, remove_root=True):
        """Dumps the json list to a json file

        :param filename: name of the json file
        :param remove_root: if True, the root directory up to the dataset path is removed to store the images
            relative to the dataset path
        """
        dump_location = os.path.join(self.dataset_path, filename)
        if remove_root:
            root_stringlength = len(self.dataset_path) + 1
            for i in range(len(self.json_dict['folders'])):
                for j in range(len(self.json_dict['folders'][i])):
                    folder_file = self.json_dict['folders'][i][j]
                    self.json_dict['folders'][i][j] = folder_file[root_stringlength:]
            for i in range(len(self.json_dict['files'])):
                for j in range(len(self.json_dict['files'][i])):
                    image_file = self.json_dict['files'][i][j]
                    self.json_dict['files'][i][j] = image_file[root_stringlength:]
        with open(dump_location, 'w') as fp:
            json.dump(self.json_dict, fp)

    def create_json_from_list(self, json_list, stereo_replace):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.
        This method has to be implemented for each dataset-specific child class.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...]
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...]
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: dicionary that defines the strings that have to be interchanged in order to get the
            right stereo image from the left stereo image: {left_image_string: right_image_string}
        """
        raise NotImplementedError


class KITTIFilelistCreator(FilelistCreator):
    """Class to create the KITTI file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: dicionary that defines the strings that have to be interchanged in order to get the
            right stereo image from the left stereo image: {left_image_string: right_image_string}
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            folders_list.append(folders)
            file_list.append(files)
            # positions contains 4-tuples, where the single entries have the following meaning:
            # 1. global position inside the dataset (sorted by frame number and sequence
            # 2. number of preceding frames in the sequence
            # 3. number of frames in the sequence after the current frame
            # 4. local position inside the list of the elements (e.g. depth has 20000 elements but color has 40000
            # then the first entry will contain the mapping from depth to color and the fourth entry will contain
            # numbers from 0 to 20000
            positions = []
            lower_limit = [0]
            upper_limit = []
            old_frame_number = None
            new_frame_number = None
            # get the sequence limits (upper and lower)
            for file in files:
                old_frame_number = new_frame_number
                new_frame_number = int(os.path.splitext(os.path.split(file)[1])[0])
                if old_frame_number != new_frame_number - 1 and old_frame_number is not None:
                    upper_limit.append(files.index(file) - 1)
                    lower_limit.append(files.index(file))
            upper_limit.append(len(files) - 1)
            index = 0
            # get the position entries and the file names of all image files, numerical values are handled
            # differently
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if index < len(lower_limit) - 1 and j == lower_limit[index + 1]:
                    index += 1
                if i == 0:
                    positions.append((len(positions), j - lower_limit[index], upper_limit[index] - j, j))
                    main_files.append(file)
                else:
                    if 'right' in name:
                        for key in stereo_replace.keys():
                            file = file.replace(stereo_replace[key], key)
                    positions.append((main_files.index(file), j - lower_limit[index], upper_limit[index] - j, j))
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))

        # camera intrinsic parameters
        camera_intrinsics = []
        camera_intrinsics_right = []
        json_list['names'].extend(['camera_intrinsics', 'camera_intrinsics_right'])
        json_list['types'].extend(['.txt', '.txt'])
        json_list['filters'].extend([['Raw_data'], ['Raw_data']])
        for file in main_files:
            base = file.split(os.sep)[0]
            # get the corresponding calibration files, every color frame has a calibration file
            if 'test' in file:
                param_file_name = os.path.split(file)[1].replace('png', 'txt')
                calib_file = os.path.join(self.dataset_path, 'Raw_data', base, 'intrinsics', param_file_name)
                left = open(calib_file).readlines()[0][6:].split()
                right = open(calib_file).readlines()[0][6:].split()
            else:
                calib_file = os.path.join(self.dataset_path, 'Raw_data', base, 'calib_cam_to_cam.txt')
                left = open(calib_file).readlines()[:20][-1][6:].split()
                right = open(calib_file).readlines()[:28][-1][6:].split()
            left_matrix = np.eye(4)
            right_matrix = np.eye(4)
            left_matrix[:3, :3] = np.array([float(l) for l in left]).reshape((3, 3))
            right_matrix[:3, :3] = np.array([float(r) for r in right]).reshape((3, 3))
            left_matrix = list(left_matrix)
            right_matrix = list(right_matrix)
            left_matrix = [list(l) for l in left_matrix]
            right_matrix = [list(r) for r in right_matrix]
            camera_intrinsics.append(left_matrix)
            camera_intrinsics_right.append(right_matrix)
        print('camera_intrinsics:', len(camera_intrinsics))
        print('camera_intrinsics_right:', len(camera_intrinsics_right))
        folders_list.extend([folders_list[0], folders_list[1]])
        position_list.extend([position_list[0], position_list[0]])
        file_list.extend([file_list[0].copy(), file_list[1].copy()])
        numerical_list.extend([camera_intrinsics, camera_intrinsics_right])

        # velocity and timestamps
        json_list['names'].extend(['timestamp', 'velocity'])
        json_list['types'].extend(['.txt', '.txt'])
        json_list['filters'].extend([['Raw_data', 'oxts'], ['Raw_data', 'oxts']])
        folders, files = self.create_filelist(['Raw_data', 'oxts'], '.txt')
        folders_vel = dl.DirLister.include_dirs_by_name(folders, 'data')
        folders_time = [os.path.split(f)[0] for f in folders_vel]
        files.extend([os.path.join(f, 'timestamps.txt') for f in folders_time])
        folders_list.extend([folders_time, folders_vel])
        times = []
        velocities = []
        for file in files:
            if 'timestamps' in file:
                temp_time = np.array(pd.read_csv(file, header=None, delimiter=' ')[1].values)
                time = [float(t.split(':')[0])*3600 + float(t.split(':')[1])*60 + float(t.split(':')[2])
                        for t in temp_time]
                times.extend(time)
            if '00000' in file:
                temp_data = np.array(pd.read_csv(file, header=None, delimiter=' ').values)[0]
                velocity = np.sqrt(temp_data[8]**2 + temp_data[9]**2 + temp_data[10]**2)
                velocities.append(velocity)
        file_list.extend([file_list[0][:len(times)], file_list[0][:len(velocities)]])
        position_list.extend([position_list[0][:len(times)], position_list[0][:len(velocities)]])
        numerical_list.extend([times, velocities])
        print('timestamps:', len(times))
        print('velocities', len(velocities))

        # poses for odometry evaluation (odometry dataset needed if desired!)
        json_list['names'].extend(['poses'])
        json_list['types'].extend(['.txt'])
        json_list['filters'].extend([['Raw_data']])

        folders, files = self.create_filelist('poses', '.txt')

        if files:
            poses = []
            drives = []
            frame_numbers = []
            # defined officially in the odometry set, seq. 8 is not available in the raw data
            frame_min_max_sequence = [[0, 270],
                                      [0, 2760],
                                      [0, 1100],
                                      [0, 1100],
                                      [1100, 5170],
                                      [0, 1590],
                                      [0, 1200],
                                      [0, 4540],
                                      [0, 4660],
                                      [0, 1100]]
            # get the poses from file
            for min_max, file in zip(frame_min_max_sequence, files):
                for i in range(min_max[0], min_max[1] + 1, 1):
                    frame_numbers.append(i)
                drive = file.split(os.sep)[-3]
                for i in range(min_max[0], min_max[1] + 1, 1):
                    drives.append(drive)
                pose = np.loadtxt(file).reshape((-1, 3, 4))
                for p in pose:
                    p = list(p)
                    p = [list(i) for i in p]
                    poses.append(p)
            counter = 0
            positions = []
            files = []
            # get the corresponding filenames and positions from the poses
            lower_limit = [0]
            upper_limit = []
            old_frame_number = None
            new_frame_number = None
            for i, number in zip(range(len(frame_numbers)), frame_numbers):
                old_frame_number = new_frame_number
                new_frame_number = number
                if old_frame_number != new_frame_number - 1 and old_frame_number is not None:
                    upper_limit.append(i - 1)
                    lower_limit.append(i)
            upper_limit.append(len(frame_numbers) - 1)
            index = 0
            # append the values of positions and files to the data dict
            for i, file in zip(range(len(main_files)), main_files):
                main_drive = file.split(os.sep)[-4]
                main_frame_number = int(os.path.splitext(os.path.split(file)[1])[0])
                if main_drive == drives[counter] and main_frame_number == frame_numbers[counter]:
                    temp_position = (main_files.index(file), counter - lower_limit[index],
                                     upper_limit[index] - counter, counter)
                    positions.append(temp_position)
                    files.append(file_list[0][main_files.index(file)])
                    counter += 1
                    if index < len(lower_limit) - 1 and counter == lower_limit[index + 1]:
                        index += 1
                    if counter == len(frame_numbers):
                        break
            print('poses:', len(poses))
            position_list.append(positions)
            file_list.append(files)
            numerical_list.append(poses)
            folders_list.append(folders)

        # save the json dict
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class KITTI2015FilelistCreator(FilelistCreator):
    """Class to create the KITTI file list"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: dicionary that defines the strings that have to be interchanged in order to get the
            right stereo image from the left stereo image: {left_image_string: right_image_string}
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            folders_list.append(folders)
            file_list.append(files)
            positions = []
            lower_limit = [0]
            upper_limit = []
            old_frame_number = None
            new_frame_number = None
            for file in files:
                old_frame_number = new_frame_number
                new_frame_number = int(os.path.splitext(os.path.split(file)[1])[0].split('_')[1])
                if old_frame_number != new_frame_number - 1 and old_frame_number is not None:
                    upper_limit.append(files.index(file) - 1)
                    lower_limit.append(files.index(file))
            upper_limit.append(len(files) - 1)
            index = 0
            for j, file in zip(range(len(files)), files):
                folder = os.path.split(os.path.split(os.path.split(file)[0])[0])[1]
                file = os.path.split(file)[1]
                file = os.path.join(folder, file)
                if index < len(lower_limit) - 1 and j == lower_limit[index + 1]:
                    index += 1
                if i == 0:
                    positions.append((len(positions), j - lower_limit[index], upper_limit[index] - j, j))
                    main_files.append(file)
                else:
                    if 'right' in name:
                        for key in stereo_replace.keys():
                            file = file.replace(stereo_replace[key], key)
                    positions.append((main_files.index(file), j - lower_limit[index], upper_limit[index] - j, j))
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))

        # camera intrinsic parameters
        camera_intrinsics = []
        camera_intrinsics_right = []
        json_list['names'].extend(['camera_intrinsics', 'camera_intrinsics_right'])
        json_list['types'].extend(['.txt', '.txt'])
        json_list['filters'].extend(['calib_cam_to_cam', 'calib_cam_to_cam'])
        for file in main_files:
            base = file.split(os.sep)[1].split('_')[0]
            folder = file.split(os.sep)[0]
            param_file_name = base + '.txt'
            calib_file = os.path.join(self.dataset_path, folder, 'calib_cam_to_cam', param_file_name)
            left = open(calib_file).readlines()[:20][-1][6:].split()
            right = open(calib_file).readlines()[:28][-1][6:].split()
            left_matrix = np.eye(4)
            right_matrix = np.eye(4)
            left_matrix[:3, :3] = np.array([float(l) for l in left]).reshape((3, 3))
            right_matrix[:3, :3] = np.array([float(r) for r in right]).reshape((3, 3))
            left_matrix = list(left_matrix)
            right_matrix = list(right_matrix)
            left_matrix = [list(l) for l in left_matrix]
            right_matrix = [list(r) for r in right_matrix]
            camera_intrinsics.append(left_matrix)
            camera_intrinsics_right.append(right_matrix)
        print('camera_intrinsics:', len(camera_intrinsics))
        print('camera_intrinsics_right:', len(camera_intrinsics_right))
        folders_list.extend([folders_list[0], folders_list[1]])
        position_list.extend([position_list[0], position_list[0]])
        file_list.extend([file_list[0].copy(), file_list[1].copy()])
        numerical_list.extend([camera_intrinsics, camera_intrinsics_right])

        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class VirtualKITTIFilelistCreator(FilelistCreator):
    """Class to create the Virtual KITTI file list"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
            'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
            'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: Not used for this Dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            folders_list.append(folders)
            file_list.append(files)
            # positions contains 4-tuples, where the single entries have the following meaning:
            # 1. global position inside the dataset (sorted by frame number and sequence
            # 2. number of preceding frames in the sequence
            # 3. number of frames in the sequence after the current frame
            # 4. local position inside the list of the elements (e.g. depth has 20000 elements but color has 40000
            # then the first entry will contain the mapping from depth to color and the fourth entry will contain
            # numbers from 0 to 20000
            positions = []
            lower_limit = [0] # Array containing the index of the starting frames for each video
            upper_limit = []  # Array containing the index of the end frame for each video
            old_frame_number = None
            new_frame_number = None
            # get the sequence limits (upper and lower)
            for file in files:
                old_frame_number = new_frame_number
                new_frame_number = int(os.path.splitext(os.path.split(file)[1])[0])
                # detect start of a new video
                if old_frame_number != new_frame_number - 1 and old_frame_number is not None:
                    upper_limit.append(files.index(file) - 1)
                    lower_limit.append(files.index(file))
            upper_limit.append(len(files) - 1)
            index = 0
            # get the position entries and the file names of all image files, numerical values are handled
            # differently
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                # detect start of a new video
                if index < len(lower_limit) - 1 and j == lower_limit[index + 1]:
                    index += 1
                if i == 0:
                    positions.append((len(positions), j - lower_limit[index], upper_limit[index] - j, j))
                    main_files.append(file)
                else:
                    positions.append((main_files.index(file), j - lower_limit[index], upper_limit[index] - j, j))
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))

            json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                              'numerical_values': numerical_list})
            self.json_dict = json_list


class CityscapesFilelistCreator(FilelistCreator):
    """Class to create the Cityscapes file list"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: dicionary that defines the strings that have to be interchanged in order to get the
            right stereo image from the left stereo image: {left_image_string: right_image_string}
        """

        # string sequences from corrupted files that should be ignored
        ignore_extra = ['troisdorf_000000_000073']
        ignore_video_left = ['frankfurt_000000_006434', 'frankfurt_000001_023592', 'frankfurt_000001_038767']
        ignore_video_right = ['frankfurt_000000_022587', 'frankfurt_000001_026781', 'frankfurt_000001_059933',
                              'frankfurt_000001_059934', 'frankfurt_000001_060157', 'frankfurt_000001_070159',
                              'frankfurt_000001_083533']
        ignore_list = ignore_extra + ignore_video_left + ignore_video_right

        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            # cityscapes_extra dataset
            if 'gtCoarse' in filter:
                filter.append('train_extra')

            folders, files = self.create_filelist(filter, type, ignore=ignore_list)
            if 'segmentation' in name:
                files = [f for f in files if 'color' not in f and 'instance' not in f]
            folders_list.append(folders)
            file_list.append(files)

            # Find the splits between the video sequences
            positions = []
            lower_limit = [0]
            upper_limit = []
            old_frame_number = None
            new_frame_number = None
            for file in files:
                old_frame_number = new_frame_number
                img_filename = os.path.splitext(os.path.split(file)[1])[0]
                new_frame_number = int(img_filename.split('_')[2])
                previous_frame_name = '_'.join(img_filename.split('_')[:2]) + '_' + str(new_frame_number-1).zfill(6)
                if previous_frame_name in ignore_list:
                    # Skip the ignored file while checking whether the images belong to the same sequence
                    if old_frame_number not in [new_frame_number - 2, new_frame_number - 3] \
                            and old_frame_number is not None:
                        upper_limit.append(files.index(file) - 1)
                        lower_limit.append(files.index(file))
                elif old_frame_number != new_frame_number - 1 and old_frame_number is not None:
                    upper_limit.append(files.index(file) - 1)
                    lower_limit.append(files.index(file))
            upper_limit.append(len(files) - 1)

            # Create the positions entries
            index = 0
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if index < len(lower_limit) - 1 and j == lower_limit[index + 1]:
                    index += 1
                if i == 0:
                    positions.append((len(positions), j - lower_limit[index], upper_limit[index] - j, j))
                    main_files.append('_'.join(file.split('_')[:-1]))
                else:
                    if 'right' in name:
                        for key in stereo_replace.keys():
                            file = file.replace(stereo_replace[key], key)
                    if 'segmentation' in name:
                        positions.append((main_files.index('_'.join(file.split('_')[:-2])),
                                          j - lower_limit[index], upper_limit[index] - j, j))
                    else:
                        positions.append((main_files.index('_'.join(file.split('_')[:-1])),
                                          j - lower_limit[index], upper_limit[index] - j, j))
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))

        # camera intrinsics
        main_folder = file_list[0][0].split(self.dataset_path + os.sep)[1]
        main_folder = main_folder.split(os.sep)[0]
        if os.path.isdir(os.path.join(self.dataset_path, 'camera')):
            camera_intrinsics = []
            camera_intrinsics_right = []
            json_list['names'].extend(['camera_intrinsics', 'camera_intrinsics_right'])
            json_list['types'].extend(['.txt', '.txt'])
            json_list['filters'].extend([['camera'], ['camera']])
            counter = 0
            for file in main_files:
                if ('mainz' in file or 'bielefeld' in file or 'bochum' in file or 'frankfurt' in file\
                        or 'hamburg' in file or 'hanover' in file or 'krefeld' in file or 'strasbourg' in file\
                        or 'monchengladbach' in file) and 'sequence' in main_folder:

                    transformed_file = file.split('_')
                    transformed_file[-1] = int(transformed_file[-1]) + 19 - counter
                    counter += 1
                    if counter % 30 == 0:
                        counter = 0
                    transformed_file[-1] = str(transformed_file[-1]).zfill(6)

                    calib_file = os.path.join(self.dataset_path, 'camera', '_'.join(transformed_file) + '_camera.json')
                elif 'sequence' in main_folder:
                    calib_file = os.path.join(self.dataset_path, 'camera', file[:-2] + '19' + '_camera.json')
                else:
                    calib_file = os.path.join(self.dataset_path, 'camera', file + '_camera.json')
                with open(calib_file) as f:
                    intrinsic_json = json.load(f)
                left_matrix = np.eye(4)
                left_matrix[0, 0] = intrinsic_json['intrinsic']['fx']
                left_matrix[1, 1] = intrinsic_json['intrinsic']['fy']
                left_matrix[0, 2] = intrinsic_json['intrinsic']['u0']
                left_matrix[1, 2] = intrinsic_json['intrinsic']['v0']
                left_matrix = list(left_matrix)
                left_matrix = [list(l) for l in left_matrix]
                right_matrix = left_matrix
                camera_intrinsics.append(left_matrix)
                camera_intrinsics_right.append(right_matrix)
            print('camera_intrinsics:', len(camera_intrinsics))
            print('camera_intrinsics_right:', len(camera_intrinsics_right))
            folders_list.extend([folders_list[0].copy(), folders_list[1].copy()])
            position_list.extend([position_list[0].copy(), position_list[0].copy()])
            file_list.extend([file_list[0].copy(), file_list[0].copy()])
            numerical_list.extend([camera_intrinsics, camera_intrinsics_right])

        # velocity and timestamps
        timestamps = False
        # Cityscapes_sequence
        if os.path.isdir(os.path.join(self.dataset_path, 'timestamp_sequence')):
            folders_time, files_time = self.create_filelist(['timestamp_sequence'], '.txt', ignore=ignore_list)
            folders_vel, files_vel = self.create_filelist(['vehicle_sequence'], '.json', ignore=ignore_list)
            timestamps = True
        # Cityscapes_video
        elif os.path.isdir(os.path.join(self.dataset_path, 'timestamp_allFrames')):
            folders_time, files_time = self.create_filelist(['timestamp_allFrames'], '.txt', ignore=ignore_list)
            folders_vel, files_vel = self.create_filelist(['vehicle_allFrames'], '.json', ignore=ignore_list)
            timestamps = True
        # Cityscapes_standard & extra
        elif os.path.isdir(os.path.join(self.dataset_path, 'vehicle')):
            folders_vel, files_vel = self.create_filelist(['vehicle'], '.json', ignore=ignore_list)

        if timestamps:
            times = []
            json_list['names'].extend(['timestamp'])
            json_list['types'].extend(['.txt'])
            json_list['filters'].extend([['timestamp']])
            for file in files_time:
                time = float(np.array(pd.read_csv(file, header=None).values)) / (10 ** 9)
                times.append(time)
            print('timestamps:', len(times))
            folders_list.extend([folders_time])
            file_list.extend([file_list[0].copy()])
            position_list.extend([position_list[0].copy()])
            numerical_list.extend([times])

        json_list['names'].extend(['velocity'])
        json_list['types'].extend(['.json'])
        json_list['filters'].extend([['vehicle']])
        velocities = []
        for file in files_vel:
            with open(file) as json_file:
                velocity = float(json.load(json_file)['speed'])
            velocities.append(velocity)

        print('velocities:', len(velocities))
        folders_list.extend([folders_vel])
        file_list.extend([file_list[0].copy()])
        position_list.extend([position_list[0].copy()])
        numerical_list.extend([velocities])

        # Save everything in the json_list
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class MapillaryFilelistCreator(FilelistCreator):
    """Class to create the Mapillary file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """

        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []

        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type, ignore=(os.path.join('test', 'Segmentation')))
            positions = []
            for j in range(len(files)):
                if filter == 'Segmentation':
                    positions.append([j+5000, 0, 0, j+5000])
                else:
                    positions.append([j, 0, 0, j])
            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class Gta5FilelistCreator(FilelistCreator):
    """Class to create the GTA5 file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """

        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []

        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            positions = []
            for j in range(len(files)):
                positions.append([j, 0, 0, j])
            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class SynthiaFilelistCreator(FilelistCreator):
    """Class to create the Snythia file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            positions = []
            for j in range(len(files)):
                positions.append([j, 0, 0, j])
            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class Bdd100kFilelistCreator(FilelistCreator):
    """Class to create the Bdd100k file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_file_identifier(self, filename):
        """
        Generates an identifier that separates different images from each other but is the same for corresponding
        images of different types, e.g. for corresponding color, depth and segmentation images.

        :param filename: Filename from which the identifier will be extracted
        :return: file identifier
        """
        filename = os.path.split(filename)[1]
        filename = os.path.splitext(filename)[0]
        filename = filename.split('_')
        return filename[0]

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        self.preprocess_directories_list(['color_labels'])
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            positions = []
            #for j in range(len(files)):
            #    positions.append([j, 0, 0, j])
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if i == 0:   # color image
                    positions.append((len(positions), 0, 0, j))
                    main_files.append(self._generate_file_identifier(file))
                else:   # segmentation
                    positions.append((main_files.index(self._generate_file_identifier(file)), 0, 0, j))
            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class Voc2012FilelistCreator(FilelistCreator):
    """Class to create the Pascal Voc 2012 file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            positions = []
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if i == 0: # color image
                    positions.append((len(positions), 0, 0, j))
                    main_files.append(file.split('.')[0])
                else: # segmentation
                    positions.append((main_files.index(file.split('.')[0]), 0, 0, j))

            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class A2d2FilelistCreator(FilelistCreator):
    """Class to create the a2d2 file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_file_identifier(self, filename):
        """
        Generates an identifier that separates different images from each other but is the same for corresponding
        images of different types, e.g. for corresponding color, depth and segmentation images.

        :param filename: Filename from which the identifier will be extracted
        :return: file identifier
        """
        filename = os.path.split(filename)[1]
        filename = filename.split('_')
        return filename[0] + '_' + filename[2] + '_' + filename[3].split('.')[0]

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type, ambiguous_names_to_ignore='camera_lidar_semantic')
            positions = []
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if i == 0: # color image
                    positions.append((len(positions), 0, 0, j))
                    main_files.append(self._generate_file_identifier(file))
                else: # segmentation
                    positions.append((main_files.index(self._generate_file_identifier(file)), 0, 0, j))

            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class LostAndFoundFilelistCreator(FilelistCreator):
    """Class to create the LostAndFound file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []

        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):

            folders, files = self.create_filelist(filter, type)
            if 'segmentation' in name:
                files = [f for f in files if 'color' not in f and 'instance' not in f and 'Train' not in f]
            folders_list.append(folders)
            file_list.append(files)
            positions = []
            lower_limit = [0]
            upper_limit = []
            old_frame_number = None
            new_frame_number = None
            old_seq_number = None
            new_seq_number = None
            frame_indicator_pos = {'color': -2, 'segmentation': -3}
            for file in files:
                old_frame_number = new_frame_number
                old_seq_number = new_seq_number
                img_filename = os.path.splitext(os.path.split(file)[1])[0]
                if 'color' in name:
                    new_frame_number = int(img_filename.split('_')[frame_indicator_pos['color']])
                    new_seq_number = int(img_filename.split('_')[frame_indicator_pos['color']-1])
                elif 'segmentation' in name:
                    new_frame_number = int(img_filename.split('_')[frame_indicator_pos['segmentation']])
                    new_seq_number = int(img_filename.split('_')[frame_indicator_pos['segmentation']-1])
                if old_seq_number != new_seq_number and old_seq_number is not None:
                    upper_limit.append(files.index(file) - 1)
                    lower_limit.append(files.index(file))
            upper_limit.append(len(files) - 1)
            index = 0
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if index < len(lower_limit) - 1 and j == lower_limit[index + 1]:
                    index += 1
                if i == 0:
                    positions.append((len(positions), j - lower_limit[index], upper_limit[index] - j, j))
                    main_files.append('_'.join(file.split('_')[:-1]))
                else:
                    if 'segmentation' in name:
                        positions.append((main_files.index('_'.join(file.split('_')[:-2])),
                                          j - lower_limit[index], upper_limit[index] - j, j))
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))

        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class CamVidFilelistCreator(FilelistCreator):
    """Class to create the CamVid file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_file_identifier(self, filename):
        """
        Generates an identifier that separates different images from each other but is the same for corresponding
        images of different types, e.g. for corresponding color, depth and segmentation images.

        :param filename: Filename from which the identifier will be extracted
        :return: file identifier
        """
        filename = os.path.split(filename)[1]
        filename = os.path.splitext(filename)[0]
        filename = filename.split('_')
        return filename[0] + '_' + filename[1]

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            positions = []
            lower_limit = [0]
            upper_limit = []
            frame_number = 0
            new_seq_id = None
            seq_number = 0
            for file in files:
                old_seq_id = new_seq_id
                new_seq_id = self._generate_file_identifier(file)[0]
                if new_seq_id != old_seq_id:
                    upper_limit.append(files.index(file) - 1)
                    lower_limit.append(files.index(file))
                    seq_number += 1
            upper_limit.append(len(files) - 1)
            index = 0
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if index < len(lower_limit) - 1 and j == lower_limit[index + 1]:
                    index += 1
                if i == 0:   # color image
                    positions.append((len(positions), j - lower_limit[index], upper_limit[index] - j, j))
                    main_files.append(self._generate_file_identifier(file))
                else:    # segmentation
                    positions.append((main_files.index(self._generate_file_identifier(file)), j - lower_limit[index],
                                      upper_limit[index] - j, j))

            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class Make3dFilelistCreator(FilelistCreator):
    """Class to create the make3d file list"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_file_identifier(self, filename):
        """
        Generates an identifier that separates different images from each other but is the same for corresponding
        images of different types, e.g. for corresponding color, depth and segmentation images.

        :param filename: Filename from which the identifier will be extracted
        :return: file identifier
        """
        filename = os.path.split(filename)[1]
        filename = os.path.splitext(filename)[0]
        filename = filename.split("img-")[-1]
        filename = filename.split("depth_sph_corr-")[-1]
        return filename

    def create_json_from_list(self, json_list, stereo_replace=None):
        """Creates a dictionary in the format of the basic_files.json.

        Takes a dictionary with the dataset-specific names and file endings and fills the dictionary self.json_dict
        with the entries from the dataset folder based on the information in the given dictionary.

        :param json_list: dataset-spicific dictionary of the form
            {'names: [list of all data categories that this dataset provides, e.g. 'color', 'depth', ...],
             'types': [list of the corresponding file types, e.g. '.png', '.txt', ...],
             'filters': [list of the corresponding filters to identify the folders for each name, e.g. 'camera', ...]}
        :param stereo_replace: not used for this dataset
        """
        folders_list = []
        file_list = []
        position_list = []
        numerical_list = []
        main_files = []
        for i, name, type, filter in zip(range(len(json_list['names'])), json_list['names'],
                                         json_list['types'], json_list['filters']):
            folders, files = self.create_filelist(filter, type)
            positions = []
            for j, file in zip(range(len(files)), files):
                file = file.split(self.dataset_path + os.sep)[1]
                file = os.path.join(*file.split(os.sep)[1:])
                if i == 0:  # color image
                    positions.append((len(positions), 0, 0, j))
                    main_files.append(self._generate_file_identifier(file))
                else:  # segmentation
                    positions.append((main_files.index(self._generate_file_identifier(file)), 0, 0, j))

            folders_list.append(folders)
            file_list.append(files)
            position_list.append(positions)
            numerical_list.append(None)
            print('name: ', name, 'num_items: ', len(files))
        json_list.update({'folders': folders_list, 'files': file_list, 'positions': position_list,
                          'numerical_values': numerical_list})
        self.json_dict = json_list


class DatasetCreator:
    """Class to create the dataset file list for different datasets"""

    def __init__(self, dataset, path=None, rewrite=False):
        """Initializes the dataset name and path

        :param dataset: name of the dataset folder
        :param path: can user-define a path and not get it automatically from get_path (not recommended)
        :param rewrite: can be set to True to ensure that the dataset is rewritten
        """
        assert dataset in SUPPORTED_DATASETS, 'Dataset not supported'
        self.data_dict = {'cityscapes': self.create_cityscapeslists,
                          'cityscapes_video': self.create_cityscapeslists,
                          'cityscapes_sequence': self.create_cityscapeslists,
                          'cityscapes_extra': self.create_cityscapeslists,
                          'cityscapes_part': self.create_cityscapeslists,
                          'kitti': self.create_kittilists,
                          'kitti_2012': self.create_kitti2012lists,
                          'kitti_2015': self.create_kitti2015lists,
                          'virtual_kitti': self.create_virtualkittilists,
                          'mapillary': self.create_mapillarylists,
                          'mapillary_by_ID': self.create_mapillarylists,
                          'gta5': self.create_gta5lists,
                          'synthia': self.create_synthialists,
                          'bdd100k': self.create_bdd100klists,
                          'voc2012': self.create_voc2012lists,
                          'a2d2': self.create_a2d2lists,
                          'lostandfound': self.create_lostandfoundlists,
                          'camvid': self.create_camvidlists,
                          'make3d': self.create_make3dlists,
                          }
        self.dataset = dataset
        if path:
            self.dataset_folder_path = path
        else:
            path_getter = gp.GetPath()
            self.dataset_folder_path = path_getter.get_data_path()
        self.filename = 'basic_files' + '.json'
        self.rewrite = rewrite

    def check_state(self):
        """Checks whether the basic_files.json already exists and contains valid filenames"""
        check = False
        if self.rewrite:
            return check
        else:
            json_file = os.path.join(self.dataset_folder_path, self.dataset, self.filename)
            is_file = os.path.isfile(os.path.join(self.dataset_folder_path, self.dataset, self.filename))
            if is_file:
                with open(json_file) as file:
                    json_data = json.load(file)
                is_valid_image = os.path.isfile(os.path.join(self.dataset_folder_path, self.dataset,
                                                             json_data['files'][0][0]))
                if is_valid_image:
                    check = True

        return check

    def create_dataset(self):
        """Executes the dataset-specific function that reads in the data and creates the basic_files.json"""
        self.data_dict[self.dataset]()

    def create_cityscapeslists(self):
        """Creates the basic_files.json for any dataset in the Cityscapes family"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        if 'extra' in self.dataset:
            json_list = {
                'names': ['color', 'color_right', 'depth', 'segmentation'],
                'types': ['.png', '.png', '.png', '.png'],
                'filters': [['leftImg8bit'],
                            ['rightImg8bit'],
                            ['disparity'],
                            ['gtCoarse']]
            }
        elif 'video' in self.dataset:
            json_list = {
                'names': ['color', 'color_right'],
                'types': ['.png', '.png'],
                'filters': [['leftImg8bit'],
                            ['rightImg8bit']]
            }
        else:
            json_list = {
                'names': ['color', 'color_right', 'depth', 'segmentation'],
                'types': ['.png', '.png', '.png', '.png'],
                'filters': [['leftImg8bit'],
                            ['rightImg8bit'],
                            ['disparity'],
                            ['gtFine']]
            }
        creator = CityscapesFilelistCreator(local_path)
        creator.preprocess_directories_list(['foggy', '_rain'])
        creator.create_json_from_list(json_list, stereo_replace={'left': 'right'})
        creator.dump_to_json(self.filename)

    def create_kittilists(self):
        """Creates the basic_files.json for any dataset in the KITTI family"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {'names': ['color', 'color_right', 'depth', 'depth_right', 'depth_improved', 'depth_improved_right',
                               'depth_processed', 'depth_processed_right', 'depth_processed_improved',
                               'depth_processed_improved_right', 'depth_completed', 'depth_completed_right',
                               'depth_completed_improved', 'depth_completed_improved_right'],
                     'types': ['.png', '.png', '.png', '.png', '.png', '.png', '.png', '.png', '.png', '.png',
                               '.png', '.png', '.png', '.png'],
                     'filters': [['Raw_data', 'image_02', 'data'],
                                 ['Raw_data', 'image_03', 'data'],
                                 ['Depth' + os.sep, 'image_02', 'data'],
                                 ['Depth' + os.sep, 'image_03', 'data'],
                                 ['Depth_improved' + os.sep, 'image_02', 'data'],
                                 ['Depth_improved' + os.sep, 'image_03', 'data'],
                                 ['Depth_processed' + os.sep, 'image_02', 'data'],
                                 ['Depth_processed' + os.sep, 'image_03', 'data'],
                                 ['Depth_processed_improved' + os.sep, 'image_02', 'data'],
                                 ['Depth_processed_improved' + os.sep, 'image_03', 'data'],
                                 ['Depth_completed' + os.sep, 'image_02', 'data'],
                                 ['Depth_completed' + os.sep, 'image_03', 'data'],
                                 ['Depth_completed_improved' + os.sep, 'image_02', 'data'],
                                 ['Depth_completed_improved' + os.sep, 'image_03', 'data']],
                     }
        creator = KITTIFilelistCreator(local_path)
        creator.preprocess_directories_list(['image_00', 'image_01', 'velodyne_points'])
        creator.create_json_from_list(json_list, stereo_replace={'image_02': 'image_03'})
        creator.dump_to_json(self.filename)

    def create_kitti2012lists(self):
        """Creates the basic_files.json for the KITTI 2012 dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {'names': ['color', 'color_right'],
                     'types': ['.png', '.png'],
                     'filters': [['image_2'],
                                 ['image_3'],
                                 ]
                     }
        creator = KITTI2015FilelistCreator(local_path)
        creator.create_json_from_list(json_list, stereo_replace={})
        creator.dump_to_json(self.filename)

    def create_kitti2015lists(self):
        """Creates the basic_files.json for the KITTI 2015 dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {'names': ['color', 'color_right', 'depth', 'depth_right',
                               'segmentation', 'flow', 'flow_noc'],
                     'types': ['.png', '.png', '.png', '.png', '.png', '.png', '.png'],
                     'filters': [['image_2'],
                                 ['image_3'],
                                 ['depth_occ_0'],
                                 ['depth_occ_1'],
                                 ['semantic'],
                                 ['flow_occ'],
                                 ['flow_noc']]
                     }
        creator = KITTI2015FilelistCreator(local_path)
        creator.preprocess_directories_list(['viz_flow'])
        creator.create_json_from_list(json_list, stereo_replace={})
        creator.dump_to_json(self.filename)

    def create_virtualkittilists(self):
        """Creates the basic_files.json for the Virtual KITTI dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {'names': ['color', 'depth', 'segmentation'],
                     'types': ['.png', '.png', '.png'],
                     'filters': [['vkitti_1.3.1_rgb'],
                                 ['vkitti_1.3.1_depthgt'],
                                 ['vkitti_1.3.1_scenegt']]
                     }
        creator = VirtualKITTIFilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_mapillarylists(self):
        """Creates the basic_files.json for the Mapillary dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.jpg', '.png'],
            'filters': ['ColorImage', 'Segmentation']
        }
        creator = MapillaryFilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_gta5lists(self):
        """Creates the basic_files.json for the GTA5 dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.png', '.png'],
            'filters': ['images', 'labels']
        }
        creator = Gta5FilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_synthialists(self):
        """Creates the basic_files.json for the Synthia dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'depth', 'segmentation'],
            'types': ['.png', '.png', '.png'],
            'filters': ['RGB', 'Depth_1_channel', ['GT_1_channel', 'LABELS']]
        }
        creator = SynthiaFilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_bdd100klists(self):
        """Creates the basic_files.json for the BDD100K dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.jpg', '.png'],
            'filters': ['images', 'labels']
        }
        creator = Bdd100kFilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_voc2012lists(self):
        """Creates the basic_files.json for the Pascal VOC 2012 dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.jpg', '.png'],
            'filters': ['JPEGImages', 'SegmentationClassAug']
        }
        creator = Voc2012FilelistCreator(local_path)
        creator.preprocess_directories_list(['__MACOSX'])
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_a2d2lists(self):
        """Creates the basic_files.json for the a2d2 dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.png', '.png'],
            'filters': [['camera', 'front_center'] , ['label', 'front_center']]
        }
        creator = A2d2FilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_lostandfoundlists(self):
        """Creates the basic_files.json for the LostAndFound dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.png', '.png'],
            'filters': ['leftImg8bit', 'gtCoarse']
        }
        creator = LostAndFoundFilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_camvidlists(self):
        """Creates the basic_files.json for the CamVid dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'segmentation'],
            'types': ['.png', '.png'],
            'filters': ['701_StillsRaw_full' , 'LabeledApproved_full']
        }
        creator = CamVidFilelistCreator(local_path)
        creator.preprocess_directories_list(['trainid'])
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)

    def create_make3dlists(self):
        """Creates the basic_files.json for the CamVid dataset"""
        local_path = os.path.join(self.dataset_folder_path, self.dataset)
        json_list = {
            'names': ['color', 'depth'],
            'types': ['.jpg', '.png'],
            'filters': ['ColorImage' , 'Depth_PNG']
        }
        creator = Make3dFilelistCreator(local_path)
        creator.create_json_from_list(json_list)
        creator.dump_to_json(self.filename)


if __name__ == '__main__':
    """this script has to be executed on each platform given the local path
    Supported datasets until now:
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
    
    Supported modes:
    
    Path can be passed optionally however you are then responsible yourself of the right format.
    Otherwise the path is taken from get_path, you may modify this file for your needs,
    which selects the right paths automatically on your machine
    
    The script creates a json file with the file names of all images which correspond 
    to a certain class and saves them in the corresponding dataset directory
    """

    dataset = 'bdd100k'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    data_creator = DatasetCreator(dataset, rewrite=True)
    check = data_creator.check_state()
    if not check:
        data_creator.create_dataset()
