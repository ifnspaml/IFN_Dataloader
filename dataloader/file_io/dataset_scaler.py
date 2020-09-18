import os
import sys
import warnings
import json
from multiprocessing import Pool

import cv2
import torchvision.transforms as transforms
import PIL.Image as pil
import numpy as np

cv2.setNumThreads(0)

import dataloader.file_io.get_path as gp
import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.pt_data_loader.dataset_parameterset as dps

NUM_WORKERS = 4
IMAGE_KEYS = ('color', 'depth', 'segmentation')
SPLIT_NAMES = ('train', 'validation', 'test')
CAMERA_KEYS = ('camera_intrinsics', 'camera_intrinsics_right')

# IMPORTANT NOTE: This dataset scaler works only  reliably when the original input files are PNG files, not JPEG
def load_sample(si):
    """
    Loads the specified sample.

    :param si: dict of the form {global_index: {"color": path, "depth": path, ...}}
    :return: - the global index of the sample in the dataset
             - the sample in the standard dataloader format
             - a path dictionary that is built similar to the sample dictionary and contains the image path instead of
               an image object
    """
    set_idx, sample = si

    dataset = sample.pop('dataset')
    dataset_path = sample.pop('dataset_path')
    depth_mode = sample.pop('depth_mode')

    new_sample = dict()
    paths = dict()

    for key, content in sample.items():
        if any(key.startswith(s) for s in IMAGE_KEYS):
            filepath = os.path.join(dataset_path, content)
            filepath = filepath.replace('/', os.sep)
            filepath = filepath.replace('\\', os.sep)

            image = cv2.imread(filepath, -1)

            new_sample[key, 0, 0] = image
            paths[key, 0, 0] = content

        else:
            new_sample[key, 0, 0] = content

    load_transforms = transforms.Compose([
        mytransforms.LoadRGB(),
        mytransforms.LoadSegmentation(),
        mytransforms.LoadDepth(),
        mytransforms.LoadNumerics()
    ])

    new_sample = load_transforms(new_sample)

    return set_idx, new_sample, paths


def save_image_file(image, img_type, dataset, path):
    """
    Brings the image in the right format to be saved. Therefore, it reverses all load transforms

    :param image: PIL image
    :param img_type: color, depth, segmentation
    :param dataset: e.g. 'cityscapes'
    :param path: path of the destination path
    """

    depth_transformer = mytransforms.LoadDepth()
    segmentation_transformer = mytransforms.LoadSegmentation()

    image = np.array(image)  # convert from PIL to openCV

    if 'color' in img_type:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    elif 'depth' in img_type:
        image = depth_transformer.inverse({img_type: image})[img_type]
        # image = image.astype(np.uint16)

    elif 'segmentation' in img_type:
        image = segmentation_transformer.inverse({img_type: image})[img_type]
        # image = np.array(image, dtype=np.uint8)

    cv2.imwrite(path, image)


def get_index_from_position(position_array, pos_0):
    """
    Returns the index of the position tuple with first entry pos_0 in the pos_array

    :param position_array: An array built out of 4-tuples in the standard positions format.
    :param pos_0: Global position index (first entry of the 4-tuple) to search for
    :return: index of the tuple in pos_array that begins with pos_0. If none is found, -1 is returned.
    """
    for i, position in zip(range(len(position_array)), position_array):
        if position[0] == pos_0:
            return i
    return -1


class DatasetScaler(object):
    def __init__(self, dataset, split=None):
        self.dataset = dataset
        self.dataset_path = self._gen_dataset_path(dataset)
        if split is not None:
            self.split_path = self.dataset_path + '_' + split
        else:
            self.split_path = self.dataset_path
        parameters = dps.DatasetParameterset(dataset)
        self.depth_mode = parameters.depth_mode

    def _gen_dataset_path(self, dataset):
        path_getter = gp.GetPath()
        dataset_folder = path_getter.get_data_path()

        return os.path.join(dataset_folder, dataset)

    def _parse_json_file(self, path, keys_to_convert):
        with open(path) as fd:
            json_data = json.load(fd)

        names = json_data['names']
        files = json_data['files']
        positions = json_data['positions']
        numerical_values = json_data['numerical_values']
        # Intuition would say that this should be a list,
        # but a user could for example decide to convert only
        # the depth ground truth from KITTI which is not
        # available for every point in time, while for example
        # color is.
        # A dict indexed by integers can represent
        # this sparseness nicely.
        samples = dict()

        for name, filenames, position, numerics in zip(names, files, positions, numerical_values):
            if name not in keys_to_convert and keys_to_convert != ():
                continue

            if os.path.splitext(filenames[0])[1].upper() in ('.JPG', '.JPEG'):
                warnings.warn(
                    'This dataset uses JPEG images. Due to the lossy JPEG compression, the scaled version of the '
                    'dataset will not be exactly equal to the unscaled dataset where the images are loaded in the '
                    'normal size and scaled afterwards.')

            for (set_idx, _, _, _), filename, i in zip(position, filenames, range(len(filenames))):
                if set_idx not in samples:
                    samples[set_idx] = dict()
                    samples[set_idx]['dataset'] = self.dataset
                    samples[set_idx]['dataset_path'] = self.dataset_path
                    samples[set_idx]['depth_mode'] = self.depth_mode
                if type(numerics) != list:
                    samples[set_idx][name] = filename
                else:
                    samples[set_idx][name] = numerics[i]

        # >>> samples[30004]['depth'] # layout example
        # 'Depth/2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000001324.png'

        return samples

    def _get_samples(self, pool, keys_to_convert):
        json_path = os.path.join(self.dataset_path, 'basic_files.json')
        samples = self._parse_json_file(json_path, keys_to_convert)

        return pool.imap_unordered(load_sample, samples.items())

    def _print_progress(self, iterator):
        for i, elem in enumerate(iterator):
            if i % 100 == 0:
                print('', flush=True)

            print('.', end='')

            yield elem

    def _load_split_data(self, split_path=None):
        """

        :param split_path: Path where the split is stored. Standard is self.split_path which is the same as
                           self.dataset_path if not specified otherwise when calling the __init__()
        :return: A dictionary containing all split dictionaries
        """
        if split_path is None:
            split_path = self.split_path
        split_data = {}
        for split in SPLIT_NAMES:
            path = os.path.join(split_path, split+'.json')
            try:
                with open(path) as fd:
                    split_json_data = json.load(fd)
                    split_data[split] = split_json_data
            except:
                print(split_path)
                print('No {} data accessible'.format(split))
        return split_data

    def _get_index_from_position(self, position_array, pos_0):
        """
        Returns the index of the position tuple with first entry pos_0 in the pos_array

        :param position_array: An array built out of 4-tuples in the standard positions format.
        :param pos_0: Global position index (first entry of the 4-tuple) to search for
        :return: index of the tuple in pos_array that begins with pos_0. If none is found, None is returned.
        """
        for i, position in zip(range(len(position_array)), position_array):
            if position[0] == pos_0:
                return i
        return None

    def _adapt_camera_intrinsics_in_split_file(self, split_data_dict, camera_intrinsics):
        """
        Writes the given camera intrinsics into the given split data dictionary. Automatically searches for the right
        place in the dictionary and saves tha camera intrinsics in the 'files' entry.

        :param split_data_dict: dictionary with the split data (as saved in the json files)
        :param camera_intrinsics: dictionary with the camera intrinsics
        :return: split_data_dict with modified camera_intrinsics
        """
        for split, split_data in split_data_dict.items():
            names = split_data['names']
            positions = split_data['positions']

            for set_idx in camera_intrinsics:
                for camera_key in camera_intrinsics[set_idx]:
                    camera_index = names.index(camera_key)
                    split_index = self._get_index_from_position(positions[camera_index], set_idx)
                    if split_index is None:
                        continue
                    split_data['files'][camera_index][split_index] = camera_intrinsics[set_idx][camera_key]

        return split_data_dict

    def _adapt_splits(self, split_names, scaled_path, camera_intrinsics):
        """
        Writes new camera intrinsics into every given split data, creates a new folder for the new version based
        on the path of the scaled dataset and saves the new json files.

        :param split_names: List containing all split names that are supposed to be adapted
        :param scaled_path: Path of the scaled dataset
        :param camera_intrinsics: dict of camera intrinsics that will be written into the split json files. It has the
            form {1: 'camera': camera_matrix, 'camera_right': camera_matrix,
                  2: ...}
        :return:
        """
        for split_name in split_names:
            split_path = self.dataset_path + '_' + split_name
            scaled_split_path = scaled_path + '_' + split_name
            split_data_dict = self._load_split_data(split_path)
            if not os.path.isdir(scaled_split_path):
                os.mkdir(scaled_split_path)
            else:
                print('{} will not be overwritten!', format(scaled_split_path))

            split_data_dict = self._adapt_camera_intrinsics_in_split_file(split_data_dict, camera_intrinsics)
            for split, split_data in split_data_dict.items():
                with open(os.path.join(scaled_split_path, split + '.json'), 'w') as fd:
                    json.dump(split_data, fd)

    def process(self, new_dataset_name, output_size=None, scale_factor=None, keys_to_convert=(), splits_to_adapt=None):
        """
        Scales every image in the dataset and saves them in the specified output folder. Also creates new json files
        with adapted camera intrinsics. One can define either a scale factor or a desired output size.

        :param new_dataset_name: name of the desired output folder. It is forbidden to use an existing folder
        :param output_size: target output size as a 2-tuple (h, w)
        :param scale_factor: Both dimensions gets scaled by this factor
        :param keys_to_convert: A tuple of keys, only the images behind these keys will be converted (optional)
        :param splits_to_adapt: Splits in seperate folders that will also be copied into a new folder and have their
            camera parameters adapted (optional)
        """

        assert self.dataset != new_dataset_name

        new_path = self._gen_dataset_path(new_dataset_name)
        assert not os.path.isdir(new_path), 'You are not allowed to write into an existing dataset folder!'
        if scale_factor is not None:
            assert output_size is None
            assert isinstance(scale_factor, int)

            scale_mode = 'relative'

        elif output_size is not None:
            assert scale_factor is None
            assert isinstance(output_size, tuple)

            scale_mode = 'absolute'

            resizer = mytransforms.Resize(output_size=output_size)
        if type(splits_to_adapt) == str:
            splits_to_adapt = (splits_to_adapt,)

        camera_intrinsics = {}
        pending_writes = []

        # Scale and save the images
        with Pool(processes=NUM_WORKERS) as pool_rd, Pool(processes=NUM_WORKERS) as pool_wr:
            for set_idx, sample, paths in self._print_progress(self._get_samples(pool_rd, keys_to_convert)):
                if scale_mode == 'relative':
                    width, height = sample[('color', 0, 0)].size
                    new_size = (
                        int(height / scale_factor),
                        int(width / scale_factor)
                    )
                    resizer = mytransforms.Resize(output_size=new_size)
                sample = resizer(sample)

                for key in sample:
                    if key in paths:
                        image = sample[key]
                        new_filepath = os.path.join(new_path, paths[key])
                        os.makedirs(os.path.split(new_filepath)[0], exist_ok=True)
                        args = (image, key[0], self.dataset, new_filepath)
                        job = pool_wr.apply_async(save_image_file, args)
                        pending_writes.append(job)
                    elif key[0] in CAMERA_KEYS:
                        if set_idx not in camera_intrinsics:
                            camera_intrinsics[set_idx] = {}
                        camera_intrinsics[set_idx][key[0]] = sample[key].tolist()
                # Limit the number of pending write operations
                # by waiting for old ones to complete
                while len(pending_writes) > NUM_WORKERS:
                    pending_writes.pop(0).get()

            while pending_writes:
                pending_writes.pop(0).get()

        # Modify the json data and safe the new json files
        with open(os.path.join(self.dataset_path, 'basic_files.json')) as fd:
            basic_json_data = json.load(fd)
        names = basic_json_data['names']
        positions = basic_json_data['positions']
        for set_idx in camera_intrinsics:
            for camera_key in camera_intrinsics[set_idx]:
                camera_index = names.index(camera_key)
                basic_index = self._get_index_from_position(positions[camera_index], set_idx)
                basic_json_data['numerical_values'][camera_index][basic_index] = camera_intrinsics[set_idx][camera_key]

        with open(os.path.join(new_path, 'basic_files.json'), 'w') as fd:
            json.dump(basic_json_data, fd)

        # Modify the train, val, test.json, if present
        split_data_dict = self._load_split_data()
        split_data_dict = self._adapt_camera_intrinsics_in_split_file(split_data_dict, camera_intrinsics)
        for split, split_data in split_data_dict.items():
            with open(os.path.join(new_path, split + '.json'), 'w') as fd:
                json.dump(split_data, fd)

        # Copy the parameters.json into the new path, adapt the split list
        with open(os.path.join(self.dataset_path, 'parameters.json')) as fd:
            parameters = json.load(fd)
        parameters['splits'] = splits_to_adapt
        with open(os.path.join(new_path, 'parameters.json'), 'w') as fd:
            json.dump(parameters, fd)

        # If there are any separate split folders given, adapt them too.
        if splits_to_adapt is not None:
            self._adapt_splits(splits_to_adapt, new_path, camera_intrinsics)

    def adapt_splits(self, scaled_dataset_name, split_names):
        """
        Adapts the json files from the given splits to an already scaled dataset. This means that a new split folder
        is created and the camera parameters in the json files are adapted.

        :param scaled_dataset_name: Name of an existing scaled dataset
        :param split_names: List containing all split names that are supposed to be adapted
        """
        if type(split_names) == str:
            split_names = [split_names]
        scaled_path = self._gen_dataset_path(scaled_dataset_name)
        with open(os.path.join(scaled_path, 'basic_files.json')) as fd:
            basic_json_data_scaled = json.load(fd)
        names = basic_json_data_scaled['names']
        positions = basic_json_data_scaled['positions']
        numerical_values = basic_json_data_scaled['numerical_values']
        camera_intrinsics = {}
        for name, position, numerics in zip(names, positions, numerical_values):
            if name not in CAMERA_KEYS:
                continue
            for (set_idx, _, _, _), camera_intrinsic in zip(position, numerics):
                if set_idx not in camera_intrinsics:
                    camera_intrinsics[set_idx] = {}
                camera_intrinsics[set_idx][name] = camera_intrinsic

        with open(os.path.join(scaled_path, 'parameters.json')) as fd:
            parameters = json.load(fd)
        if parameters['splits'] is None:
            parameters['splits'] = []
        parameters['splits'].extend(split_names)
        with open(os.path.join(scaled_path, 'parameters.json'), 'w') as fd:
            json.dump(parameters, fd)

        self._adapt_splits(split_names, scaled_path, camera_intrinsics)


if __name__ == '__main__':
    # To create a scaled, version of an existing dataset, execute something like
    #   scaler = DatasetScaler('unscaled_dataset')
    #   scaler.process('scaled_dataset', scale_factor=2)
    # To adapt split folders to this dataset, use the paramter splits_to adapt. If splits shall be adapted to an
    # existing scaled dataset, execute something like
    #   scaler = DatasetScaler('unscaled_dataset')
    #   scaler.adapt_splits('scaled_dataset', split_names=('split_1', 'split_2'))
    scaler = DatasetScaler('bdd100k')
    scaler.process('bdd100k_mross_scaled', scale_factor=2)
