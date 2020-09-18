import os
import sys
import json
import cv2
import torchvision.transforms as transforms
import PIL.Image as pil
import numpy as np
from multiprocessing import Pool

import dataloader.file_io.get_path as gp
import dataloader.pt_data_loader.mytransforms as mytransforms
import dataloader.definitions.labels_file as lf

SEGMENTATION_KEYS = ('segmentation', 'segmentation_right')
SPLIT_NAMES = ('train', 'validation', 'test')
NEW_FOLDER_NAME = 'segmentation_trainid'
BASIC_FILES_KEYS = ('names', 'types', 'filters', 'folders', 'files', 'positions', 'numerical_values')
TVT_KEYS = ('names', 'types', 'folders', 'files', 'positions')
JSON_NAMES = {'basic_files': 'basic_files.json', 'train': 'train.json', 'validation': 'validation.json',
              'test': 'test.json'}
NUM_WORKERS = 4


def insert_into_json_dict(json_dict, items, pos=None):
    """
    Inserts items into a dictionary that has the standard format for dataset json files

    :param json_dict: dict into which an item is to be inserted
    :param pos: position where the new item is to be placed
    :param items: dictionary containing an item for every key in dict
    :return: dict with the additional item in position pos in every list in the dict
    """
    keys = tuple(json_dict.keys())
    assert sorted(keys) in (sorted(BASIC_FILES_KEYS), sorted(TVT_KEYS)), \
        'Not a valid dictionary: Keys {} given, but {} or {} expected'.format(keys, BASIC_FILES_KEYS, TVT_KEYS)
    assert sorted(tuple(items.keys())) == sorted(keys), 'Given items dictionary does not have fitting keys'
    for key in keys:
        json_dict[key].insert(pos, items[key])
    return json_dict


def remove_from_json_dict(json_dict, pos=None, name=None):
    """
    Removes items from a dictionary that has the standard format for dataset json files

    :param json_dict: dict from which an item will be removed
    :param pos: position of the item in every list
    :param name: alternative for position. Gets the position from the name entry.
    :return: dict with the specified entry removed in every list in the dict
    """
    keys = tuple(json_dict.keys())
    assert sorted(keys) in (sorted(BASIC_FILES_KEYS), sorted(TVT_KEYS)), 'Not a valid dictionary'
    assert (pos == None or name == None) and not (pos == None and name == None)
    if name is not None:
        pos = json_dict['names'].index(name)
    for key in keys:
        json_dict[key].pop(pos)
    return json_dict


class TrainIDConverter(object):
    def __init__(self, dataset, labels, labels_mode, split=None):
        self.dataset = dataset
        self.dataset_path = self._gen_dataset_path(dataset)
        self.labels = labels
        self.labels_mode = labels_mode
        if split is not None:
            self.split_path = self.dataset_path + '_' + split
        else:
            self.split_path = self.dataset_path

    def _gen_dataset_path(self, dataset):
        path_getter = gp.GetPath()
        dataset_folder = path_getter.get_data_path()

        return os.path.join(dataset_folder, dataset)

    def _parse_json_file(self, path):
        """
        Generates a dictionary of samples from the json file in the given path

        :param path: Path to a json file
        :return: A dictionary of the form {set_index: {'segmentation': path}, set_index_2: {'segmentation': path}, ...}
        """
        with open(path) as fd:
            json_data = json.load(fd)

        names = json_data['names']
        files = json_data['files']
        positions = json_data['positions']
        numerical_values = json_data['numerical_values']
        samples = dict()

        for name, filenames, position, in zip(names, files, positions):
            if name not in SEGMENTATION_KEYS:
                continue

            for (set_idx, _, _, _), filename, i in zip(position, filenames, range(len(filenames))):
                if set_idx not in samples:
                    samples[set_idx] = dict()
                    samples[set_idx]['dataset'] = self.dataset
                    samples[set_idx]['dataset_path'] = self.dataset_path
                    samples[set_idx][name] = filename

        # >>> samples[30004]['segmentation'] # layout example
        # 'Segmentation/2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000001324.png'

        return samples

    def _load_sample(self, si):
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

        new_sample = dict()
        paths = dict()

        for key, content in sample.items():
            assert any(key.startswith(s) for s in SEGMENTATION_KEYS)
            filepath = os.path.join(dataset_path, content)
            filepath = filepath.replace('/', os.sep)
            filepath = filepath.replace('\\', os.sep)
            image = cv2.imread(filepath, -1)
            new_sample[key, 0, 0] = image
            paths[key, 0, 0] = content

        # By using the transform ConvertSegmentation, the segmentation images will be converted to the train_id format
        load_transforms = transforms.Compose([
            mytransforms.LoadSegmentation(),
            mytransforms.ConvertSegmentation(labels=self.labels, labels_mode=self.labels_mode)
        ])
        new_sample = load_transforms(new_sample)

        return set_idx, new_sample, paths

    def _save_image_file(self, image, path):
        """
        Brings the image in the right format to be saved.

        :param image: PIL image
        :param path: path of the destination directory
        """

        image = np.array(image, dtype=np.uint8)
        cv2.imwrite(path, image)

    def _get_samples(self, pool):
        json_path = os.path.join(self.dataset_path, JSON_NAMES['basic_files'])
        samples = self._parse_json_file(json_path)

        return pool.imap_unordered(self._load_sample, samples.items())

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
            path = os.path.join(split_path, JSON_NAMES[split])
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

    def _add_new_segmentation_to_split_file(self, split_data_dict, segmentation_keys):
        """
        Duplicates every segmentation entry to a segmentation_trainid entry with the modified file paths

        :param split_data_dict: split dictionary as saved in the json file
        :param segmentation_keys: list of segmentation keys that will be dupilicated
        :return: The split_data_dict with the additional entries
        """
        for split, split_data in split_data_dict.items():
            # If there are already trainid entries in the basic_files.json, remove them
            for seg_name in segmentation_keys:
                new_seg_name = seg_name + '_trainid'
                if new_seg_name in split_data['names']:
                    basic_json_data = remove_from_json_dict(split_data, name=new_seg_name)

            # Find the index after the last segmentation entry. This is the index where the new entries will be inserted
            first_seg_key_found = False
            for i, name in zip(range(len(split_data['names'])), split_data['names']):
                if 'segmentation' in name:
                    first_seg_key_found = True
                    new_seg_index = i + 1
                elif first_seg_key_found and 'segmentation' not in name:
                    new_seg_index = i
                    break
            assert first_seg_key_found, "No segmentation keys have been found in the basic_files.json"

            # Insert the trainid entries into the json list
            segmentation_keys.reverse()
            for seg_name in segmentation_keys:
                new_entry = {}
                original_seg_index = split_data['names'].index(seg_name)
                new_seg_name = seg_name + '_trainid'
                new_entry['names'] = new_seg_name
                for key in ('types', 'positions', 'files', 'folders'):
                    old_entry = split_data[key][original_seg_index]
                    if type(old_entry) == list:
                        new_entry[key] = old_entry.copy()
                    else:
                        new_entry[key] = old_entry
                for i in range(len(new_entry['files'])):
                    old_path = new_entry['files'][i]
                    new_entry['files'][i] = os.path.join(NEW_FOLDER_NAME, old_path)
                for i in range(len(new_entry['folders'])):
                    old_path = new_entry['folders'][i]
                    new_entry['folders'][i] = os.path.join(NEW_FOLDER_NAME, old_path)
                basic_json_data = insert_into_json_dict(split_data, new_entry, new_seg_index)
        return split_data_dict

    def _adapt_splits(self, split_names, segmentation_keys):
        """
        Writes the new segmentation data into every given split.

        :param split_names: List containing all split names that are supposed to be adapted
        :param basic_data:
        :return:
        """
        for split_name in split_names:
            split_path = self.dataset_path + '_' + split_name
            split_data_dict = self._load_split_data(split_path)
            split_data_dict = self._add_new_segmentation_to_split_file(split_data_dict, segmentation_keys)
            for split, split_data in split_data_dict.items():
                with open(os.path.join(split_path, JSON_NAMES[split]), 'w') as fd:
                    json.dump(split_data, fd)

    def _adapt_json_files(self, splits_to_adapt=None, segmentation_keys=None):
        """
        Adds the new trainid segmentation images to the json files as new entries with the suffix _trainid

        :param splits_to_adapt: Splits in seperate folders that will have the segmentation_trainid entry copied into
                                their json files
        :param segmentation_keys: Segmentation names which will be copied to a new trainid entry. Default are all
                                  segmentation names in the basic_files
        """
        # Load the basic files json data
        with open(os.path.join(self.dataset_path, JSON_NAMES['basic_files'])) as fd:
            basic_json_data = json.load(fd)
        names = basic_json_data['names']
        if segmentation_keys == None:
            segmentation_keys = []
            for name in names:
                if 'segmentation' in name:
                    segmentation_keys.append(name)

        # If there are already trainid entries in the basic_files.json, remove them
        for seg_name in segmentation_keys:
            new_seg_name = seg_name + '_trainid'
            if new_seg_name in basic_json_data['names']:
                basic_json_data = remove_from_json_dict(basic_json_data, name=new_seg_name)
                segmentation_keys.remove(new_seg_name)

        # Find the index where the trainid entries will be inserted
        first_seg_key_found = False
        for i, name in zip(range(len(names)), names):
            if 'segmentation' in name:
                first_seg_key_found = True
                new_seg_index = i+1
            elif first_seg_key_found and 'segmentation' not in name:
                new_seg_index = i
                break
        assert first_seg_key_found, "No segmentation keys have been found in the basic_files.json"

        # Insert the trainid entries into the json list
        segmentation_keys.reverse()
        for seg_name in segmentation_keys:
            new_entry = {}
            original_seg_index = basic_json_data['names'].index(seg_name)
            new_seg_name = seg_name + '_trainid'
            new_entry['names'] = new_seg_name
            for key in ('types', 'positions', 'numerical_values', 'filters', 'files', 'folders'):
                old_entry = basic_json_data[key][original_seg_index]
                if type(old_entry) == list:
                    new_entry[key] = old_entry.copy()
                elif key == 'filters':
                    new_entry[key] = [old_entry]
                else:
                    new_entry[key] = old_entry

            new_entry['filters'].append(NEW_FOLDER_NAME)
            for i in range(len(new_entry['files'])):
                old_path = new_entry['files'][i]
                new_entry['files'][i] = os.path.join(NEW_FOLDER_NAME, old_path)
            for i in range(len(new_entry['folders'])):
                old_path = new_entry['folders'][i]
                new_entry['folders'][i] = os.path.join(NEW_FOLDER_NAME, old_path)
            basic_json_data = insert_into_json_dict(basic_json_data, new_entry, new_seg_index)

        with open(os.path.join(self.dataset_path, JSON_NAMES['basic_files']), 'w') as fd:
            json.dump(basic_json_data, fd)

        # Modify the train, val, test.json, if present
        split_data_dict = self._load_split_data()
        split_data_dict = self._add_new_segmentation_to_split_file(split_data_dict, segmentation_keys)
        for split, split_data in split_data_dict.items():
            with open(os.path.join(self.dataset_path, JSON_NAMES[split]), 'w') as fd:
                json.dump(split_data, fd)

        # If there are any separate split folders given, adapt them too.
        if splits_to_adapt is not None:
            self._adapt_splits(splits_to_adapt, segmentation_keys)

    def adapt_json_files(self, splits_to_adapt=None):
        """
        This function is meant for the case that the scaled segmentation files have already been created but the json
        files have been altered by the filelist creator, which does remove the segmentation_trainid entries. Using this
        function, the segmentation_trainid entries will be restored. If there is no segmentation_trainid Folder, this
        function will have no effect

        :param splits_to_adapt: Splits in seperate folders that will have the segmentation_trainid entry copied into
                                their json files
        """
        if not os.path.isdir(os.path.join(self.dataset_path, NEW_FOLDER_NAME)):
            print('No segmentation_trainid folder found in the dataset directory')
            return
        self._adapt_json_files(splits_to_adapt)

    def process(self, splits_to_adapt=None):
        """
        Converts all segmentation images in the dataset to train_ids and saves them into a new folder in the dataset
        root directory. The new segmentation images will be added to the json_files as a new entry
        "segmentation_trainid"

        :param splits_to_adapt: Splits in seperate folders that will have the segmentation_trainid entry copied into
                                their json files
        """

        new_path = os.path.join(self.dataset_path, NEW_FOLDER_NAME)

        if type(splits_to_adapt) == str:
            splits_to_adapt = (splits_to_adapt,)

        camera_intrinsics = {}
        pending_writes = []
        segmentation_keys = []

        # Save the images
        with Pool(processes=NUM_WORKERS) as pool_rd, Pool(processes=NUM_WORKERS) as pool_wr:
            for set_idx, sample, paths in self._print_progress(self._get_samples(pool_rd)):
                for key in sample:
                    if key[0] not in segmentation_keys:
                        segmentation_keys.append(key[0])
                    image = sample[key]
                    new_filepath = os.path.join(new_path, paths[key])
                    os.makedirs(os.path.split(new_filepath)[0], exist_ok=True)
                    args = (image, new_filepath)
                    job = pool_wr.apply_async(self._save_image_file, args)
                    pending_writes.append(job)
                # Limit the number of pending write operations
                # by waiting for old ones to complete
                while len(pending_writes) > NUM_WORKERS:
                    pending_writes.pop(0).get()

            while pending_writes:
                pending_writes.pop(0).get()
        self._adapt_json_files(splits_to_adapt, segmentation_keys)


if __name__ == '__main__':
    pass
    # Example workflow:
    # To create a new folder containing the segmentation images encoded as train_ids, perform
    # converter = TrainIDConverter('cityscapes', labels=lf.labels_cityscape_seg.getlabels(), labels_mode='fromid')
    # converter.process()
    #
    # If you have already created the train_id images but have to update the json files, use
    # converter.adapt_json_files()

    # converter = TrainIDConverter('cityscapes', labels=lf.labels_cityscape_seg.getlabels(), labels_mode='fromid')
    # converter.process()
    # converter.adapt_json_files()