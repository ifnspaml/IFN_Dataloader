The code was created with Python 3.7 and is possibly not compatible to previous
versions of Python at the moment.

This readme file is still under construction.

Basic Workflow
==============
If you want to create the needed json-files for a dataset, do the following:
1. Execute `filelist_creator.py` with the desired dataset sepcified in the code's `main` part.
This creates a `basic_files.json` file in the dataset folder.
2. Execute `split_creator.py` with the desired dataset sepcified in the code's `main` part.
This creates a `train.json`, `validation.json` and `test.json` file in the dataset folder.
3. Execute the `dataset_index.py` with the dataset name as an input parameter.

For some datasets, some preprocessing scripts have to be performed first. These can be found in
the `dataloader/data_preprocessing` folder.

If you want to create a scaled version of an existing dataset, execute `dataset_scaler.py` 
with the desired dataset and output size/scale factor in the code's main part.

If you want to convert the segmentation images of a dataset to image consisting of the train_ids,
use the  `trainid_converter.py`. Note that an execution of the `filelist_creator.py` afterwards will
remove the new entries from the json files.

Filelist Creator
================
There are specific FilelistCreator classes for every supported Dataset. They are designed 
s.t. they understand the specific structure of the dataset

Dataset Creator
---------------
    data_creator = DatasetCreator(dataset, rewrite)
The `DatasetCreator` class is the class that is used in the main method to include the dataset.
It contains a list of all supported datasets and checks whether the `dataset` that was given
is supported. If yes, and when the method 

    data_creator.create_dataset()
is called, the DatasetCretor calls its method corresponding to the given dataset, e.g. `self.create_kittilists()` 
for the KITTI dataset. 

The `create_dataset()` method should only be used if the `check_state()` method returns `True`, which is the case 
if there is no valid `basic_files.json` in the dataset folder or if the parameter `rewrite` is 
set to `True`, in which case a valid `basic_files.json` will be overwritten.

Structure of the `basic_files.json`
-----------------------------------
The `basic_files.json` contains a dictionary with 7 entries. The following example shows what this dictionary might
look like for a specific dataset.

    { "names": ["color", "color_right", "depth", "depth_right", "segmentation", ...],
      "types": [".png", ".png", ".png", ...],
      "filters:" ["image", "image_right", "depth", "depth_right", "semantics", ...],
      "folders": [["folder_path_1a", "folder_path_1b, ...], ["folder_path_2a", ...]],
      "files": [["file_path_1a", "file_path_1b, ...], ["file_path_2a", ...]],
      "positions": [(0,0,499,0), (1,1,498,1), ...],  [(0,0,499,0), (1,1,498,1), ...], ...],
      "numerical_values": None, None, None, None, None, [0.0, 0.058810459, ...], ...]}

The keys are always the same. The value of each key is a list as shown above. The number of list
entries can vary depending on the dataset. However, it is important that each list contains the same
number of entries. The first entry in "names" corresponds the first entry in the other lists, and so on.
For example, if the first entry in `"names"` is `"color"`, then the first entry in `"types"` will define
in which format the color images are stored, the first entry in `"folders"` lists all folders containing
color images, and so on. The meaning of the seven keys are further explained in the following subsections.

### `names`
Names of the data category. This can be 

* `"color"`: A color image from the left/right camera
* `"depth"`: A depth image from the left/right camera, usually stored as a `uint_16`
* `"segmenation"`: Segmentation images. Can have different formats, e.g. stored by IDs or
by a color mapping
* `"camera_intrinsics"`: The matrix with the intrinsic camera parameters
* Other possible names, depending on the dataset, are `"timestamp"`, `"velocity"`, `"poses"`, ...

The above list is not exhaustive and the parameters can e.g. be modified by a succeeding
`_right` to indicate the same data category but belonging to the right camera in a stereo dataset.

### `types`
File type of the corresponding name, e.g. `".jpg"`, `".png"` for images or `".txt"` for
textual data

### `filters`
List of strings by which the folders are filtered in order to map them to the corresponding name

### `folders`, `files`
List containing all folders/files in the dataset that belong to the corresponding
`name`. The mapping of folders to the `name` is done by the `FilelistCreator` using the 
`filters`.

### `positions`
`positions` contains 4-tuples, where the single entries have the following meaning:
1. global position inside the dataset (sorted by frame number and sequence)
2. number of preceding frames in the sequence
3. number of frames in the sequence after the current frame
4. local position inside the list of the elements (e.g. depth has 20000 elements but color has 40000
then the first entry will contain the mapping from depth to color and the fourth entry will contain
numbers from 0 to 20000)

### `numerical_values`
If the datatype of `name` is not an image but numerical, e.g. `"velocity"` or 
`"camera_intrinsics"`, these numerical values are stored here as a list, each entry corresponding
to one file. If the data type is not numerical, the respecive entry in `numerical_values` is `None`.

Dataset Index
=============
There are several parameters that vary from dataset to dataset. This includes the format in which depth
and segmentation images are stored or camera matrices. It is necessary to create an entry in the dictionary
for each new dataset.

Dataset Scaler
==============
The `DatasetScaler` creates a scaled version of an existing dataset. The dataset must have a `basic_files.json`.
To scale a dataset, create a `DatasetScaler` object with the unscaled dataset as a parameter
    
    scaler = DatasetScaler('unscaled_dataset')
   
To create the scaled dataset, call

    scaler.process(self, new_dataset_name, output_size, scale_factor, keys_to_convert, splits_to_adapt)

The `new_dataset_name` must be specified and the desired `output_size` as a tuple (height, width) or a 
`scale_factor` by which both dimensions will be scaled uniformly. You must not use both parameters.

The optional parameter `keys_to_convert` specifies which images types will be converted. If e.g.
only color and segmentation images are needed but no depth images, use `keys_to_convert = 
('color', 'segmentation')`. Note that you always need to include the 'color' key as this key is used to 
determine the native shape of the image. Note using this key will most definitely result in an error
message. If this parameter is not used, all keys will be converted.

The optional parameter `splits_to_adapt` can contain a tuple of split folders. The split files in these
folders will be copied into a new folder with a name that corresponds to the name of the scaled dataset folder.

All json-files from the old dataset folder will be copied into the folder of the new dataset s.t. the
new dataset can be used exactly like the old dataset. If camera intrinsics are stored in the json files,
they will be adapted to match the new image size. If the parameter `splits_to_adapt` is used, the same
holds for the json-files in the new split folders.

It is also possible to adapt a split folder to a scaled dataset after the scaled dataset was created.
Create a new `DatasetScaler` object with the unscaled dataset name as a parameter. Then, use 

    scaler.adapt_splits(self, scaled_dataset_name, split_names)

The parameter `scaled_dataset_name` must refer to an already existing scaled dataset folder. `split names`
is a tuple of split names like in the `process()` method.

Note that it is not possible in general to execute the filelist_creator on a scaled dataset, since
only image files are copied into the new folder, but no textual data like txt-files.

Train-ID Converter
==================
The `TrainIDConverter` takes all segmentation images from a dataset and performs the
`ConvertSegmentation` transform which results in a segmentation image consisting of train_ids.
These images are saved into a new folder called `segmentation_trainid` and added to all existing
json files as a new entry. It is a requirement that the `basic_files.json` has already been
generated.

    converter = TrainIDConverter('dataset')
    
To create the new folder called `segmentation_trainid` with the trainid-images and
add these images as new entries to the json files, call

    converter.process(splits_to_adapt)
    
where `splits_to_adapt` is an optional parameter that can specify split names from splits 
in sperate folders. The new entries will also be added to their json files. 

The json files are adapted in the following way: Since for each segmentation image a new
`trainid` image is created, each segmentation entry will be duplicated and the 
new entry name gets the suffix `_trainid`. The other lists in the 
dictionaries are simply copied ot modified accordingly. If for example the `basic_files.json` 
originally contains (`files`, `positions` and `numerical values` are omitted for reasons of clarity)

    { "names":  [..., "segmentation", "segmentation_right" ...],
      "types":   [..., ".png", ".png", ...],
      "filters:" [..., ["semantics"], ["semantics_right"], ...],
      "folders": [..., ["seg_path_l_1", "seg_path_l_2"], ["seg_path_r_1", "seg_path_r_2"], ...],
      ...}

then the new `basic_files.json` will contain

    { "names":  [..., "segmentation", "segmentation_right", 
                      "segmentation_trainid", "segmentation_right_trainid",  ...],
      "types":   [..., ".png", ".png", ".png", ".png", ...],
      "filters:" [..., ["semantics"], ["semantics_right"], 
                       ["segmentation_trainid", "semantics"], ["segmentation_trainid", "semantics_right"],...],
      "folders": [..., ["seg_path_l_1", "seg_path_l_2"], ["seg_path_r_1", "seg_path_r_2"],
                        ["segmentation_trainid/seg_path_l_1", "segmentation_trainid/seg_path_l_2"], 
                        ["segmentation_trainid/seg_path_r_1", "segmentation_trainid/seg_path_r_2"], ...],
      ...}

IMPORTANT NOTE: If the `filelist_creator.py` is executed on this dataset after the trainid_converter
has been executed, it will create a new `basic_files.json` without the `_trainid` entries. This can occur e.g.
if the `basic_files.json` is created at first without depth information, then the `trainid_converter`ist 
executed and at some point, someone decides to include the depth information and executes the `filelist_creator.py`
again. In this case, it is not necessary to convert all segmentation images to train_ids again. The method

    converter.adapt_json_files(splits_to_adapt)
    
will add the `_trainid` entries to the json files if a `segmentation_trainid` folder exists.
It will, however, not check if there is still a one-to-one mapping from the `segmentation` data to
the data in the `segmentation_trainid` folder. Thus, if the `filelist_creator` has modified the `segmentation`
entries in some form, you must not use this function. The parameter `splits_to_adapt` has the same meaning
as in `converter.process(splits_to_adapt)`