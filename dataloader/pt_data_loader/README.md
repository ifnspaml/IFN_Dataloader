The code was created with Python 3.7 and is possibly not compatible to previous
versions of Python at the moment.

Datasets
========
BEFORE using the dataloader be aware that you change the PYTHONPATH to point 
on the computer vision repository as for example:

export PYTHONPATH=$PYTHONPATH:"/home/klingner/Desktop/Code/computer_vision"

To create a Dataset, use one of the classes in `specialdatasets.py`.
There are classes for KITTI datasets and Cityscape datasets, both of
which require that the dataset has been preprocessed by creating a 
JSON file containing the desired structure. At the core of the dataset
is a dictionary of the form

    {(name1, frame_index1, resolution1): [filepath11, filepath12, ...]
     (name2, frame_index2, resolution2): [filepath21, filepath22, ...]
     ... }

where `name` denotes the category like `color`, `depth` or `segmentation`. 
If the dataset deals only with images and not with videos, `frame_index` is
always `0`. `resolution` has a standard value of `-1` that can be modified e.g.
via data transforms. The values in the dictionary contain lists of image file names
corresponding to the key. The actual image file is only loaded when the 
`__getitem()__` method of the dataset is called. 

There is also a `SimpleDataset` class which simply reads the data 
directly from the folders. However, it should be noted that as of now, a
source for the labels is still needed and there is no way to
specify how the depth information is stored in the dataset. To enable
this simple reading mode, the parameter `simple_mode` must be set to True.

In all cases, the file `get_path.py` in the `file_io` folder has to be adjusted
in order to return the path to the datasets folder on the user's PC.

For testing purposes, the file `specialdatasets.py` ist executable. There are
several functions implemented in order to test the datasets and transforms.

Necessary parameters
--------------------
The following parameters must be specified when creating a new Dataset instance:

`dataset:` Name of the folder in which the dataset is saved

`trainvaltest_split:` Can be `'train'`, `'validation'` or `'test'` to specify what
type of dataset it is


Optional Parameters
-------------------
`data_transforms:` Takes a list of data transforms that are to be executed each 
time a sample from the dataset is loaded (see next section). Default: `[mytransforms.CreateScaledImage(),
mytransforms.CreateColoraug(), mytransforms.ToTensor()]`. These transforms are required
in every case. If e.g. you include depth or segmentation images, you also need to add
`mytransforms.ConvertDepth()` or `mytransforms.ConvertSegmentation()`, respectively.

`video_mode:` can be `'mono'` or `'video'` and defines if only the images or image 
sequences are to be loaded 

`stereo_mode:` can be `'mono'` or `'stereo'` and defines if the stereo images are 
to be loaded

`width:` defines the width at which the images are loaded

`height:` defines the height at which the images are loaded

`labels:` can be used if the different labels than specified in the 
`dataloader/file_io/dataset_index.py` are desired. It takes the labels as defined in the named tuples style in Cityscapes. 
Get the labels from defintions folder.
                  
`labels_mode:` can be used if the a different labels mode than specified in the 
`dataloader/file_io/dataset_index.py` is desired. It can be `'fromid'`, `'fromrgb'` or 
`'fromtrainid'` and defines if the segmentation masks are given as ID, RGB color code or trainID. If the 
segmentation is given color coded, it is strongly
recommended to convert the labels into IDs first (do it on a GPU!) and use `'fromid'` instead.

The following parameters have a standard value and therefore it is not necessary 
to assign them a value

`simple_mode:` if `True`, the Data is read directly from a folder without using 
a .json file. Default: `False`

`scales:` list of all scales at which the images should be loaded (list of exponents 
for powers of 2)

`keys_to_load:` a list of all key names that are to be loaded, e.g. 
`['color', 'segmentation', ...]`. Default: `None` (meaning all keys). In simple mode, 
this parameter has to be specified!

`keys_to_video:` defines for which keys the sequences are to be loaded

`keys_to_stereo:` defines for which keys the stereo images are supposed to be loaded

`split:` dataset split that is supposed to be loaded. Default is the complete 
dataset itself (`None`)

`video_frames:` all frames of the sequence that are supposed to be loaded. Default: `[0, -1, 1]`

`folders_to_load`: list of folders from which data should be loaded; folders not mentioned are skipped in
                    the respective set. Filter is case insensitive.
           
`files_to_load`: list of files that should be loaded; files not mentioned are skipped in the respective
                    set. File names need not be complete; filter is case insensitive.

`n_files`: How many files shall be loaded. Files are selected randomly if there are more files than n_files.
                        Seeded by numpy.random.seed()

Transforms
==========
There are several transforms available. There are LoadTransforms

    LoadRGB()
    LoadSegmentation()
    LoadDepth()
    LoadNumerics()
    
that are automatically performed every time a dataset is initialized.

When initializing a dataset, a list of other transforms can be given in the
parameter `data_transforms` e.g. like this:

    example_dataset = KITTIDataset(..., data_transforms = 
                                   [RandomHorizontalFlip(), 
                                    RandomRotate(rotation=90, fraction = 1.0),
                                    ...], ...)
 
These are transforms that are performed every time 
a sample from the dataset is loaded. 

Converting the segmentation to trainIDs
-------------------------------------
Passing ``ConvertSegmentation()`` is mandatory if segmentation images are used. However,
if the standard labels for a dataset are to be used, the arguments don't have to be specified.
They wil be included automatically in the constructor of the `BaseDataset` class.

For the sake of computational efficiency, the conversion of the segmentation ground truths into
trainIDs was moved from ``LoadSegmentation()`` to a new transform called
 
    ConvertSegmentation(labels, lables_mode, original=False)
 
This new transform is passed via ``data_transforms`` and should be passed after the data 
transforms that scale the image. That way, the segmentation is only converted on the scaled image
which is noticeably faster if the image size was reduced. Optionally, by passing the argument 
``original=True``, the segmentation can be converted on the original image. It should also appear in the
`data_transforms` list before `CreateColorAug`.

Converting the depth image
--------------------------
Passing ``ConvertDepth()`` is mandatory if depth images are used. You don't have to specify the `depth_mode`
since it will be loaded automatically in the constructor of the `BaseDataset` class from the `parameters.json`
generated by the script `dataloader/file_io/dataset_index`.

Data Transforms without pre-processing
--------------------------------------
All of these transforms include randomness and do therefore yield different results every time
they are performed. They are

    RandomExchangeStereo()
    RandomHorizontalFlip()
    RandomVerticalFlip()
    
These exchange the left and right side of a stereo image or flip the image horizontally
or vertically with a probability of 0.5.

Data Transforms that scale the image
------------------------------------
For all of the following transforms, the transform `CreateScaledImage()` must be performed first.

    RandomRotate(rotation, fraction)
    RandomTranslate(translation, fraction)
    RandomRescale(scale, fraction)
    
These transforms again include randomness. The probability that these tranforms are performed
can be adjusted with the parameter `fraction`.

    Resize(output_size)
    MultiResize(scales)
    
These transforms resize the whole image.
   
    RandomCrop(output_size)
    CenterCrop(output_size)
    SidesCrop(hw, tl)
    
These transforms crop the image either randomly, centered or with a given offset to the side.

Data Transforms that augment the image
--------------------------------------
For all of the following transforms, the transform `CreateColorAug()` must be performed first.

    ColorJitter(brightness, contrast, saturation, hue, gamma, fraction)
    GaussianBlurr(fraction, max_rad)

