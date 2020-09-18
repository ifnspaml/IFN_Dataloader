from dataloader.file_io.filelist_creator import DatasetCreator
from dataloader.file_io.split_creator import DatasetSplitter
from dataloader.file_io.dataset_index import create_parameter_files

# In this example, all necessary preparation steps that are needed to use the cityscapes dataset with this
# dataloader are performed. These steps consits of creating json files with the necessary information for the 
# dataloader to recognize all data correctly.
dataset = 'cityscapes'

# In order to prepare a dataset for the dataloader, the following three steps are necessary:
# 1. Create the basic_files.json
data_creator = DatasetCreator(dataset, rewrite=True)
check = data_creator.check_state()  # If rewrite=False, this will make sure that existing files will not be overwritten
if not check:
    data_creator.create_dataset()

# 2. Create the train.json, validation.json and test.json from the basic_files.json
# If a dataset (e.g. KITTI) has more than one possible split in train, validation and test, tha variable split
# should contain a name of the predefined splits. Since Cityscapes has fixed splits based on the folder structure,
# we can set this variable to None.
splits = None   
data_splitter = DatasetSplitter(dataset, splits)
data_splitter.create_splits()

# 3. Create the parameters.json. It is also possible to create parameters files for all existing datasets at once
# by just executing the dataset_index.py
create_parameter_files(dataset)