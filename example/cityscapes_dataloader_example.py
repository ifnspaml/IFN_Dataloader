from torch.utils.data import DataLoader

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as mytransforms
import cv2
import torch
import numpy as np

# This examples shows how to load the cityscapes dataset with this dataloader.
dataset = 'cityscapes'          # dataset name
trainvaltest_split = 'train'    # train, validation or test

# If for a dataset several possible splits into train, validation and test exist, you have
# to specify the split name (e.g. the KITTI dataset). This is not needed for cityscapes.
split = None

# Specify the keys you want to load. These correspond to the 'names' entries in the train.json,
# validation.json and test.json files. In this example, we use all available keys for cityscapes.
keys_to_load = ['color', 'color_right', 'depth', 'segmentation', 
                'camera_intrinsics', 'camera_intrinsics_right', 'velocity']

# When loading an image, some data transforms are performed on it. These transform will alter all
# image category in the same way. At minimum, the CreateScaledImage() and CreateColoraug() have to
# be included. For each image category like depth and segmentation, the corresponding Convert-tranform
# ist also necessary.
data_transforms = [mytransforms.CreateScaledImage(),
                   mytransforms.RemoveOriginals(),
                   mytransforms.RandomCrop((1024, 2048)),
                   mytransforms.RandomHorizontalFlip(),
                   mytransforms.CreateColoraug(new_element=True),
                   mytransforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.5, gamma=0.5, fraction=1.0), # values for visualization only
                   mytransforms.ConvertDepth(),            # The convert transforms should come after the
                   mytransforms.ConvertSegmentation(),     # Scaling/Rotating/Cropping
                   mytransforms.ToTensor(),
                   ]

# With the parameters specified above, a StandardDataset can now be created. You can interate through
# it using the PyTorch DataLoader class.
# There are several optional arguments in the my_dataset class that are not featured here for the sake of
# simplicity. Note that, for example, it is possible to include the previous and/or subsequent frames
# in a video sequence using tha parameter video_frames.
my_dataset = StandardDataset(dataset,
                                split=split,
                                trainvaltest_split=trainvaltest_split,
                                keys_to_load=keys_to_load,
                                data_transforms=data_transforms,
                                )
my_loader = DataLoader(my_dataset, batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

# Print the sizes of the first 3 elements to show how the elements are indexed. Each element
# of the dataset is a dictionary with the 3-tuples as keys. The first entry corrsponds to the image
# category. The second is a video frame index which will be always zero or non-video datasets.
# The third entry is a resolution parameter showing if it is a scaled image. 
for element, _ in zip(my_loader, range(3)):
    print("###########################################################################################################")
    #regular color image
    image_tensor = np.array(element[('color', 0, 0)]) # read image tensor to numpy array
    image_trans = image_tensor[0].transpose(1, 2, 0) # transpose: opencv needs color channels at last
    image = image_trans[..., ::-1] # Revert Color Channels

    #augmented color image
    aug_img_tensor = np.array(element[('color_aug', 0, 0)])
    aug_img = aug_img_tensor[0].transpose(1, 2, 0)
    aug_img = aug_img[..., ::-1]

    # Fuse images together for displaying
    whole_img = np.concatenate((image, aug_img), 0)
    whole_img = cv2.resize(whole_img, (int(2048/2.2), int(1024*2/2.2))) #resize them for better visiablity

    cv2.imshow("augmented image", whole_img)
    cv2.waitKey()

    for key in element:
        print('Key: {}, Element Size: {}'.format(key, element[key].shape))

