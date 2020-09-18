from collections import namedtuple
import numpy as np

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label.
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


class ClassDefinitions(object):
    """This class contains the classdefintions for the segmentation masks and the
    procedures to work with them"""

    def __init__(self, classlabels):
        self.labels = classlabels
        for i, label in zip(range(len(self.labels)), self.labels):
            if isinstance(label.color, int):
                self.labels[i] = label._replace(color=tuple([int(label.color/(256.0**2)) % 256,
                                    int(label.color/256.0) % 256,
                                    int(label.color) % 256]))

    def getlabels(self):
        return self.labels

    def getname2label(self):
        name2label = {label.name: label for label in self.labels}
        return name2label

    def getid2label(self):
        id2label = {label.id: label for label in self.labels}
        return id2label

    def gettrainid2label(self):
        trainid2label = {label.trainId: label for label in reversed(self.labels)}
        return trainid2label

    def getcategory2label(self):
        category2labels = {}
        for label in self.labels:
            category = label.category
            if category in category2labels:
                category2labels[category].append(label)
            else:
                category2labels[category] = [label]

    def assureSingleInstanceName(self,name):
        # if the name is known, it is not a group
        name2label = self.getname2label()
        if name in name2label:
            return name
        # test if the name actually denotes a group
        if not name.endswith("group"):
            return None
        # remove group
        name = name[:-len("group")]
        # test if the new name exists
        if not name in name2label:
            return None
        # test if the new name denotes a label that actually has instances
        if not name2label[name].hasInstances:
            return None
        # all good then
        return name


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!


labels_cityscape_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_virtual_kitti_seg = ClassDefinitions([
    #     name         id trainId category catId hasInstances ignoreInEval color
    Label('building',  0, 2 , 'void', 0, False, True, (140, 140, 140)),
    Label('car',  0,  13 , 'void', 0, False, True, (200, 200, 200)),
    Label('car',  0,  13 , 'void', 0, False, True, (200, 205, 220)),
    Label('car',  0,  13 , 'void', 0, False, True, (200, 210, 238)),
    Label('car',  0,  13 , 'void', 0, False, True, (200, 240, 200)),
    Label('car',  0,  13 , 'void', 0, False, True, (201, 209, 240)),
    Label('car',  0,  13 , 'void', 0, False, True, (201, 236, 234)),
    Label('car',  0,  13 , 'void', 0, False, True, (202, 218, 241)),
    Label('car',  0,  13 , 'void', 0, False, True, (202, 228, 242)),
    Label('car',  0,  13 , 'void', 0, False, True, (203, 219, 201)),
    Label('car',  0,  13 , 'void', 0, False, True, (203, 224, 202)),
    Label('car',  0,  13 , 'void', 0, False, True, (203, 227, 244)),
    Label('car',  0,  13 , 'void', 0, False, True, (204, 215, 235)),
    Label('car',  0,  13 , 'void', 0, False, True, (204, 216, 236)),
    Label('car',  0,  13 , 'void', 0, False, True, (204, 229, 222)),
    Label('car',  0,  13 , 'void', 0, False, True, (204, 234, 242)),
    Label('car',  0,  13 , 'void', 0, False, True, (206, 206, 244)),
    Label('car',  0,  13 , 'void', 0, False, True, (206, 224, 239)),
    Label('car',  0,  13 , 'void', 0, False, True, (206, 244, 234)),
    Label('car',  0,  13 , 'void', 0, False, True, (207, 203, 224)),
    Label('car',  0,  13 , 'void', 0, False, True, (207, 213, 232)),
    Label('car',  0,  13 , 'void', 0, False, True, (207, 248, 202)),
    Label('car',  0,  13 , 'void', 0, False, True, (207, 249, 204)),
    Label('car',  0,  13 , 'void', 0, False, True, (208, 244, 236)),
    Label('car',  0,  13 , 'void', 0, False, True, (209, 214, 215)),
    Label('car',  0,  13 , 'void', 0, False, True, (209, 221, 235)),
    Label('car',  0,  13 , 'void', 0, False, True, (209, 235, 245)),
    Label('car',  0,  13 , 'void', 0, False, True, (210, 210, 227)),
    Label('car',  0,  13 , 'void', 0, False, True, (210, 218, 236)),
    Label('car',  0,  13 , 'void', 0, False, True, (210, 223, 206)),
    Label('car',  0,  13 , 'void', 0, False, True, (210, 227, 203)),
    Label('car',  0,  13 , 'void', 0, False, True, (211, 223, 237)),
    Label('car',  0,  13 , 'void', 0, False, True, (212, 214, 246)),
    Label('car',  0,  13 , 'void', 0, False, True, (212, 234, 247)),
    Label('car',  0,  13 , 'void', 0, False, True, (212, 238, 217)),
    Label('car',  0,  13 , 'void', 0, False, True, (213, 243, 237)),
    Label('car',  0,  13 , 'void', 0, False, True, (214, 205, 205)),
    Label('car',  0,  13 , 'void', 0, False, True, (214, 210, 230)),
    Label('car',  0,  13 , 'void', 0, False, True, (214, 226, 233)),
    Label('car',  0,  13 , 'void', 0, False, True, (215, 202, 238)),
    Label('car',  0,  13 , 'void', 0, False, True, (215, 203, 229)),
    Label('car',  0,  13 , 'void', 0, False, True, (215, 208, 249)),
    Label('car',  0,  13 , 'void', 0, False, True, (215, 215, 225)),
    Label('car',  0,  13 , 'void', 0, False, True, (216, 213, 219)),
    Label('car',  0,  13 , 'void', 0, False, True, (216, 243, 247)),
    Label('car',  0,  13 , 'void', 0, False, True, (217, 223, 228)),
    Label('car',  0,  13 , 'void', 0, False, True, (217, 234, 206)),
    Label('car',  0,  13 , 'void', 0, False, True, (217, 239, 231)),
    Label('car',  0,  13 , 'void', 0, False, True, (218, 212, 221)),
    Label('car',  0,  13 , 'void', 0, False, True, (218, 228, 231)),
    Label('car',  0,  13 , 'void', 0, False, True, (218, 231, 239)),
    Label('car',  0,  13 , 'void', 0, False, True, (219, 222, 248)),
    Label('car',  0,  13 , 'void', 0, False, True, (219, 232, 201)),
    Label('car',  0,  13 , 'void', 0, False, True, (220, 220, 224)),
    Label('car',  0,  13 , 'void', 0, False, True, (221, 202, 233)),
    Label('car',  0,  13 , 'void', 0, False, True, (221, 209, 216)),
    Label('car',  0,  13 , 'void', 0, False, True, (221, 213, 207)),
    Label('car',  0,  13 , 'void', 0, False, True, (221, 218, 232)),
    Label('car',  0,  13 , 'void', 0, False, True, (221, 248, 212)),
    Label('car',  0,  13 , 'void', 0, False, True, (222, 207, 203)),
    Label('car',  0,  13 , 'void', 0, False, True, (222, 209, 241)),
    Label('car',  0,  13 , 'void', 0, False, True, (223, 201, 249)),
    Label('car',  0,  13 , 'void', 0, False, True, (223, 217, 219)),
    Label('car',  0,  13 , 'void', 0, False, True, (224, 217, 244)),
    Label('car',  0,  13 , 'void', 0, False, True, (224, 222, 214)),
    Label('car',  0,  13 , 'void', 0, False, True, (224, 247, 233)),
    Label('car',  0,  13 , 'void', 0, False, True, (225, 225, 222)),
    Label('car',  0,  13 , 'void', 0, False, True, (225, 227, 234)),
    Label('car',  0,  13 , 'void', 0, False, True, (225, 238, 242)),
    Label('car',  0,  13 , 'void', 0, False, True, (226, 214, 214)),
    Label('car',  0,  13 , 'void', 0, False, True, (226, 230, 200)),
    Label('car',  0,  13 , 'void', 0, False, True, (227, 237, 226)),
    Label('car',  0,  13 , 'void', 0, False, True, (227, 242, 246)),
    Label('car',  0,  13 , 'void', 0, False, True, (228, 222, 218)),
    Label('car',  0,  13 , 'void', 0, False, True, (228, 226, 234)),
    Label('car',  0,  13 , 'void', 0, False, True, (228, 246, 216)),
    Label('car',  0,  13 , 'void', 0, False, True, (229, 211, 210)),
    Label('car',  0,  13 , 'void', 0, False, True, (229, 217, 243)),
    Label('car',  0,  13 , 'void', 0, False, True, (230, 207, 208)),
    Label('car',  0,  13 , 'void', 0, False, True, (230, 208, 202)),
    Label('car',  0,  13 , 'void', 0, False, True, (230, 212, 228)),
    Label('car',  0,  13 , 'void', 0, False, True, (230, 216, 248)),
    Label('car',  0,  13 , 'void', 0, False, True, (230, 220, 213)),
    Label('car',  0,  13 , 'void', 0, False, True, (231, 205, 235)),
    Label('car',  0,  13 , 'void', 0, False, True, (231, 209, 205)),
    Label('car',  0,  13 , 'void', 0, False, True, (232, 227, 239)),
    Label('car',  0,  13 , 'void', 0, False, True, (232, 246, 244)),
    Label('car',  0,  13 , 'void', 0, False, True, (233, 217, 208)),
    Label('car',  0,  13 , 'void', 0, False, True, (233, 231, 210)),
    Label('car',  0,  13 , 'void', 0, False, True, (233, 236, 230)),
    Label('car',  0,  13 , 'void', 0, False, True, (233, 237, 203)),
    Label('car',  0,  13 , 'void', 0, False, True, (235, 233, 237)),
    Label('car',  0,  13 , 'void', 0, False, True, (236, 201, 241)),
    Label('car',  0,  13 , 'void', 0, False, True, (236, 206, 211)),
    Label('car',  0,  13 , 'void', 0, False, True, (236, 214, 203)),
    Label('car',  0,  13 , 'void', 0, False, True, (236, 225, 245)),
    Label('car',  0,  13 , 'void', 0, False, True, (237, 210, 232)),
    Label('car',  0,  13 , 'void', 0, False, True, (237, 216, 204)),
    Label('car',  0,  13 , 'void', 0, False, True, (237, 221, 229)),
    Label('car',  0,  13 , 'void', 0, False, True, (238, 212, 238)),
    Label('car',  0,  13 , 'void', 0, False, True, (239, 204, 246)),
    Label('car',  0,  13 , 'void', 0, False, True, (239, 211, 249)),
    Label('car',  0,  13 , 'void', 0, False, True, (239, 221, 223)),
    Label('car',  0,  13 , 'void', 0, False, True, (239, 225, 243)),
    Label('car',  0,  13 , 'void', 0, False, True, (239, 230, 213)),
    Label('car',  0,  13 , 'void', 0, False, True, (240, 200, 230)),
    Label('car',  0,  13 , 'void', 0, False, True, (240, 245, 205)),
    Label('car',  0,  13 , 'void', 0, False, True, (241, 219, 202)),
    Label('car',  0,  13 , 'void', 0, False, True, (241, 241, 205)),
    Label('car',  0,  13 , 'void', 0, False, True, (242, 200, 245)),
    Label('car',  0,  13 , 'void', 0, False, True, (242, 208, 244)),
    Label('car',  0,  13 , 'void', 0, False, True, (242, 241, 239)),
    Label('car',  0,  13 , 'void', 0, False, True, (242, 245, 225)),
    Label('car',  0,  13 , 'void', 0, False, True, (243, 227, 205)),
    Label('car',  0,  13 , 'void', 0, False, True, (243, 232, 248)),
    Label('car',  0,  13 , 'void', 0, False, True, (244, 210, 237)),
    Label('car',  0,  13 , 'void', 0, False, True, (244, 216, 247)),
    Label('car',  0,  13 , 'void', 0, False, True, (244, 224, 206)),
    Label('car',  0,  13 , 'void', 0, False, True, (244, 229, 231)),
    Label('car',  0,  13 , 'void', 0, False, True, (245, 215, 207)),
    Label('car',  0,  13 , 'void', 0, False, True, (245, 220, 227)),
    Label('car',  0,  13 , 'void', 0, False, True, (245, 220, 240)),
    Label('car',  0,  13 , 'void', 0, False, True, (246, 211, 249)),
    Label('car',  0,  13 , 'void', 0, False, True, (247, 208, 232)),
    Label('car',  0,  13 , 'void', 0, False, True, (247, 230, 218)),
    Label('car',  0,  13 , 'void', 0, False, True, (248, 235, 238)),
    Label('car',  0,  13 , 'void', 0, False, True, (248, 239, 209)),
    Label('car',  0,  13 , 'void', 0, False, True, (249, 221, 246)),
    Label('car',  0,  13 , 'void', 0, False, True, (249, 249, 241)),
    Label('guardrail',  0, 255 , 'void', 0, False, True, (255, 100, 255)),
    Label('misc',  0, 255 , 'void', 0, False, True, (80 , 80 , 80 )),
    Label('pole',  0, 5 , 'void', 0, False, True, (255, 130, 0  )),
    Label('road',  0, 0 , 'void', 0, False, True, (100, 60 , 100)),
    Label('sky',  0, 10 , 'void', 0, False, True, (90 , 200, 255)),
    Label('terrain',  0, 9 , 'void', 0, False, True, (210, 0  , 200)),
    Label('trafficlight',  0, 6 , 'void', 0, False, True, (200, 200, 0  )),
    Label('trafficsign',  0, 7 , 'void', 0, False, True, (255, 255, 0  )),
    Label('tree',  0, 8 , 'void', 0, False, True, (0  , 199, 0  )),
    Label('truck',  0, 14 , 'void', 0, False, True, (160, 60 , 60 )),
    Label('van',  0,  15 , 'void', 0, False, True, (203, 224, 202)),
    Label('van',  0,  15 , 'void', 0, False, True, (206, 244, 234)),
    Label('van',  0,  15 , 'void', 0, False, True, (207, 248, 202)),
    Label('van',  0,  15 , 'void', 0, False, True, (210, 218, 236)),
    Label('van',  0,  15 , 'void', 0, False, True, (210, 227, 203)),
    Label('van',  0,  15 , 'void', 0, False, True, (212, 218, 230)),
    Label('van',  0,  15 , 'void', 0, False, True, (212, 234, 247)),
    Label('van',  0,  15 , 'void', 0, False, True, (216, 213, 219)),
    Label('van',  0,  15 , 'void', 0, False, True, (219, 222, 248)),
    Label('van',  0,  15 , 'void', 0, False, True, (221, 248, 212)),
    Label('van',  0,  15 , 'void', 0, False, True, (224, 217, 244)),
    Label('van',  0,  15 , 'void', 0, False, True, (227, 237, 226)),
    Label('van',  0,  15 , 'void', 0, False, True, (229, 217, 243)),
    Label('van',  0,  15 , 'void', 0, False, True, (230, 208, 202)),
    Label('van',  0,  15 , 'void', 0, False, True, (231, 205, 235)),
    Label('van',  0,  15 , 'void', 0, False, True, (232, 246, 244)),
    Label('van',  0,  15 , 'void', 0, False, True, (235, 225, 211)),
    Label('van',  0,  15 , 'void', 0, False, True, (236, 201, 241)),
    Label('van',  0,  15 , 'void', 0, False, True, (246, 224, 200)),
    Label('van',  0,  15 , 'void', 0, False, True, (247, 213, 242)),
    Label('van',  0,  15 , 'void', 0, False, True, (248, 235, 238)),
    Label('vegetation',  0, 8 , 'void', 0, False, True, (90 , 240, 0  ))
])

#TODO finalize the labels: they are not finished yet
labels_a2d2_seg = [
    #       name                     id    trainId   category            catId   hasInstances   ignoreInEval   color
    Label('Car 1'                   ,  0,      2,   'vehicle'         ,     2,   False    , True   , (255, 0, 0)),
    Label('Car 2'                   ,  1,      2,   'vehicle'         ,     2,   False    , True   ,  (200, 0, 0)),
    Label('Car 3'                   ,  2,      2,   'vehicle'         ,     2,   False    , True   ,  (150, 0, 0)),
    Label('Car 4'                   ,  3,      2,   'vehicle'         ,     2,   False    , True   ,  (128, 0, 0)),
    Label('Bicycle 1'               ,  4,      0,   'vehicle'         ,     2,   False    , True   ,  (182, 89, 6)),
    Label('Bicycle 2'               ,  5,      0,   'vehicle'         ,     2,   False    , True   ,  (150, 50, 4)),
    Label('Bicycle 3'               ,  6,      0,   'vehicle'         ,     2,   False    , True   ,  (90, 30, 1)),
    Label('Bicycle 4'               ,  7,      0,   'vehicle'         ,     2,   False    , True   ,  (90, 30, 30)),
    Label('Pedestrian 1'            ,  8,      1,   'alive'           ,     0,   False    , True   ,  (204, 153, 255)),
    Label('Pedestrian 2'            ,  9,      1,   'alive'           ,     0,   False    , True   ,  (189, 73, 155)),
    Label('Pedestrian 3'            , 10,      1,   'alive'           ,     0,   False    , True   ,  (239, 89, 191)),
    Label('Truck 1'                 ,  8,      3,   'vehicle'         ,     2,   False    , True   ,  (255, 128, 0)),
    Label('Truck 2'                 ,  9,      3,   'vehicle'         ,     2,   False    , True   ,  (200, 128, 0)),
    Label('Truck 3'                 , 10,      3,   'vehicle'         ,     2,   False    , True   ,  (150, 128, 0)),
    Label('Small vehicles 1'        , 11,      2,   'vehicle'         ,     2,   False    , True   ,  (0, 255, 0)),
    Label('Small vehicles 1'        , 12,      2,   'vehicle'         ,     2,   False    , True   ,  (0, 200, 0)),
    Label('Small vehicles 1'        , 13,      2,   'vehicle'         ,     2,   False    , True   ,  (0, 150, 0)),
    Label('Traffic signal 1'        , 14,      7,   'object'          ,     1,   False    , True   ,  (0, 128, 255)),
    Label('Traffic signal 2'        , 15,      7,   'object'          ,     1,   False    , True   ,  (30, 28, 158)),
    Label('Traffic signal 3'        , 16,      7,   'object'          ,     1,   False    , True   ,  (60, 28, 100)),
    Label('Traffic sign 1'          , 17,      7,   'object'          ,     1,   False    , True   ,  (0, 255, 255)),
    Label('Traffic sign 2'          , 18,      7,   'object'          ,     1,   False    , True   ,  (30, 220, 220)),
    Label('Traffic sign 3'          , 19,      7,   'object'          ,     1,   False    , True   ,  (60, 157, 199)),
    Label('Utility vehicle 1'       , 21,      3,   'vehicle'         ,     2,   False    , True   ,  (255, 255, 0)),
    Label('Utility vehicle 2'       , 22,      3,   'vehicle'         ,     2,   False    , True   ,  (255, 255, 200)),
    Label('Sidebars'                , 24,      7,   'object'          ,     2,   False    , True   ,  (233, 100, 0)),
    Label('Speed bumper'            , 25,      7,   'construction'    ,     3,   False    , True   ,  (110, 110, 0)),
    Label('Curbstone'               , 26,      5,   'object'          ,     1,   False    , True   ,  (128, 128, 0)),
    Label('Solid line'              , 27,     12,   'construction'    ,     3,   False    , True   ,  (255, 193, 37)),
    Label('Irrelevant signs'        , 28,      6,   'object'          ,     1,   False    , True   ,  (64, 0, 64)),
    Label('Road blocks'             , 29,      6,   'object'          ,     1,   False    , True   ,  (185, 122, 87)),
    Label('Tractor'                 , 30,      3,   'vehicle'         ,     2,   False    , True   ,  (0, 0, 100)),
    Label('Non-drivable street'     , 31,      5,   'construction'    ,     3,   False    , True   ,  (139, 99, 108)),
    Label('Zebra crossing'          , 32,     11,   'construction'    ,     3,   False    , True   ,  (210, 50, 115)),
    Label('Obstacles / trash'       , 33,      6,   'object'          ,     1,   False    , True   ,  (255, 0, 128)),
    Label('Poles'                   , 34,      6,   'object'          ,     1,   False    , True   ,  (255, 246, 143)),
    Label('RD restricted area'      , 35,      5,   'construction'    ,     3,   False    , True   ,  (150, 0, 150)),
    Label('Animals'                 , 36,      6,   'alive'           ,     0,   False    , True   ,  (204, 255, 153)),
    Label('Grid structure'          , 37,      6,   'construction'    ,     3,   False    , True   ,  (238, 162, 173)),
    Label('Signal corpus'           , 38,      7,   'object'          ,     1,   False    , True   ,  (33, 44, 177)),
    Label('Drivable cobbleston'     , 39,      4,   'construction'    ,     3,   False    , True   ,  (180, 50, 180)),
    Label('Electronic traffic'      , 40,      7,   'object'          ,     1,   False    , True   ,  (255, 70, 185)),
    Label('Slow drive area'         , 41,      5,   'construction'    ,     3,   False    , True   ,  (238, 233, 191)),
    Label('Nature object'           , 42,     10,   'object'          ,     1,   False    , True   ,  (147, 253, 194)),
    Label('Parking area'            , 43,      4,   'construction'    ,     3,   False    , True   ,  (150, 150, 200)),
    Label('Sidewalk'                , 44,      5,   'construction'    ,     3,   False    , True   ,  (180, 150, 200)),
    Label('Ego car'                 , 45,      2,   'vehicle'         ,     2,   False    , True   ,  (72, 209, 204)),
    Label('Painted driv. instr.'    , 46,      7,   'object'          ,     1,   False    , True   ,  (200, 125, 210)),
    Label('Traffic guide obj.'      , 47,      7,   'object'          ,     1,   False    , True   ,  (159, 121, 238)),
    Label('Dashed line'             , 48,     12,   'construction'    ,     3,   False    , True   ,  (128, 0, 255)),
    Label('RD normal street'        , 49,      4,   'construction'    ,     3,   False    , True   ,  (255, 0, 255)),
    Label('Sky'                     , 50,      8,   'sky'             ,     4,   False    , True   ,  (135, 206, 255)),
    Label('Buildings'               , 51,      9,   'construction'    ,     3,   False    , True   ,  (241, 230, 255)),
    Label('Blurred area'            , 52,      5,   'construction'    ,     3,   False    , True   ,  (96, 69, 143)),
    Label('Rain dirt'               , 53,      6,   'construction'    ,     3,   False    , True   ,  (53, 46, 82)),
]

labels_kitti_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_apollo_seg = ClassDefinitions([
    #     name                      id   trainId   category  catId  hasInstanceignoreInEval   color
    Label('others'              ,     0,   255   , '其他'    ,   0  ,False , True  , 0x000000 ),
    Label('rover'               ,     1,   255   , '其他'    ,   0  ,False , True  , 0X000000 ),
    Label('sky'                 ,    17,    0    , '天空'    ,   1  ,False , False , 0x4682B4 ),
    Label('car'                 ,    33,    1    , '移动物体',   2  ,True  , False , 0x00008E ),
    Label('car_groups'          ,   161,    1    , '移动物体',   2  ,True  , False , 0x00008E ),
    Label('motorbicycle'        ,    34,    2    , '移动物体',   2  ,True  , False , 0x0000E6 ),
    Label('motorbicycle_group'  ,   162,    2    , '移动物体',   2  ,True  , False , 0x0000E6 ),
    Label('bicycle'             ,    35,    3    , '移动物体',   2  ,True  , False , 0x770B20 ),
    Label('bicycle_group'       ,   163,    3    , '移动物体',   2  ,True  , False , 0x770B20 ),
    Label('person'              ,    36,    4    , '移动物体',   2  ,True  , False , 0x0080c0 ),
    Label('person_group'        ,   164,    4    , '移动物体',   2  ,True  , False , 0x0080c0 ),
    Label('rider'               ,    37,    5    , '移动物体',   2  ,True  , False , 0x804080 ),
    Label('rider_group'         ,   165,    5    , '移动物体',   2  ,True  , False , 0x804080 ),
    Label('truck'               ,    38,    6    , '移动物体',   2  ,True  , False , 0x8000c0 ),
    Label('truck_group'         ,   166,    6    , '移动物体',   2  ,True  , False , 0x8000c0 ),
    Label('bus'                 ,    39,    7    , '移动物体',   2  ,True  , False , 0xc00040 ),
    Label('bus_group'           ,   167,    7    , '移动物体',   2  ,True  , False , 0xc00040 ),
    Label('tricycle'            ,    40,    8    , '移动物体',   2  ,True  , False , 0x8080c0 ),
    Label('tricycle_group'      ,   168,    8    , '移动物体',   2  ,True  , False , 0x8080c0 ),
    Label('road'                ,    49,    9    , '平面'    ,   3  ,False , False , 0xc080c0 ),
    Label('siderwalk'           ,    50,    10   , '平面'    ,   3  ,False , False , 0xc08040 ),
    Label('traffic_cone'        ,    65,    11   , '路间障碍',   4  ,False , False , 0x000040 ),
    Label('road_pile'           ,    66,    12   , '路间障碍',   4  ,False , False , 0x0000c0 ),
    Label('fence'               ,    67,    13   , '路间障碍',   4  ,False , False , 0x404080 ),
    Label('traffic_light'       ,    81,    14   , '路边物体',   5  ,False , False , 0xc04080 ),
    Label('pole'                ,    82,    15   , '路边物体',   5  ,False , False , 0xc08080 ),
    Label('traffic_sign'        ,    83,    16   , '路边物体',   5  ,False , False , 0x004040 ),
    Label('wall'                ,    84,    17   , '路边物体',   5  ,False , False , 0xc0c080 ),
    Label('dustbin'             ,    85,    18   , '路边物体',   5  ,False , False , 0x4000c0 ),
    Label('billboard'           ,    86,    19   , '路边物体',   5  ,False , False , 0xc000c0 ),
    Label('building'            ,    97,    20   , '建筑'    ,   6  ,False , False , 0xc00080 ),
    Label('bridge'              ,    98,    255  , '建筑'    ,   6  ,False , True  , 0x808000 ),
    Label('tunnel'              ,    99,    255  , '建筑'    ,   6  ,False , True  , 0x800000 ),
    Label('overpass'            ,   100,    255  , '建筑'    ,   6  ,False , True  , 0x408040 ),
    Label('vegatation'          ,   113,    21   , '自然'    ,   7  ,False , False , 0x808040 ),
    Label('unlabeled'           ,   255,    255  , '未标注'  ,   8  ,False , True  , 0xFFFFFF ),
])

labels_gta_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_synthia_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'road'                 ,  3 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'road_work'            , 14 ,        0 , 'flat'            , 1       , False        , False        , (128, 64, 64) ),
    Label(  'sidewalk'             ,  4 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              , 13 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'lane_marking'         , 22 ,        0 , 'flat'            , 1       , False        , False        , (102,102,156) ),
    Label(  'building'             ,  2 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 21 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                ,  5 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'traffic light'        , 15 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         ,  9 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'pole'                 ,  7 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'vegetation'           ,  6 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 16 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  ,  1 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 10 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 17 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  ,  8 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 18 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 19 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 20 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 12 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 11 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
])

labels_bdd100k_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , (180,100,180) ),
    Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , (250,170,100) ),
    Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , (220,220,250) ),
    Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
    Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , (220,220,100) ),
    Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
    Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , (220,220,220) ),
    Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , (250,170,250) ),
    Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
])

labels_lostandfound_seg = ClassDefinitions([
    #       name                     id      trainId    hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,       0       , 'void'            , 0       ,False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  0 ,       0       , 'void'            , 0       ,False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  0 ,       0       , 'void'            , 0       ,False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  0 ,       0       , 'void'            , 0       ,False        , True         , (  0,  0,  0) ),
    Label(  'background'           ,  0 ,       0       , 'void'            , 0       ,False        , False        , (  0,  0,  0) ),
    Label(  'free'                 ,  1 ,       1       , 'void'            , 0       ,False        , False        , (128, 64,128) ),
    Label(  '01'                   ,  2 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '02'                   ,  3 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '03'                   ,  4 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '04'                   ,  5 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '05'                   ,  6 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '06'                   ,  7 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '07'                   ,  8 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '08'                   ,  9 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '09'                   , 10 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '10'                   , 11 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '11'                   , 12 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '12'                   , 13 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '13'                   , 14 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '14'                   , 15 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '15'                   , 16 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '16'                   , 17 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '17'                   , 18 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '18'                   , 19 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '19'                   , 20 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '20'                   , 21 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '21'                   , 22 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '22'                   , 23 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '23'                   , 24 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '24'                   , 25 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '25'                   , 26 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '26'                   , 27 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '27'                   , 28 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '28'                   , 29 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '29'                   , 30 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '30'                   , 31 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '31'                   , 32 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '32'                   , 33 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '33'                   , 34 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '34'                   , 35 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '35'                   , 36 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '36'                   , 37 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '37'                   , 38 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '38'                   , 39 ,       0       , 'void'            , 0       ,True         , False        , (  0,  0,  0) ),
    Label(  '39'                   , 40 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '40'                   , 41 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '41'                   , 42 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
    Label(  '42'                   , 43 ,       2       , 'void'            , 0       ,True         , False        , (  0,  0,142) ),
])

# Mapping according to definitions from: https://www.cityscapes-dataset.com/dataset-overview/#labeling-policy
labels_mapillary_seg_cityscapes_def = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        0 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        2 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        0 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'curb cut'                 ,  9 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'fence'                    , 10 ,        4 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        3 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,       18 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,       15 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,       13 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,      255 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,       17 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,       16 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        5 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,      255 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        6 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,        7 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,       14 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,       10 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        9 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        8 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,       12 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,       12 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,       12 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,       11 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,        0 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,        0 , 'marking'         ,       8 , False        , False       , (128, 64,128) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

# 1:1 mapping to cityscapes; ignoring extra road classes
labels_mapillary_seg_cityscapes_ign = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        2 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,      255 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'curb cut'                 ,  9 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'fence'                    , 10 ,        4 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        3 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,       18 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,       15 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,       13 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,      255 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,       17 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,       16 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        5 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,      255 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        6 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,        7 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,       14 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,       10 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        9 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        8 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,       12 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,       12 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,       12 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,       11 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,      255 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,      255 , 'marking'         ,       8 , False        , False       , (128, 64,128) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

labels_camvid_seg = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

def decode_labels(mask, num_images=1, database='Cityscapes', mode = 'id'):
    """
    Decode batch of segmentation masks.

    :param mask: result of inference after taking argmax. 3D numpy array
    :param num_images: number of images to decode from the batch.
    :param num_classes: number of classes to predict (including background).
    :param database: the underlying database of the labels (currently Pascal VOC2012 or Cityscapes).
    :return: A batch with num_images RGB images of the same size as the input. 4D numpy array
    """

    assert mode in ['id', 'trainid']
    assert database in ['Cityscapes', 'KITTI']
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)

    database_dict = {'Cityscapes': labels_cityscape_seg, 'KITTI': labels_kitti_seg}
    database_def = database_dict[database]
    mode_dict = {'id': database_def.getid2label, 'trainid': database_def.gettrainid2label}
    label_dict = mode_dict[mode]()

    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = mask[i]
        new_img = np.zeros((h, w, 3))
        for key in label_dict:
            new_img[img == key, :] = np.array(label_dict[key].color)
        outputs[i] = np.array(new_img)

    return outputs


dataset_labels = {
    'a2d2': labels_a2d2_seg,
    'bdd100k': labels_bdd100k_seg,
    'camvid': labels_camvid_seg,
    'cityscapes': labels_cityscape_seg,
    'gta5': labels_gta_seg,
    'kitti': labels_kitti_seg,
    'lostandfound': labels_lostandfound_seg,
    'mapillary': labels_mapillary_seg_cityscapes_def,
    'synthia': labels_synthia_seg,
    'virtual_kitti': labels_virtual_kitti_seg,
}

if __name__ == "__main__":
    # The labels for a segmentation dataset are stored in ClassDefinition objects. The easiest way to get these is by
    # using the dataset_labels dict:
    labels = dataset_labels['cityscapes'].getlabels()

    # If you don't want to use the standard labels for a dataset, you can use the according ClassDefinitions object
    non_standard_labels = labels_mapillary_seg_cityscapes_def.getlabels()

    # Print all the labels
    print("List of Cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")