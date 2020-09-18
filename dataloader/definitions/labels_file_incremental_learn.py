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

labels_apollo_lanes = ClassDefinitions([
    #           name     id trainId      category  catId hasInstances ignoreInEval            color
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
    Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
    Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),
    Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
    Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
    Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
    Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
    Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
])

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

labels_cityscape_seg_train1 = ClassDefinitions([
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
    Label(  'building'             , 11 ,      255 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,      255 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,      255 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,      255 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,      255 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,      255 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        2 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        3 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        4 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,      255 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,      255 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,      255 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_cityscape_seg_train2 = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,      255 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,      255 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        5 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        6 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        7 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        8 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        9 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,       10 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,      255 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,      255 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,      255 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,      255 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,      255 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_cityscape_seg_train2_only = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,      255 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,      255 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        0 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        1 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        2 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        3 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        4 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        5 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,      255 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,      255 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,      255 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,      255 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,      255 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_cityscape_seg_train2_eval = ClassDefinitions([
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
    Label(  'building'             , 11 ,        5 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        6 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        7 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        8 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        9 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,       10 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        2 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        3 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        4 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,      255 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,      255 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,      255 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,      255 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,      255 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_cityscape_seg_train3 = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,      255 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,      255 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,      255 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,      255 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,      255 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,      255 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,      255 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,      255 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,      255 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,      255 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
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

labels_cityscape_seg_train3_only = ClassDefinitions([
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,      255 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,      255 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,      255 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,      255 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,      255 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,      255 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,      255 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,      255 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,      255 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,      255 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,      255 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,        0 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,        1 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,        2 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,        3 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,        4 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,        5 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,        6 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,        7 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
])

labels_cityscape_seg_train3_eval = ClassDefinitions([
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
    Label(  'building'             , 11 ,        5 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        6 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        7 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        8 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        9 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,       10 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        2 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        3 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        4 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
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
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
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

labels_mapillary_seg = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            , 0       , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            , 0       , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            , 0       , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,        0 , 'construction'    , 2       , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        1 , 'construction'    , 2       , False        , False       , (128, 64,255) ),
    Label(  'bridge'                   ,  5 ,        2 , 'construction'    , 2       , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        3 , 'construction'    , 2       , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        4 , 'construction'    , 2       , True         , False       , (140,140,200) ),
    Label(  'curb'                     ,  8 ,        5 , 'construction'    , 2       , False        , False       , (196,196,196) ),
    Label(  'curb cut'                 ,  9 ,        6 , 'construction'    , 2       , False        , False       , (170,170,170) ),
    Label(  'fence'                    , 10 ,        7 , 'construction'    , 2       , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,        8 , 'construction'    , 2       , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,        9 , 'construction'    , 2       , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,       10 , 'construction'    , 2       , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,       11 , 'construction'    , 2       , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,       12 , 'construction'    , 2       , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,       13 , 'construction'    , 2       , False        , False       , (110,110,110) ),
    Label(  'sidewalk'                 , 17 ,       14 , 'construction'    , 2       , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,       15 , 'construction'    , 2       , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,       16 , 'construction'    , 2       , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,       17 , 'object'          , 3       , True         , False       , (255,255,128) ),
    Label(  'bench'                    , 21 ,       18 , 'object'          , 3       , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,       19 , 'object'          , 3       , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,       20 , 'object'          , 3       , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,       21 , 'object'          , 3       , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,       22 , 'object'          , 3       , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,       23 , 'object'          , 3       , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,       24 , 'object'          , 3       , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,       25 , 'object'          , 3       , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,       26 , 'object'          , 3       , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,       27 , 'object'          , 3       , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,       28 , 'object'          , 3       , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,       29 , 'object'          , 3       , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,       30 , 'object'          , 3       , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,       31 , 'object'          , 3       , True         , False       , (100,128,160) ),
    Label(  'motorcycle'               , 35 ,       32 , 'object'          , 3       , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,       33 , 'object'          , 3       , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,       34 , 'object'          , 3       , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,       35 , 'object'          , 3       , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,       36 , 'object'          , 3       , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,       37 , 'object'          , 3       , False        , False       , ( 70,100,150) ),
    Label(  'street light'             , 41 ,       38 , 'object'          , 3       , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,       39 , 'object'          , 3       , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,       40 , 'object'          , 3       , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,       41 , 'object'          , 3       , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,       42 , 'object'          , 3       , True         , False       , (128,128,128) ),
    Label(  'trailer'                  , 46 ,       43 , 'object'          , 3       , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,       44 , 'object'          , 3       , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,       45 , 'object'          , 3       , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,       46 , 'object'          , 3       , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,       47 , 'object'          , 3       , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,       48 , 'nature'          , 4       , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,       49 , 'nature'          , 4       , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,       50 , 'nature'          , 4       , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,       51 , 'nature'          , 4       , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,       52 , 'nature'          , 4       , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,       53 , 'nature'          , 4       , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,       54 , 'nature'          , 4       , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,       55 , 'human'           , 6       , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,       56 , 'human'           , 6       , True         , False       , (255,  0,100) ),
    Label(  'other rider'              , 60 ,       57 , 'human'           , 6       , True         , False       , (255,  0,200) ),
    Label(  'person'                   , 61 ,       58 , 'human'           , 6       , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,       59 , 'marking'         , 8       , True         , False       , (200,128,128) ),
    Label(  'lane marking - general'   , 63 ,       60 , 'marking'         , 8       , False        , False       , (255,255,255) ),
    Label(  'bird'                     , 64 ,       61 , 'animal'          , 9       , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,       62 , 'animal'          , 9       , True         , False       , (  0,192,  0) ),
])


labels_mapillary_seg_cityscapes = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        5 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        0 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,        7 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        6 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
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
    Label(  'manhole'                  , 34 ,        0 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,       17 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,       16 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        8 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,        0 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        9 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,       10 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,       14 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,        4 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        3 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        2 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
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

labels_mapillary_seg_cityscapes_train1 = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,      255 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        0 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,      255 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,      255 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,      255 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,      255 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,        0 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,      255 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,      255 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,        0 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,      255 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,      255 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,        4 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        3 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        2 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,      255 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,        0 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,        0 , 'marking'         ,       8 , False        , False       , (128, 64,128) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

labels_mapillary_seg_cityscapes_train2 = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        5 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,      255 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,        7 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        6 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,      255 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,      255 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,      255 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,      255 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        8 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,      255 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        9 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,       10 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,      255 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,      255 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,      255 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,      255 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,      255 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,      255 , 'marking'         ,       8 , False        , False       , (128, 64,128) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

labels_mapillary_seg_cityscapes_train2_only = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        0 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,      255 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,        2 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        1 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,      255 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,      255 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,      255 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,      255 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        3 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,      255 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        4 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,        5 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,      255 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,      255 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,      255 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,      255 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,      255 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,      255 , 'marking'         ,       8 , False        , False       , (128, 64,128) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

labels_mapillary_seg_cityscapes_train2_eval = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        5 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        0 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,        7 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        6 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,      255 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,      255 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,        0 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,      255 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        8 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,        0 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        9 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,       10 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,        4 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        3 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        2 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,      255 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,        0 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,        0 , 'marking'         ,       8 , False        , False       , (128, 64,128) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

labels_mapillary_seg_cityscapes_train3 = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,      255 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,      255 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,      255 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,      255 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
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
    Label(  'pole'                     , 39 ,      255 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,      255 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,      255 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,      255 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,       14 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,      255 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,      255 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,      255 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
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

labels_mapillary_seg_cityscapes_train3_eval = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,        5 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        0 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,        7 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,        6 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
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
    Label(  'manhole'                  , 34 ,        0 , 'object'          ,       3 , True         , False       , (128, 64,128) ),
    Label(  'motorcycle'               , 35 ,       17 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,       16 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,        8 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,        0 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,        9 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,       10 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,       14 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,        4 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        3 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        2 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
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

labels_mapillary_seg_cityscapes_train4 = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,      255 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,      255 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,      255 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,      255 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,      255 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,      255 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,      255 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,      255 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,        5 , 'object'          ,       3 , True         , False       , (100,128,160) ),
    Label(  'motorcycle'               , 35 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,      255 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,      255 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,      255 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,      255 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,      255 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,      255 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,      255 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,      255 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,      255 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,      255 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,        6 , 'marking'         ,       8 , False        , False       , (255,255,255) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
])

labels_mapillary_seg_cityscapes_train4_eval = ClassDefinitions([
    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color
    Label(  'car mount'                ,  0 ,      255 , 'void'            ,       0 , False        , True        , ( 32, 32, 32) ),
    Label(  'ego vehicle'              ,  1 ,      255 , 'void'            ,       0 , False        , True        , (120, 10, 10) ),
    Label(  'unlabeled'                ,  2 ,      255 , 'void'            ,       0 , False        , True        , (  0,  0,  0) ),
    Label(  'barrier'                  ,  3 ,      255 , 'construction'    ,       2 , False        , False       , ( 90,120,150) ),
    Label(  'bike lane'                ,  4 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt, aber nicht 100% eindeutig, z.b. Radstreifen
    Label(  'bridge'                   ,  5 ,      255 , 'construction'    ,       2 , False        , False       , (150,100,100) ),
    Label(  'building'                 ,  6 ,      255 , 'construction'    ,       2 , False        , False       , ( 70, 70, 70) ),
    Label(  'crosswalk - plain'        ,  7 ,        0 , 'construction'    ,       2 , True         , False       , (128, 64,128) ),
    Label(  'curb'                     ,  8 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'curb cut'                 ,  9 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ), # passt
    Label(  'fence'                    , 10 ,      255 , 'construction'    ,       2 , False        , False       , (190,153,153) ),
    Label(  'guard rail'               , 11 ,      255 , 'construction'    ,       2 , False        , False       , (180,165,180) ),
    Label(  'parking'                  , 12 ,      255 , 'construction'    ,       2 , False        , False       , (250,170,160) ),
    Label(  'pedestrian area'          , 13 ,      255 , 'construction'    ,       2 , False        , False       , ( 96, 96, 96) ),
    Label(  'rail track'               , 14 ,      255 , 'construction'    ,       2 , False        , False       , (230,150,140) ),
    Label(  'road'                     , 15 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ),
    Label(  'service lane'             , 16 ,        0 , 'construction'    ,       2 , False        , False       , (128, 64,128) ), # passt denke ich
    Label(  'sidewalk'                 , 17 ,        1 , 'construction'    ,       2 , False        , False       , (244, 35,232) ),
    Label(  'tunnel'                   , 18 ,      255 , 'construction'    ,       2 , False        , False       , (150,120, 90) ),
    Label(  'wall'                     , 19 ,      255 , 'construction'    ,       2 , False        , False       , (102,102,156) ),
    Label(  'banner'                   , 20 ,      255 , 'object'          ,       3 , True         , False       , (255,255,128) ),    #
    Label(  'bench'                    , 21 ,      255 , 'object'          ,       3 , True         , False       , (250,  0, 30) ),
    Label(  'bicycle'                  , 22 ,      255 , 'object'          ,       3 , True         , False       , (119, 11, 32) ),
    Label(  'bike rack'                , 23 ,      255 , 'object'          ,       3 , True         , False       , (100,140,180) ),
    Label(  'billboard'                , 24 ,      255 , 'object'          ,       3 , True         , False       , (220,220,220) ),
    Label(  'boat'                     , 25 ,      255 , 'object'          ,       3 , True         , False       , (150,  0,255) ),
    Label(  'bus'                      , 26 ,      255 , 'object'          ,       3 , True         , False       , (  0, 60,100) ),
    Label(  'cctv camera'              , 27 ,      255 , 'object'          ,       3 , True         , False       , (222, 40, 40) ),
    Label(  'car'                      , 28 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,142) ),
    Label(  'caravan'                  , 29 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 90) ),
    Label(  'catch basin'              , 30 ,      255 , 'object'          ,       3 , True         , False       , (220,128,128) ),
    Label(  'fire hydrant'             , 31 ,      255 , 'object'          ,       3 , True         , False       , (100,170, 30) ),
    Label(  'junction box'             , 32 ,      255 , 'object'          ,       3 , True         , False       , ( 40, 40, 40) ),
    Label(  'mailbox'                  , 33 ,      255 , 'object'          ,       3 , True         , False       , ( 33, 33, 33) ),
    Label(  'manhole'                  , 34 ,        5 , 'object'          ,       3 , True         , False       , (100,128,160) ),
    Label(  'motorcycle'               , 35 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,230) ),
    Label(  'on rails'                 , 36 ,      255 , 'object'          ,       3 , False        , False       , (  0, 80,100) ),
    Label(  'other vehicle'            , 37 ,      255 , 'object'          ,       3 , True         , False       , (128, 64, 64) ),
    Label(  'phone booth'              , 38 ,      255 , 'object'          ,       3 , True         , False       , (142,  0,  0) ),
    Label(  'pole'                     , 39 ,      255 , 'object'          ,       3 , True         , False       , (153,153,153) ),
    Label(  'pothole'                  , 40 ,        0 , 'object'          ,       3 , False        , False       , (128, 64,128) ),
    Label(  'street light'             , 41 ,      255 , 'object'          ,       3 , True         , False       , (210,170,100) ),
    Label(  'traffic light'            , 42 ,      255 , 'object'          ,       3 , True         , False       , (250,170, 30) ),
    Label(  'traffic sign (back)'      , 43 ,      255 , 'object'          ,       3 , True         , False       , (192,192,192) ),
    Label(  'traffic sign (front)'     , 44 ,      255 , 'object'          ,       3 , True         , False       , (220,220,  0) ),
    Label(  'traffic sign frame'       , 45 ,      255 , 'object'          ,       3 , True         , False       , (128,128,128) ),    #
    Label(  'trailer'                  , 46 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,110) ),
    Label(  'trash can'                , 47 ,      255 , 'object'          ,       3 , True         , False       , (140,140, 20) ),
    Label(  'truck'                    , 48 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 70) ),
    Label(  'utility pole'             , 49 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0, 80) ),
    Label(  'wheeled slow'             , 50 ,      255 , 'object'          ,       3 , True         , False       , (  0,  0,192) ),
    Label(  'mountain'                 , 51 ,      255 , 'nature'          ,       4 , False        , False       , ( 64,170, 64) ),
    Label(  'sand'                     , 52 ,      255 , 'nature'          ,       4 , False        , False       , (230,160, 50) ),
    Label(  'sky'                      , 53 ,        4 , 'nature'          ,       4 , False        , False       , ( 70,130,180) ),
    Label(  'snow'                     , 54 ,      255 , 'nature'          ,       4 , False        , False       , (190,255,255) ),
    Label(  'terrain'                  , 55 ,        3 , 'nature'          ,       4 , False        , False       , (152,251,152) ),
    Label(  'vegetation'               , 56 ,        2 , 'nature'          ,       4 , False        , False       , (107,142, 35) ),
    Label(  'water'                    , 57 ,      255 , 'nature'          ,       4 , False        , False       , (  0,170, 30) ),
    Label(  'bicyclist'                , 58 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'motorcyclist'             , 59 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'other rider'              , 60 ,      255 , 'human'           ,       6 , True         , False       , (255,  0,  0) ),
    Label(  'person'                   , 61 ,      255 , 'human'           ,       6 , True         , False       , (220, 20, 60) ),
    Label(  'lane marking - crosswalk' , 62 ,        0 , 'marking'         ,       8 , True         , False       , (128, 64,128) ),
    Label(  'lane marking - general'   , 63 ,        6 , 'marking'         ,       8 , False        , False       , (255,255,255) ),
    Label(  'bird'                     , 64 ,      255 , 'animal'          ,       9 , True         , False       , (165, 42, 42) ),
    Label(  'ground animal'            , 65 ,      255 , 'animal'          ,       9 , True         , False       , (  0,192,  0) ),
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



if __name__ == "__main__":
    labels = labels_apollo_lanes.getlabels()
    # Print all the labels
    print("List of apolloscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'noise'
    name2label = labels_apollo_lanes.getname2label()
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    id2label = labels_apollo_lanes.getid2label()
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    trainId2label = labels_apollo_lanes.gettrainid2label()
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
