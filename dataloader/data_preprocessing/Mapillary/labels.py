#+++++++++++++++++++++++++++++++++++++++++++++++
#
# Author: Philipp Donn
#
# Date: 11/10/2019
#
# Supervisor: Marvin Klingner
# 
#+++++++++++++++++++++++++++++++++++++++++++++++
#
# Create labels_file.py from config.json 
#
#+++++++++++++++++++++++++++++++++++++++++++++++

import json
from operator import itemgetter


def write_line(f, name, id, trainId, category, catId, hasInstances, ignoreInEval, color):
    f.write("    Label(  {},{:3d} ,{:9d} , {}, {}, {}, {}, ({:3d},{:3d},{:3d}) ),\n".format("'{}'".format(name).ljust(27).lower(), id, trainId, "'{}'".format(category).ljust(18), '{}'.format(catId).ljust(8), str(hasInstances).ljust(13), ignoreInEval.ljust(12), color[0], color[1], color[2]))


def create_labels_file():
    with open('config.json', 'r') as f:
        config = json.load(f)

    labels = config['labels']

    with open('labels_file.py', 'w') as f:
        id = 0
        train_id = 0
        animal = []
        construction = []
        human = []
        marking = []
        nature = []
        object = []
        void = []
        for label in labels:
            if label['name'].split('-')[0] == 'animal':
                animal.append(label)
            elif label['name'].split('-')[0] == 'construction':
                construction.append(label)
            elif label['name'].split('-')[0] == 'human':
                human.append(label)
            elif label['name'].split('-')[0] == 'marking':
                marking.append(label)
            elif label['name'].split('-')[0] == 'nature':
                nature.append(label)
            elif label['name'].split('-')[0] == 'object':
                object.append(label)
            elif label['name'].split('-')[0] == 'void':
                void.append(label)
            else:
                print(label['name'].split('-')[0])

        animal = sorted(animal, key=itemgetter('readable'))
        construction = sorted(construction, key=itemgetter('readable'))
        human = sorted(human, key=itemgetter('readable'))
        marking = sorted(marking, key=itemgetter('readable'))
        nature = sorted(nature, key=itemgetter('readable'))
        object = sorted(object, key=itemgetter('readable'))
        void = sorted(void, key=itemgetter('readable'))
        f.write('labels_mapillary_seg = ClassDefinitions([\n    #       name                         id    trainId   category            catId     hasInstances   ignoreInEval  color\n')
        for label in void:
            cat_id = 0
            write_line(f, label['readable'], id, 255, label['name'].split('-')[0], cat_id, label['instances'], 'True', label['color'])
            id += 1
        for label in construction:
            cat_id = 2
            write_line(f, label['readable'], id, train_id, label['name'].split('-')[0], cat_id, label['instances'], 'False', label['color'])
            id += 1
            train_id += 1
        for label in object:
            cat_id = 3
            write_line(f, label['readable'], id, train_id, label['name'].split('-')[0], cat_id, label['instances'], 'False', label['color'])
            id += 1
            train_id += 1
        for label in nature:
            cat_id = 4
            write_line(f, label['readable'], id, train_id, label['name'].split('-')[0], cat_id, label['instances'], 'False', label['color'])
            id += 1
            train_id += 1
        for label in human:
            cat_id = 6
            write_line(f, label['readable'], id, train_id, label['name'].split('-')[0], cat_id, label['instances'], 'False', label['color'])
            id += 1
            train_id += 1
        for label in marking:
            cat_id = 8
            write_line(f, label['readable'], id, train_id, label['name'].split('-')[0], cat_id, label['instances'], 'False', label['color'])
            id += 1
            train_id += 1
        for label in animal:
            cat_id = 9
            write_line(f, label['readable'], id, train_id, label['name'].split('-')[0], cat_id, label['instances'], 'False', label['color'])
            id += 1
            train_id += 1
        f.write('])\n')


if __name__ == '__main__':
    create_labels_file()
