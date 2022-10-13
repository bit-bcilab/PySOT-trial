

import json
from os.path import exists


def clear_ybb():
    with open('json_labels/ybb-val.json') as f:
        data = json.load(f)
        f.close()
    old_num = len(data.keys())
    root = '/home/Data/training_dataset/yt_bb/crop511/'
    new_data = {}
    for video in data:
        path = root + video
        if exists(path):
            new_data[video] = data[video]
    new_num = len(new_data.keys())
    print('%d videos to %d videos' % (old_num, new_num))
    file = json.dumps(new_data, indent=4)
    fileObject = open('ybb-val-new.json', 'w')
    fileObject.write(file)
    fileObject.close()


if __name__ == '__main__':
    clear_ybb()
    pass
