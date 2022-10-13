
import json
import os
from glob import glob

from tqdm import tqdm

from .dataset import Dataset
from .video import Video


class GOT10kVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(GOT10kVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path) if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name + '/' + self.name + '_001.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:
                    data = f.readlines()
                    num = len(data)
                    pred_traj = []
                    for i in range(num):
                        line = data[i].split('\n')[0]
                        if ',' in line:
                            line = line.split(',')
                        elif '\t' in line:
                            line = line.split('\t')
                        else:
                            line = line.split()
                        bbox = list(map(float, line))
                        pred_traj.append(bbox)
                    # pred_traj = [list(map(float, x.strip().split(',')))
                    #         for x in f.readlines()]
                    if len(pred_traj) != len(self.gt_traj):
                        print(name, len(pred_traj), len(self.gt_traj), self.name)
                        if self.name == 'CarScale':
                            pred_traj = pred_traj[:207]

                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())


class GOT10kDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(GOT10kDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = GOT10kVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          ['Out-of-Plane Rotation', 'Occlusion', 'Deformation', 'Background Clutters'])
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
