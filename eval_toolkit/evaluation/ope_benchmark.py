

import numpy as np

from ..utils import success_overlap, success_error

import os
import cv2
from colorama import Style, Fore

draw_color = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (0, 255, 255), (255, 255, 0), (255, 0, 255),
              (196, 228, 255), (0, 165, 255)]


class OPEBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh + 1e-16)

    def draw(self, dataset, video_name, eval_trackers=None, width=2, font=1.5,
             colors=None, draw_gt=False, save=False, wait_key=40):
        if eval_trackers is None:
            # eval_trackers = self.dataset.tracker_names
            eval_trackers = []
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        if colors is None:
            colors = draw_color

        video = self.dataset.videos[video_name]

        gt_traj = video.gt_traj
        n_frame = len(gt_traj)

        results = {}
        for tracker_name in eval_trackers:
            if tracker_name not in video.pred_trajs:
                tracker_traj = video.load_tracker(self.dataset.tracker_path, tracker_name, False)
            else:
                tracker_traj = video.pred_trajs[tracker_name]
            results[tracker_name] = tracker_traj

        if save:
            save_path = os.path.join('draw_results', dataset, video.name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

        for idx in range(n_frame):
            frame_name = video.img_names[idx]
            img = cv2.imread(frame_name)
            if idx == 0:
                ih, iw = img.shape[:2]
            pos = int(40 * font / 1.5)
            cv2.putText(img, str(idx), (pos, pos), cv2.FONT_HERSHEY_TRIPLEX, font, (0, 255, 255), 2)
            if draw_gt:
                gt_bbox = gt_traj[idx]
                if gt_bbox[2] > 0 and gt_bbox[3] > 0:
                    bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[0], width)
                else:
                    cv2.putText(img, 'LOST', (iw - 4 * pos, pos), cv2.FONT_HERSHEY_TRIPLEX, font, (0, 0, 255), 2)

            for i in range(len(eval_trackers)):
                tracker_name = eval_trackers[i]
                pred_bbox = results[tracker_name][idx]
                bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i + 1], width)

            cv2.imshow(video.name, img)
            k = cv2.waitKey(wait_key) & 0xff
            if k == 27:
                cv2.destroyWindow(video.name)
                return 0
            if save:
                cv2.imwrite(os.path.join(save_path, frame_name.split('/')[-1]), img, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
        cv2.destroyWindow(video.name)

    def debug_auc(self, eval_trackers=None):
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path, tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                success = success_overlap(gt_traj, tracker_traj, n_frame)
                success = np.mean(success)
                success_ret_[video.name] = success
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_success(self, eval_trackers=None):
        """
        Args: 
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path, tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                success_ret_[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center = self.convert_bb_to_center(gt_traj)
                tracker_center = self.convert_bb_to_center(tracker_traj)
                thresholds = np.arange(0, 51, 1)
                precision_ret_[video.name] = success_error(gt_center, tracker_center,
                        thresholds, n_frame)
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path, 
                            tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret_[video.name] = success_error(gt_center_norm,
                        tracker_center_norm, thresholds, n_frame)
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret, precision_ret=None,
                    norm_precision_ret=None, show_video_level=False, helight_threshold=0.6, max_num=20):
        """

        :param success_ret:
        :param precision_ret:
        :param norm_precision_ret:
        :param show_video_level:
        :param helight_threshold:
        :param max_num:
        :return:
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                             key=lambda x:x[1],
                             reverse=True)[:max_num]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^9}|{:^16}|{:^11}|").format(
                "Tracker name", "Success", "Norm Precision", "Precision")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            else:
                precision = 0
            if norm_precision_ret is not None:
                norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()), axis=0)[20]
            else:
                norm_precision = 0
            print(formatter.format(tracker_name, success, norm_precision, precision))
        print('-'*len(header))

        if show_video_level and len(success_ret) < 10 \
                and precision_ret is not None \
                and len(precision_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f'{Fore.RED}{success_str}{Style.RESET_ALL}|'
                    else:
                        row += success_str+'|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str+'|'
                print(row)
            print('-'*len(header1))

    def eval_mIoU(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        mIou_ret = {}
        mIou_scen = {}
        video_scenario = self.dataset.video_scenario

        for tracker_name in eval_trackers:
            mIou_ret_ = {}
            mIou_scen_ = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                                      tracker_name, False)
                    tracker_traj = np.array(tracker_traj)
                    tracker_traj = tracker_traj[:len(gt_traj), :]
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if hasattr(video, 'absent'):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                _, iou = success_overlap(gt_traj, tracker_traj, n_frame, return_iou=True)
                mIou_ret_[video.name] = np.mean(iou[iou > -0.1])
                save_perframe_rs = True
                if save_perframe_rs:
                    per_rs_file = os.path.join(self.dataset.tracker_path, tracker_name, video.name + '_pfiou.txt')
                    if not os.path.exists(per_rs_file):
                        aa = '\n'
                        ious = [str(x) for x in iou]
                        f = open(per_rs_file, 'w')
                        f.write(aa.join(ious))
                        f.close()
                if video_scenario[video.name] in list(mIou_scen_.keys()):
                    mIou_scen_[video_scenario[video.name]] += mIou_ret_[video.name]
                else:
                    mIou_scen_[video_scenario[video.name]] = mIou_ret_[video.name]

            mIou_ret[tracker_name] = mIou_ret_
            mIou_scen[tracker_name] = mIou_scen_

        return mIou_ret, mIou_scen

    def show_result_ITB(self, mIoU_ret, mIou_scen,
                        success_ret=None, precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """

        :param mIoU_ret:
        :param mIou_scen:
        :param success_ret:
        :param precision_ret:
        :param norm_precision_ret:
        :param show_video_level:
        :param helight_threshold:
        :return:
        """
        # scenairo names
        scens = ['human-part', 'sport-ball', '3d-object', 'animal', 'uav',
                 'human-body', 'vehicle', 'sign-logo', 'cartoon']

        # sort tracker
        tracker_mIoU = {}
        for tracker_name in mIoU_ret.keys():
            mIoU = np.mean(list(mIoU_ret[tracker_name].values()))
            tracker_mIoU[tracker_name] = mIoU
            # compute the score of each scenario

        tracker_mIoU_ = sorted(tracker_mIoU.items(),
                               key=lambda x: x[1],
                               reverse=True)
        tracker_names = [x[0] for x in tracker_mIoU_]

        tracker_name_len = max((max([len(x) for x in mIoU_ret.keys()]) + 2), 12)
        header1 = ("|{:^" + str(tracker_name_len) + "}|{:^8}{:^8}{:^9}{:^9}{:^6}{:^8}{:^10}{:^7}{:^9}|{:^13}|").format(
            "Tracker", "human", "sport", "   3D   ", "      ", "   ", "human", "    ", "sign", " ", " overall ")
        header2 = ("|{:^" + str(
            tracker_name_len) + "}|{:^7}|{:^7}|{:^8}|{:^8}|{:^5}|{:^7}|{:^9}|{:^6}|{:^9}|{:^6}|{:^6}|").format(
            "name", "part", "ball", "object", "animal", "uav", "body", "vehicle", "logo", "cartoon", "mIoU", "Suc.")

        # header = ("|{:^" + str(tracker_name_len) + "}|{:^9}|{:^16}|{:^11}|{:^6}|").format(
        #     "Tracker name", "Success", "Norm Precision", "Precision", "mIoU")
        formatter = "|{:^" + str(
            tracker_name_len) + "}|{:^7.1f} {:^7.1f} {:^8.1f} {:^8.1f} {:^5.1f} {:^7.1f} {:^9.1f} {:^6.1f} {:^9.1f}|{:^6.1f}|{:^6.1f}|"
        print('-' * len(header1))
        print(header1)
        print(header2)
        print('-' * len(header1))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            if success_ret is not None:
                success = np.mean(list(success_ret[tracker_name].values()))
            else:
                success = 0
            sce_iou = mIou_scen[tracker_name]
            print(formatter.format(tracker_name, sce_iou[scens[0]] * 5, sce_iou[scens[1]] * 5, sce_iou[scens[2]] * 5,
                                   sce_iou[scens[3]] * 5,
                                   sce_iou[scens[4]] * 5, sce_iou[scens[5]] * 5, sce_iou[scens[6]] * 5,
                                   sce_iou[scens[7]] * 5,
                                   sce_iou[scens[8]] * 5, tracker_mIoU[tracker_name] * 100, success * 100))
        print('-' * len(header1))

        if show_video_level and len(success_ret) < 10 and precision_ret is not None and len(precision_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print('-' * len(header1))
            print(header1)
            print('-' * len(header1))
            print(header2)
            print('-' * len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f'{Fore.RED}{success_str}{Style.RESET_ALL}|'
                    else:
                        row += success_str + '|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str + '|'
                print(row)
            print('-' * len(header1))
