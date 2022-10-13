

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from .draw_utils import COLOR, LINE_STYLE, DRAW_COLOR


def draw(dataset, dataset_name, video_name, eval_trackers=None, width=2, font=1.5,
         colors=None, draw_gt=False, save=False, wait_key=40):
    if eval_trackers is None:
        # eval_trackers = self.dataset.tracker_names
        eval_trackers = []
    if isinstance(eval_trackers, str):
        eval_trackers = [eval_trackers]

    if colors is None:
        colors = DRAW_COLOR
    is_vot = ('VOTDataset' in str(type(dataset)))
    video = dataset.videos[video_name]

    gt_traj = video.gt_traj
    n_frame = len(gt_traj)

    results = {}
    for tracker_name in eval_trackers:
        if tracker_name not in video.pred_trajs:
            tracker_traj = video.load_tracker(dataset.tracker_path, tracker_name, False)
        else:
            tracker_traj = video.pred_trajs[tracker_name]
        if is_vot:
            results[tracker_name] = tracker_traj[0]
        else:
            results[tracker_name] = tracker_traj

    if save:
        save_path = os.path.join('draw_results', dataset_name, video.name)
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
            if len(pred_bbox) == 4 and pred_bbox[2] > 0 and pred_bbox[3] > 0:
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


def get_plot_name(ori_name):
    if ori_name == 'UAV':
        name = 'UAV-123'
    elif ori_name == 'OTB100':
        name = 'OTB-100'
    else:
        name = ori_name
    return name


def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
                           norm_precision_ret=None, bold_name=None, axis=[0, 1], save=False, save_format='eps',
                           linewidth=2, title_fontsize=14, x_fontsize=15, y_fontsize=15, legend_fontsize=12.5):
    """
    保存格式有 pdf, eps, png,前两种为无损保存矢量图
    """
    # 刻度线内向
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold', fontsize=x_fontsize)
    plt.ylabel('Success rate', fontsize=y_fontsize)
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots of OPE on %s}' % (get_plot_name(name)), fontsize=title_fontsize)
        # plt.title(r'\textbf{Success plots of OPE on %s}' % ('UAV-123'))
        # plt.title(r'\textbf{Success plots of OPE on %s}' % ('OTB-100'))
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr), fontsize=title_fontsize)
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in enumerate(sorted(success.items(), key=lambda x: x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                 color=COLOR[idx], linestyle=LINE_STYLE[idx], label=label, linewidth=linewidth)
    ax.legend(loc='lower left', labelspacing=0.2, fontsize=legend_fontsize)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    # UAV画图时设置
    ymin = 0.
    ymax = 1.
    # ymax += 0.03
    ax.autoscale(enable=False)
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax+0.01, 0.1))
    plt.axis([xmin, xmax, ymin, ymax])
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    plt.show()

    # 保存图像
    if save:
        fig.savefig('{:s}-AUC-{:s}.{:s}'.format(name, attr, save_format), dpi=1200, format=save_format)

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold', fontsize=x_fontsize)
        plt.ylabel('Precision', fontsize=y_fontsize)
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (get_plot_name(name)), fontsize=title_fontsize)
            # plt.title(r'\textbf{Precision plots of OPE on %s}' % ('UAV-123'))
            # plt.title(r'\textbf{Precision plots of OPE on %s}' % ('OTB-100'))
        else:
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr), fontsize=title_fontsize)
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x: x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx], linestyle=LINE_STYLE[idx], label=label, linewidth=linewidth)
        ax.legend(loc='lower right', labelspacing=0.2, fontsize=legend_fontsize)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        # UAV画图设置
        ymin = 0.
        ymax = 1.
        # ymax += 0.03
        ax.autoscale(enable=False)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax+0.01, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()

        if save:
            fig.savefig('{:s}-Pr-{:s}.{:s}'.format(name, attr, save_format), dpi=1200, format=save_format)

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        ax.grid(b=True)
        plt.xlabel('Location error threshold', fontsize=x_fontsize)
        plt.ylabel('Normalized Precision', fontsize=y_fontsize)
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (get_plot_name(name)), fontsize=title_fontsize)
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr), fontsize=title_fontsize)
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                     color=COLOR[idx], linestyle=LINE_STYLE[idx], label=label, linewidth=linewidth)
        ax.legend(loc='lower right', labelspacing=0.2, fontsize=legend_fontsize)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        # ymax += 0.03
        # plt.axis([xmin, xmax, ymin, ymax])
        # plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        # plt.yticks(np.arange(ymin, ymax, 0.1))
        ymin = 0.
        ymax = 1.
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax+0.01, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()

        if save:
            fig.savefig('{:s}-NP-{:s}.{:s}'.format(name, attr, save_format), dpi=1200, format=save_format)
        pass
