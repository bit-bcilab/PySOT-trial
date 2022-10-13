from eval_toolkit.vot20.vot20_builder import run_vot_exp
from configs.get_config import Config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ROOT = '/home/-/pysot_trial/'
# ROOT = 'E://PySOT-Trial/'
print(str(os.getcwd()))
print(ROOT)
print(str(os.path.join(ROOT, 'experiments/siamban/config.yaml')))
print(os.path.exists(os.path.join(ROOT, 'experiments/siamban/config.yaml')))
track_cfg = Config()
track_cfg.context_amount = 0.50
track_cfg.window_influence = 0.28
track_cfg.penalty_k = 0.10
track_cfg.size_lr = 0.54
track_cfg.confidence = 0.
run_vot_exp(tracker_name='SiamBAN',
            config_file=os.path.join(ROOT, 'experiments/siamban/config.yaml'),
            weight_path=os.path.join(ROOT, 'snapshot/para-0.30-0.15/checkpoint_e20.pth'),
            vis=False, track_cfg=track_cfg)
