from eval_toolkit.vot20.vot20_builder import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# ROOT = '/home/-/pysot_trail/'
ROOT = 'E://PySOT-Trial/'
print(str(os.getcwd()))
print(ROOT)
print(str(os.path.join(ROOT, 'experiments/transt/config.yaml')))
print(os.path.exists(os.path.join(ROOT, 'experiments/transt/config.yaml')))
run_vot_exp(tracker_name='TransT',
            config_file=os.path.join(ROOT, 'experiments/transt/config.yaml'),
            weight_path=os.path.join(ROOT, 'experiments/transt/transt.pth'),
            vis=False)
