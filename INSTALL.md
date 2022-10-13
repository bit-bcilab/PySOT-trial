

### Requirments
* Conda with Python 3.7
* Nvidia GPU
* Cuda 10.2 with corresponding CUDNN
* PyTorch==1.10.0
* yacs
* pyyaml
* matplotlib
* tqdm
* opencv-python
* optuna

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name pysot python=3.7
conda activate pysot
```

#### Install pytorch
```
conda install numpy
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install opencv-python Tensorboard optuna pyyaml yacs tqdm colorama matplotlib cython tensorboardX

pip install mmcv-full_xxxxxxxxxx.whl
pip install jpeg4py
```

#### > Pytorch 1.10
1. Official Pytorch implemention of the AMP: Mixed Precision Training
2. Official Pytorch implemention of the Multi-head Attention: nn.MultiheadAttention 
3. MMCV-full library:  Convenient and Efficient cuda operations. 
For instance, no longer need to compile DCN manually

#### Install other requirements
```
pip install opencv-python Tensorboard optuna pyyaml yacs tqdm colorama matplotlib cython tensorboardX
```
缺什么库pip安装即可

若提示没有 Shapely 库, linux系统直接pip下载安装即可, windows系统手动下载并pip安装.whl文件

#### MMCV
!!!! 参考环境：python 3.7, pytorch 1.10, cuda 10.2

linux环境下, 在mmcv的下载检索页查找配对的.whl文件，进行手动安装
https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html

windows环境下, 可pip安装目录中的.whl

#### Build extensions
编译C/C++代码，如果在运行eval.py或者SiamAtt算法出问题时，可尝试在`cmd窗口`中运行以下编译代码
```
# 编译评价工具箱所用的c代码
python setup.py build_ext --inplace

# 目前已改为直接使用MMCV库提供的DCN实现，无需手动编译
# ！！编译SiamAtt算法中所利用的DCN网络层的C++源码
cd pysot/models/head/dcn/  # 切换路径到DCN源代码所在路径
python setup.py build_ext --inplace
```

#### Speed up reading images By jpeg4py library
Requirements:
1. numpy
2. libjpeg-turbo

##### Install libjpeg-turbo
###### Ubuntu
On Ubuntu, the shared library is included in libturbojpeg package:
```bash
sudo apt-get install libturbojpeg
```
Or download '.deb' file from the official libjpeg-turbo [repository](https://sourceforge.net/projects/libjpeg-turbo/files) 

###### CentOS
!!!!! On CentOS, the work is kind of hard and complicated:
```bash
# install lib
sudo yum install libjpeg-turbo-official
```
Then use
```bash
rpm -ql libjpeg-turbo-official
```
to find where the lib was installed and copy the path of "libturbojpeg.so.0" 
(like "/opt/libjpeg-turbo/lib64/libturbojpeg.so.0").

After the jpeg4py get installed, open the "_cffi.py" in the folder of jpeg4py 
(like "/home/user/anaconda3/envs/pysot/lib/python3.7/site-packages/jpeg4py/_cffi.py"),
then, modify function initialize at the line 196:

from
```python
def initialize(
        backends=(
        "libturbojpeg.so.0",  # for Ubuntu
        "turbojpeg.dll",  # for Windows
        "/opt/libjpeg-turbo/lib64/libturbojpeg.0.dylib"  # for Mac OS X
        )):
    ......
```       
to
```python
def initialize(
        backends=(
        "libturbojpeg.so.0",  # for Ubuntu
        "turbojpeg.dll",  # for Windows
        "/opt/libjpeg-turbo/lib64/libturbojpeg.0.dylib",  # for Mac OS X
        "/opt/libjpeg-turbo/lib64/libturbojpeg.so.0",  # for CentOS
        )):
    ......
```      

###### Windows
!!!! On Windows, download '.exe' installer from [SourceWeb](https://sourceforge.net/projects/libjpeg-turbo/files),
-vc version for Visual Studio, -gcc version for GCC,
install it and copy turbojpeg.dll to the directory from the system PATH.

##### Install jpeg4py
After successfully install libjpeg-turbo, run below to install the jpeg4py:
```bash
python -m pip install jpeg4py
```
###### Example usage:
```python
import jpeg4py as jpeg
import matplotlib.pyplot as pp

if __name__ == "__main__":
    pp.imshow(jpeg.JPEG("test.jpg").decode())
    pp.show()
```