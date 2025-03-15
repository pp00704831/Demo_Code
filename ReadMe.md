## Installation
```
git clone https://github.com/pp00704831/Demo_Code.git
cd Demo_Code
conda create -n Demo python=3.7
source activate Demo
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python tqdm glog scikit-image tensorboardX thop
```
## Training
Download "[NH-Haze](https://drive.google.com/file/d/1iI-NqpbhXDUzct4H7EYK0hYfHwfAxUwM/view?usp=drive_link)" dataset.
Unzip NH-Haze.zip and put NH-Haze into './datasets' as: </br> './datasets/NH-Haze'

* Run the following command for training on single GPU
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py
```

* Run the following command for training on multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## Testing
* Run the following command for testing
```
CUDA_VISIBLE_DEVICES=0 python test.py
```
Results will be saved in the file './out'
