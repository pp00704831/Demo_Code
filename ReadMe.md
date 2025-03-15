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
1. Download "[NH-Haze](https://drive.google.com/file/d/1iI-NqpbhXDUzct4H7EYK0hYfHwfAxUwM/view?usp=drive_link)" dataset.
2. Unzip NH-Haze.zip and put NH-Haze into './datasets' as: </br> './datasets/NH-Haze'
