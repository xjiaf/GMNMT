# GMNMT
Source code for "A Novel Graph-based Multi-modal Fusion Encoder for Neural Machine Translation"

## Requirements

* Python: 3.6
* Pytorch == 1.1.0 or 1.2.0
* CUDA 10.0
* torchtext == 0.3.1

```bash
conda create -n "gmnmt" python=3.6
conda activate gmnmt
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install torchtext==0.3.1

pip install gdown
gdown "https://drive.google.com/uc?export=download&id=1ihwtA99M3e1476N-cCWWaRYkI6Lpj1jL"
python3 -m zipfile -e bpe_data.zip .
mv bpe_data GMNMT/
```
## Data
[One drive]  https://uofr-my.sharepoint.com/:u:/g/personal/zyang39_ur_rochester_edu/EctVpqzTSRRGknSL9uC-sRMBtFSv6XFNK2N0IePf4aHr9g?e=9Xg9aC
[GDrive]
https://drive.google.com/file/d/1ihwtA99M3e1476N-cCWWaRYkI6Lpj1jL/view?usp=sharing
