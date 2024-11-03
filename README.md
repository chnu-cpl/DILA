
# 🌈 **Installation and Get Started**

## 🏆 **Dependencies:**

* Linux

* CUDA

* Python 

* PyTorch 

* TorchVision 

* mmcv-full 

* numpy 
  
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)

## 🛒 **Install:**

This repository is build on MMDetection 2.13.0 which can be installed by running the following scripts. Please ensure that all dependencies have been satisfied before setting up the environment.

```
git clone https://github.com/chnu/DILA.git
cd DILA
pip install -r requirements.txt
python setup.py develop
```
# 🚀 **Training and Evaluation:**

Single GPU:

```
python ./tools/train.py ${CONFIG_FILE}
python ./tools/test.py ${CONFIG_FILE} ${WORK_DIR} --eval bbox
```
## **Main Results**

Pretrained Models: [Baidu Netdisk](https://pan.baidu.com/s/1gyIlj1UZGViYBjZzq8N6-w?pwd=dynv) Password: dynv 

