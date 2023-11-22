
# 🌈 **Installation and Get Started**

## 🏆 **Dependencies:**

* Linux

* CUDA 11.3

* Python 3.7

* PyTorch 1.10.0

* TorchVision 0.11.0

* mmcv-full 1.5.0

* numpy 1.21.6

* GCC 5+
  
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

# 👑 Main Results

## 🥇 ***Result on AI-TOD***

Table 1. **Training Set:** AI-TOD trainval set, **Validation Set:** AI-TOD test set, 12 epochs. [AI-TOD dataset](https://github.com/jwwangchn/AI-TOD)

Method  | Backbone  | AP | AP50 | AP75 | APvt | APt | APs | APm | weight
 ---- | ----- | ------ | ------- | -------- | --------- | ---------- | ----------- | ------------ | --
 RetinaNet | R-50	| 8.7	| 22.3	| 4.8	| 2.4	| 8.9	| 12.2	| 16.0
 Faster R-CNN  | R-50 | 12.0 | 28.0 | 8.6 | 0.1 | 8.8 | 24.2 | 36.4
 Cascade R-CNN | R-50 | 13.8 | 30.8 | 10.5	| 0.0	| 10.6 | 25.5	| 26.6
 RetinaNet w/DILA | R-50	| 11.9	| 30.8	| 6.9	| 4.7	| 13.8	| 13.7	| 14.6 | [model](https://pan.baidu.com/s/1u2_Layg-RpkZ__sZ5CRpcg?pwd=DILA)
 Faster R-CNN W/DILA  | R-50 | 22.3	| 53.7	| 14.4	| 9.4	| 22.4	| 28.1	| 32.4 | [model](https://pan.baidu.com/s/1u2_Layg-RpkZ__sZ5CRpcg?pwd=DILA)
 Cascade R-CNN w/DILA | R-50 | 23.0	| 52.6	| 16.6	| 7.9	| 23.5	| 28.8	| 34.6 | [model](https://pan.baidu.com/s/1u2_Layg-RpkZ__sZ5CRpcg?pwd=DILA)

## 🏅 ***Result on AI-TOD***

Table 2. **Training Set:** SODA-D train set, **Validation Set:** AI-TOD test set, 12 epochs. [SODA-D dataset](https://shaunyuan22.github.io/SODA/)

Method  | Schedule  | AP | AP50 | AP75 | APvt | APt | APs | APm | weight
 ---- | ----- | ------ | ------- | -------- | --------- | ---------- | ----------- | ------------ | --
RetinaNet	| 1x | 28.2	| 57.6	| 23.7	| 11.9	| 25.2	| 34.1	| 44.2
FCOS	| 1x |23.9	| 49.5	| 19.9	| 6.9	| 19.4	| 30.9	| 40.9
RepPoints	| 1x | 28.0	| 55.6	| 24.7	| 10.1	| 23.8	| 35.1	| 45.3
ATSS	| 1x | 26.8	| 55.6	| 22.1	| 11.7	| 23.9	| 32.2	| 41.3
YOLOX | 70e| 26.7	| 53.4	| 23.0	| 13.6	| 25.1	| 30.9	| 30.4
CornerNet | 2x|	24.6 | 49.5	| 21.7	| 6.5	| 20.5	| 32.2	| 43.8
CenterNet	| 70e | 21.5	| 48.8	| 15.6	| 5.1	| 16.2	| 29.6	| 42.4
Deformable-DETR	| 50e | 19.2	| 44.8	| 13.7	| 6.3 | 15.4	| 24.9	| 34.2
Faster RCNN	| 1x | 28.9	| 59.4	| 24.1	| 13.8	| 25.7	| 34.5	| 43.0
Cascade RPN	 | 1x | 29.1	| 56.5	| 25.9	| 12.5	| 25.5	| 35.4	| 44.7
RFLA	| 1x | 29.7	| 60.2	| 25.2	| 13.2	| 26.9	| 35.4	| 44.6
DILA(Ours) | 1X |	30.4 | 	61.4	| 26.1	|14.0 |	27.0	| 36.5	| 47.0 | [model](https://pan.baidu.com/s/1m57Z_gQZabLsk8Gokd6eYw?pwd=DILA)

# 📧 **Contact**

If you have any problems about this repo, please be free to contact us at 12211060763@chnu.edu.cn 🙆‍♀️
