# CV Library
本工程基于 :link: [OpenMMLab](https://openmmlab.com/codebase)开发，用于个人学习、记录。

:link: [MMEngine 官方文档](https://mmengine.readthedocs.io/zh-cn/latest/)

[MMCV 官方文档](https://mmcv.readthedocs.io/zh-cn/latest/)

[MMDetection 官方文档](https://mmdetection.readthedocs.io/zh-cn/latest/index.html)

[MMDetection3D 官方文档](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html)

******

## Algorithm Library

**2D目标检测**

|   Model   | Input Size | Epochs (best) | F1 | mAP | mAP@.5 | mAP@.75 | mAP@S | mAP@M | mAP@L | Train Time |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | --------- | :-------: | :-------: |
| CenterNet | 288 x 512 | 12 * 2, (12) | 0.12 | 0.104 | 0.218 | 0.086 | 0.035 | 0.125 | 0.219 | 12h, 03m, 16s |
| TTFNet | 288 x 512  | 12 * 2, (12) | 0.131 | 0.115 | 0.226 | 0.099 | 0.039 | 0.143 | 0.243 | 12h, 57m, 52s |
| TTFNet_Plus| 288 x 512  | 12 * 2, (12) | 0.145 | 0.125 | 0.252 | 0.105 | 0.038 | 0.156 | 0.265 | 20h, 27m, 01s |
| FCOS | 288 x 512 | 12 * 2, (9) | 0.09 | 0.076 | 0.16 | 0.068 |0.016| 0.109 |0.186|5h, 53m, 59s|


## Install

**步骤0. 创建并激活一个 conda 环境**

~~~bash
conda create --name alchemy python=3.8 -y
conda activate alchemy
~~~

**步骤1. 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch**

在 GPU 平台上：

```bash
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

**步骤2. 安装MIM**

~~~bash
pip install -U openmim
~~~

**步骤3. 使用mim安装MMEngine**

~~~Bash
mim install mmengine
~~~

**步骤4. 使用mim安装MMCV**

~~~Bash
mim install 'mmcv>=2.0.0rc4'
~~~

**步骤5. 使用mim将 mmdet 作为依赖或第三方 Python 包安装**

~~~Bash
mim install 'mmdet>=3.0.0'
~~~


**步骤6. 安装相关依赖包**

~~~bash
pip install future tensorboard
~~~

**步骤7. 将alchemy作为Python包以开发模式安装**

~~~bash
python setup.py develop
~~~
