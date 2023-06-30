### 环境准备

```
- 安装anaconda
- 构建独立环境：conda create -n dml python=3.9
- 安装pytorch: conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- 安装依赖包：pip install -r requirements.txt
- 安装tensorRT：
    - 下载（需注册登录）：https://developer.nvidia.com/nvidia-tensorrt-8x-download
    - 版本选择：TensorRT 8.6 GA for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 ZIP Package
    - 解压目录中执行：
                pip install python/tensorrt-8.6.1-cp39-none-win_amd64.whl
                pip install graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl
                pip install onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
                pip install uff/uff-0.6.9-py2.py3-none-any.whl
```




### 数据准备
```
custom_dataset
    |
    |--images           # 存放所有原图
    |  |--1.jpg
    |  |--2.jpg
    |  |--...

    |
    |--labels           # 存放所有标注图
    |  |--1.png
    |  |--2.png
    |  |--...
    |
    |--train.txt
    |--val.txt
    |--test.txt

train.txt
    |
    |--images/1.jpg labels/1.png
    |--images/2.jpg labels/2.png
    |--..
val.txt
    |
    |--images/3.jpg labels/3.png
    |--images/4.jpg labels/4.png
    |--..
test.txt
    |
    |--images/5.jpg 
    |--images/6.jpg 
    |--..
```

### 模型训练接口调用示例(默认加载config.yaml配置文件，可以在配置文件中修改参数，或者通过类方法set_设置参数)
```shell
python train.py
```
### 模型推理接口调用示例(默认加载infer.yaml,可通过set_model_path,set_input_size设置必要参数)
```shell
python test.py
```
