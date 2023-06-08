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
python train_demo.py
```
### 模型推理接口调用示例(默认加载infer.yaml,可通过set_model_path,set_input_size设置必要参数)
```shell
python test_demo.py
```
