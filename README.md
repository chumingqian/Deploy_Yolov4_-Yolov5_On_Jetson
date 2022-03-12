#  This reporsity  introdce the Deployment on edge device, Jetson
#  I : Deploy yolov4 on Jetson Nano; 
#  II: Deploy yolov5 on Jetson TX2;



##  I : Deploy yolov4 on Jetson Nano;  ##

**这一部分主要介绍将yolov4部署到Jetson Nano 上，并使用Deepstrem 和 TensorRT 加速网络的推理，主体内容如下**

Part1. 在主机端host machine上,使用Pytorch生成yolov4的ONNX模型.  

Part2. 目标设备上(此处是边缘端设备 Jetson Nano), 安装Deepstream. 

Part3. 目标设备上使用TensorRT 生成yolov4的 .engine 文件(Jetson Nano 自带cuda 10.2, TensorRT 7.1.3).

Part4. 使用Deepstream 运行yolov4的 .engine 文件.

Note: 如果想先对YOLOV4网络进行剪枝， 可以参考[channel pruning for yolov4](https://github.com/chumingqian/Model_Compression_For_YOLOV4).


## part1. 在主机端host machine上,使用Pytorch生成yolov4的ONNX模型: ##
1.1 准备训练好的权重文件yolov4.weights 以及对应 .cfg 网络文件

1.2 安装 ONNX 模型生成工具，注意该仓库中的版本对应问题：

          `Pytorch 1.4.0 for TensorRT 7.0 and higher`
          `Pytorch 1.5.0 and 1.6.0 for TensorRT 7.1.2 and higher`
           git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
           cd pytorch-YOLOv4 
           pip install onnxruntime
           
1.3 将.weights 文件转换成 ONNX 模型，使用该脚本生成ONNX 文件，注意batchSize 的选择：

     `python demo_darknet2onnx.py <cfgFile> <weightFile> <imageFile> <batchSize>`
     
      usage demo: `python demo_darknet2onnx.py cfg/yolov4.cfg yolov4.weights 123.png 0`
      
      if batchsize 0: 代表生成一个动态的 onnx 文件， 类似  yolov4_dynamic.onnx 文件
      if batchsize n (n>0):  代表生成两个静态的 onnx 文件，比如 yolov4_1_static.onnx & yolov4_n_static.onnx,
                  yolov4_1_static.onnx 用于运行示例的。
                  yolov4_n_static.onnx  后续推理时需要达到对应的batchsize, 具体暂未使用。
       此处使用batchsize 0 
    
1.4 若得到的dynamic.onnx 文件，在后续生成.engine 文件时出错，此处的解决方式是：

      安装 pip3 install onnx-simplifier, 将.onnx 的模型图进行简化再操作。
      python  -m onnxsim   your_dynamic.onnx  --input-shape "1,3,416,416"




## part2 目标设备上(此处是边缘端设备 Jetson Nano), 安装Deepstream： ##

2.1 关于 DeepStream 和 TensorRT 的介绍可以参考 ./tensorRT.pdf, 两者的关系是 TensorRT 可以独立于 deepstream, TensorRT 可以在主机端或者边缘端安装，用于加速网络的推理. Deepstream 则是可以调用TensorRT生成的.egine 文件.


2.2 安装deepStream 的依赖库：

      `sudo apt install libssl1.0.0 libgstreamer1.0-0  gstreamer1.0-tools gstreamer1.0-plugins-good \
                        gstreamer1.0-plugins-bad  gstreamer1.0-plugins-ugly gstreamer1.0-libav \
                        libgstrtspserver-1.0-0    libjansson4=2.11-1`
                        
    
2.3 安装deepStream, 此处提供安装的下载包DeepStream5.1 版本 (链接: https://pan.baidu.com/s/1I_-Cpg8KjWYOjIs7fVcmZQ 提取码:s4ka)： 

       `sudo tar -xvf deepstream_sdk_v5.1.0_jetson.tbz2 -C /
        cd /opt/nvidia/deepstream/deepstream-5.1
        sudo ./install.sh
        sudo ldconfig`
    
2.4 测试DeepStream:

           `cd /opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_Yolo
           编辑文件prebuild.sh，注释掉除yolov3-tiny的语句,执行：./prebuild.sh
           下载yolov3-tiny.cfg和yolov3-tiny.weights
           执行命令：deepstream-app -c deepstream_app_config_yoloV3_tiny.txt`


## part3 目标设备上使用TensorRT 生成yolov4的 .engine 文件(Jetson Nano 自带cuda 10.2, TensorRT 7.1.3). ##
       
3.1 cd 到该路径下，使用trtexec 执行文件，将 .ONNX 文件生成 .engine 文件.

          `/usr/src/tensorrt/bin/trtexec --onnx=<onnx_file> \
           --minShapes=input:1x3x416x416 --optShapes=input:<batchsize>x3x416x416
           --maxShapes=input:<max_batchsize>x3x416x416 \
            --workspace=<size_in_megabytes> --saveEngine=<tensorRT_engine_file> --fp16 `
    
3.2  usage-demo: 

              `/usr/src/tensorrt/bin/trtexec --onnx=yolov4_1_3_416_416_dynamic.onnx \
              --minShapes=input:1x3x416x416 --optShapes=input:8x3x416x416 --maxShapes=input:8x3x416x416 \
              --workspace=4096 --saveEngine=yolov4-dynamic.engine --fp16`
     
3.3  参数说明： 

                          --onnx：生成的onnx文件  --minShapes：最小的batchsize * 通道数 * 输入尺寸x * 输入尺寸y
                          --optShapes：最佳输入维度，可取maxShapes一样  --maxShapes：最大输入维度
                          --workspace：默认为4096；--saveEngine：输出的engine名   --fp16：使用fp16精度；
   
   
3.4 注意到转换过程出现memory 不足的情况时，将--workspace 调整为512 ；
     


## part4.使用Deepstream 运行yolov4的 .engine 文件. ##

4.1 编写解析 yolov4  算子的层插件， 此处提供两个版本，推荐Nvidia官方的插件。

           https://github.com/marcoslucianops/DeepStream-Yolo
           git clone  https://github.com/NVIDIA-AI-IOT/yolov4_deepstream
           cd yolov4_deepstream/


4.2  拷贝  deepstream_yolov4/ 该文件夹，到目标设备上的这个位置：/opt/nvidia/deepstream/deepstream-5.1/sources/，
              确保路径正确；


4.3 修改配置文件, 编译生成 yolov4 的动态链接库：
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`:修改其中的 NUM_CLASSES_YOLO 为对应的类别数；

- `cd 到 deepstream_yolov4/ , 执行 `export CUDA_VER=10.2  make -C nvdsinfer_custom_impl_Yolo` 
    编译生成一个动态链接库 libnvdsinfer_custom_impl_Yolo.so `
    
- `deepstream_app_config_yoloV4.txt`:model-engine-file = your .engine file, labelfile-path = labels.txt, source、sink :for you need;

- `config_infer_primary_yoloV4.txt`: model-engine-file=<onnx_engine_file>

- `labels.txt`: 标签名字，一行一个标签名，顺序要按照训练时的标签顺序;
                                    

4.4 测试视频文件推理, cd  /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_yolov4/:

```sh
  $ deepstream-app -c deepstream_app_config_yoloV4.txt
```


4.5 USB摄像头视频测试:

           摄像头简单检测指令：ls /dev/video*
           安装v4l-utils工具：sudo apt install v4l-utils
           
           检测摄像头比较完整信息的指令：v4l2-ctl --list-devices
           
           摄像头更细致规格的查看指令：
                           v4l2-ctl --device=/dev/video0 --list-formats-ext
                          v4l2-ctl --device=/dev/video1 --list-formats-ext
             YOLOv4 USB摄像头视频测试命令：
             deepstream-app -c source1_usb_dec_infer_yolov4.txt
             
             
4.6 CSI摄像头视频测试:

               deepstream-app -c source1_csi_dec_infer_yolov4.txt
-------

4.7 对比原始的YOLOV4 和各个剪枝后网络的推理一张图片的速度：


|<center> Model Size </center> |<center>  inference speed</center> |
| --- | --- |
|BaseModel  256M    |414 ms    |
|  98 M             |311 ms   |
|90 M               |305 ms    |
|84 M               |303 ms    |
|77 M               |291 ms    |
|68M                |274 ms    |

Note: 以上模型的体积大小是剪枝后的体积大小， 并非是实际经过量化后体积大小， 实际量化后的大小，在开发板上可查看。

###   使用 TensorRT 运行YOLOV4网络的两种方式 ###


1.单独使用TensorRT 运行yolov4 的推理引擎：

   - 参考[README.md](./tensorrt_yolov4/README.md) in `./tensorrt_yolov4` 
    
2.使用Deepstream 5.0 结合Tensorrt, 运行YOLOV4 的.engine 文件

   - 详情参考 [README.md](./deepstream_yolov4/README.md) in `./deepstream_yolov4` 



------------------------    II : Deploy Yolov5 on Jetson TX2 ;  ----------------------


##  1. 步骤一 环境准备
https://developer.nvidia.com/jetpack-sdk-46

Jetpack4.6  环境:  ubuntu 18.04; 
TensorRT 8.0.1;   重要！！ 用于生成 .engine 文件；
includes CUDA 10.2;  includes cuDNN 8.2.1;

The next version of NVIDIA DeepStream SDK 6.0 will support JetPack 4.6;

检查 1. cuda  对应版本是否 安装， 2. 安装 Tensor RT; 3.  安装opencv；

dpkg -l | grep cuda
dpkg -l | grep nvinfer  
dpkg -l | grep opencv



##  2. 项目文件准备

2.1 下载yolov5 项目到 边缘设备端
git clone -b v4.0 https://github.com/ultralytics/yolov5.git
cd yolov5

2.2  下载 DeepstreamYolov
https://github.com/marcoslucianops/DeepStream-Yolo 

2.3  下载yolov5s.pt 文件；  此处以yolov5s 文件  对应为权重为例子；



##  3.准备 gen_wts_yolov5.py 文件， 用于生成.wts 文件；
3.1 将DeepStream-Yolo/utils 中的 gen_wts_yoloV5.py  文件拷贝到 /yolov5  文件下面；


3.2 生成 .cfg 网络文件 和 .wts 格式的权重文件；


python3 gen_wts_yoloV5.py -w yolov5s.pt

注意此过程， 是在 yolov5  项目中运行的， 如果出现  cv2. (set_NUM) 对应的问题， 这里本人 是将那一行 注释掉；
此时会产生 对应的yolov5s.cfg  和 yolov5s.pt 两个文件；


 

##  4.  移动 DeepStream-Yolo 文件;
4.1 将 DeepStream-Yolo 移动到 目标设备上的这个位置：/opt/nvidia/deepstream/deepstream-6/sources/， 确保路径正确；

4.2 将 对应的yolov5s.cfg  和 yolov5s.pt 两个文件 移动到 DeepStream-Yolo 文件中；


##  5.   编译文件
 5.1   在 DeepStream-Yolo  文件下， 打开终端，  编译文件

CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo；



##  6.  修改配置文件；
6.1  打开  config_infer_primary_yoloV5.txt ，  修改 对应的 cfg ，  .wts 文件
[property]
...
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
# CFG
custom-network-config=yolov5n.cfg
# WTS
model-file=yolov5n.wts
# Generated TensorRT model (will be created if it doesn't exist)
model-engine-file=model_b1_gpu0_fp32.engine
# Model labels file
labelfile-path=labels.txt
# Batch size
batch-size=1
# 0=FP32, 1=INT8, 2=FP16 mode
network-mode=0
# Number of classes in label file
num-detected-classes=80
...
[class-attrs-all]
# IOU threshold
nms-iou-threshold=0.6
# Socre threshold
pre-cluster-threshold=0.25



6.2  修改 deepstream_app_config.txt；
...
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV5.txt



## 7. run  运行 
deepstream-app -c deepstream_app_config.txt；
注意， 此步骤中 deepstream_app_config.txt  运行时， 调用了config_infer_primary_yoloV5.txt，  
该文件 config_infer_primary_yoloV5.txt 中 如果没有实现 生成.engine 文件， 运行过程 中 会自动生成.engine 文件；

而生成. engine 文件过程中， 会调用 依赖tensor RT;


##  8. 运行 摄像头文件；

 将 deepstream_app_config.txt 中的  [source0] 改成如下部分；

[source0]
enable=1
type=1
camera-width=640
camera-height=480
camera-fps-n=30
camera-fps-d=1
camera-v4l2-dev-node=0





------------------------  End  ----------------------






Reference:

0.https://github.com/Tianxiaomo/pytorch-YOLOv4

1.https://github.com/dusty-nv/jetson-inference

2.https://developer.nvidia.com/blog/real-time-redaction-app-nvidia-deepstream-part-2-deployment/

3.https://github.com/NVIDIA-AI-IOT/yolov4_deepstream

4.https://developer.nvidia.com/zh-cn/embedded/jetpack 

5.https://github.com/marcoslucianops/DeepStream-Yolo
