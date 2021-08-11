# Deploy yolov4 on Jetson Nano #

## 介绍 ##

**本仓库主要介绍将yolov4部署到Jetson Nano 上，并使用Deepstrem 和 TensorRT 加速网络的推理，主体内容如下**

Part1. 在主机端host machine上,使用Pytorch生成yolov4的ONNX模型.  

Part2. 目标设备上(此处是边缘端设备 Jetson Nano), 安装Deepstream. 

Part3. 目标设备上使用TensorRT 生成yolov4的 .engine 文件(Jetson Nano 自带cuda 10.2, TensorRT 7.1.3).

Part4. 使用Deepstream 运行yolov4的 .engine 文件.

Note: 如果想先对YOLOV4网络进行剪枝， 可以参考 https://github.com/chumingqian/Model_Compression_For_YOLOV4.


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

###  本仓库包含 YOLOV4网络使用 TensorRT 的两种方式 ###


1.单独使用TensorRT 运行yolov4 的推理引擎：

   - 参考[README.md](./tensorrt_yolov4/README.md) in `./tensorrt_yolov4` 
    
2.使用Deepstream 5.0 结合Tensorrt, 运行YOLOV4 的.engine 文件

   - 详情参考 [README.md](./deepstream_yolov4/README.md) in `./deepstream_yolov4` 



Reference:

0.https://github.com/Tianxiaomo/pytorch-YOLOv4

1.https://github.com/dusty-nv/jetson-inference

2.https://developer.nvidia.com/blog/real-time-redaction-app-nvidia-deepstream-part-2-deployment/

3.https://github.com/NVIDIA-AI-IOT/yolov4_deepstream

4.https://developer.nvidia.com/zh-cn/embedded/jetpack 

5.https://github.com/marcoslucianops/DeepStream-Yolo
