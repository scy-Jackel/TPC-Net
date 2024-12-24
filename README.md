## 数据准备

以z轴切片形式保存到脑图tiff格式文件。<br><br>
人工标记得到的swc文件。<br>

## 运行步骤

### 生成终端点数据

运行generate_image/generate_image.py，可以得到以swc文件中记录的终端点为中心的50x50x50图像，以及在xy，xz，yz平面上的投影图。<br><br>
我们将图像分为三类，正确标记的终端点图像（正样本），过度追踪的终端点图像（负样本）以及未追踪完成的终端点图像（负样本），正常情况下正样本数据远多于负样本，所以为了数据均衡，需要分别运行generate_image/generate_image_over.py和generate_image/generate_image_less.py对两种负样本图像进行生成。

### 终端点数据分类

我们在identify_image文件夹中提供了使用多种神经网络模型对图像分类任务进行训练的方法，调用identify_image/jiance.py可以调用训练好的模型进行分类测试。

### 终端点数据修正

上述步骤可以检测出来过度追踪的终端点图像和未追踪完成的终端点图像，对于前者，我们可以使用change_image/xiujian-new-2.py进行swc标记修剪得到标记正确的终端点图像和swc文件，对于后者，我们可以使用change_image/tianbu.py进行swc标记填补得到标记正确的终端点图像和swc文件。
