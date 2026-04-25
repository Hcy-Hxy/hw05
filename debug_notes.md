# 调试记录

## 问题1：MNIST 数据下载失败

### 现象
```
 urllib.error.HTTPError: HTTP Error 403: Forbidden
```

### 原因分析
1. 网络连接问题或防火墙阻止
2. Keras 数据集下载源访问受限
3. 数据缓存目录权限问题

### 修改点
1. 检查网络连接是否正常
2. 设置代理（如果需要）：
   ```python
   import os
   os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
   os.environ['HTTPS_PROXY'] = 'https://proxy.example.com:8080'
   ```
3. 手动下载数据集到 `~/.keras/datasets/` 目录
4. 使用备用数据源或镜像站点

---

## 问题2：CUDA/GPU 不可用

### 现象
```
 tensorflow/core/common_runtime/gpu/gpu_init.cc:...
 Could not find CUDA capable GPU
```

### 原因分析
1. 未安装 NVIDIA 驱动
2. CUDA 版本与 TensorFlow 版本不匹配
3. cuDNN 未正确配置

### 修改点
1. 确认 NVIDIA 驱动已安装：运行 `nvidia-smi`
2. 检查 CUDA 版本兼容性：
   - TensorFlow 2.10 需要 CUDA 11.x
   - TensorFlow 2.12+ 需要 CUDA 11.x 或 12.x
3. 安装对应版本的 cuDNN
4. 设置环境变量：
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
5. 使用 CPU 模式运行（添加代码）：
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
   ```

---

## 问题3：中文路径导致文件保存失败

### 现象
```
UnicodeEncodeError: 'ascii' codec can't encode characters
```

### 原因分析
1. 工作目录或文件路径包含中文字符
2. Python 默认编码设置问题

### 修改点
1. 使用英文路径作为工作目录
2. 在代码开头添加编码声明：
   ```python
   # -*- coding: utf-8 -*-
   import sys
   reload(sys)
   sys.setdefaultencoding('utf-8')
   ```
3. 保存文件时使用英文文件名

---

## 问题4：依赖包版本冲突

### 现象
```
ImportError: cannot import name 'xxx' from 'tensorflow.keras'
```

### 原因分析
1. TensorFlow 与 Keras 版本不匹配
2. 安装了多个冲突的深度学习框架

### 修改点
1. 创建干净的虚拟环境：
   ```bash
   conda create -n tf-env python=3.8
   conda activate tf-env
   pip install tensorflow>=2.10.0
   ```
2. 统一使用 TensorFlow 内置的 Keras：
   ```python
   from tensorflow import keras
   ```
3. 卸载冲突的包：
   ```bash
   pip uninstall keras
   pip install tensorflow
   ```

---

## 问题5：内存不足 (OOM)

### 现象
```
ResourceExhaustedError: OOM when allocating tensor with shape...
```

### 原因分析
1. 批量大小 (batch_size) 设置过大
2. 系统内存不足
3. 模型参数过多

### 修改点
1. 减小批量大小：
   ```python
   batch_size=16  # 原32改为16
   ```
2. 使用梯度累积（如果需要大批量）
3. 关闭其他占用内存的程序
4. 增加系统虚拟内存
5. 使用更小的模型或减少网络层数

---

## 问题6：模型训练不收敛

### 现象
```
loss: nan  or  accuracy: 0.1 (几乎不变)
```

### 原因分析
1. 学习率设置过大
2. 数据未正确归一化
3. 标签编码错误

### 修改点
1. 减小学习率：
   ```python
   model.compile(optimizer=Adam(learning_rate=0.0001), ...)
   ```
2. 检查数据归一化：
   ```python
   x_train = x_train.astype('float32') / 255.0
   ```
3. 检查标签独热编码：
   ```python
   y_train = to_categorical(y_train, 10)
   ```

---

## 环境配置防坑指南

### 1. 虚拟环境配置
```bash
# 推荐使用 conda 创建独立环境
conda create -n tf-env python=3.8
conda activate tf-env

# 安装 TensorFlow（根据需求选择 CPU 或 GPU 版本）
pip install tensorflow  # CPU 版本
# pip install tensorflow-gpu  # GPU 版本
```

### 2. 验证安装
```python
import tensorflow as tf
print(tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### 3. 常用命令
```bash
# 查看已安装的包
pip list | grep -i tensorflow

# 更新 TensorFlow
pip install --upgrade tensorflow

# 卸载重装
pip uninstall tensorflow
pip install tensorflow>=2.10.0
```

### 4. Jupyter Notebook 配置
如果使用 Jupyter Notebook，需要在对应环境中安装 ipykernel：
```bash
python -m ipykernel install --user --name=tf-env --display-name="TensorFlow Environment"
```
