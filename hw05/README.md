# 极简 CNN 与 LeNet-5 实现

## 项目结构

```
hw05/
├── simple_cnn.py        # 任务一：极简 CNN 实现
├── lenet5.py            # 任务二：LeNet-5 实现
├── requirements.txt     # 依赖包列表
├── debug_notes.md       # 调试记录
├── report.md            # 实验报告
├── simple_cnn_model.h5  # 极简 CNN 模型（训练后生成）
├── lenet5_model.h5      # LeNet-5 模型（训练后生成）
├── training_curves.png  # 极简 CNN 训练曲线（训练后生成）
└── lenet5_training_curves.png  # LeNet-5 训练曲线（训练后生成）
```

## 环境要求

- **Python 版本**：3.8 或更高
- **操作系统**：Windows / Linux / macOS
- **硬件要求**：
  - CPU：任意多核处理器
  - 内存：至少 4GB RAM（推荐 8GB 或以上）
  - GPU（可选）：NVIDIA 显卡 + CUDA（用于加速训练）

## 环境设置

### 1. 创建虚拟环境（推荐）

```bash
# 使用 conda 创建虚拟环境
conda create -n tf-env python=3.8
conda activate tf-env

# 或使用 venv
python -m venv tf-env
# Windows: .\tf-env\Scripts\activate
# Linux/Mac: source tf-env/bin/activate
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

### 3. GPU 加速配置（可选）

如果使用 NVIDIA GPU 加速训练，需要安装 CUDA 和 cuDNN：

1. 安装 [CUDA Toolkit 11.x](https://developer.nvidia.com/cuda-downloads)
2. 安装 [cuDNN 8.x](https://developer.nvidia.com/cudnn)
3. 安装 tensorflow-gpu：

```bash
pip install tensorflow-gpu>=2.10.0
```

## 数据说明

本项目使用 MNIST 手写数字数据集，脚本会自动下载：
- 训练集：60,000 张图像
- 测试集：10,000 张图像
- 数据会在首次运行时自动从 Keras datasets 下载

## 一键运行

### 任务一：极简 CNN

```bash
python simple_cnn.py
```

### 任务二：LeNet-5

```bash
python lenet5.py
```

### 批量运行

```bash
# 运行极简 CNN
python simple_cnn.py

# 运行 LeNet-5
python lenet5.py
```

## 运行输出

每个脚本运行后会：
1. 自动下载 MNIST 数据集
2. 显示训练进度（每个 epoch 的损失和准确率）
3. 在测试集上评估模型
4. 输出测试准确率
5. 保存模型文件（`.h5`）
6. 保存训练曲线图（`.png`）

## 实验结果

| 模型 | 测试准确率 | 参数数量 | 训练时间（CPU） |
|------|------------|----------|-----------------|
| 极简 CNN | ~98.5% | ~10,000 | ~10-15 分钟 |
| LeNet-5 | ~99.2% | 61,722 | ~20-25 分钟 |

详细结果和分析请查看 `report.md` 文件。

## 调试说明

如果遇到问题，请参考 `debug_notes.md` 文件中的解决方案。

## 常见问题

### 1. MNIST 数据下载失败
- 检查网络连接
- 可以手动下载并放置到 `~/.keras/datasets/` 目录

### 2. 内存不足
- 减小批量大小（batch_size）
- 关闭其他占用内存的程序

### 3. GPU 不可用
- 确认 CUDA 和 cuDNN 已正确安装
- 或使用 CPU 模式训练（速度较慢）

## 参考资料

- [TensorFlow 官方文档](https://www.tensorflow.org/)
- [Keras 官方文档](https://keras.io/)
- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
