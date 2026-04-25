"""
LeNet-5 实现 - MNIST 手写数字识别
任务二：从零实现经典 LeNet-5 网络结构

代码来源：基于 LeNet-5 原始论文和深度学习教材实现
参考文献：
- LeCun, Y., et al. "Gradient-Based Learning Applied to Document Recognition." (1998)
- 《计算机视觉》第10篇 - 极简卷积神经网络 CNN 识别手写数字
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D
from tensorflow.keras.utils import to_categorical

# 加载数据
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
print("Preprocessing data...")
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建LeNet-5模型
print("Building LeNet-5 model...")
model = Sequential()
# C1: 卷积层
model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
# S2: 平均池化层
model.add(AveragePooling2D((2, 2)))
# C3: 卷积层
model.add(Conv2D(16, (5, 5), activation='relu'))
# S4: 平均池化层
model.add(AveragePooling2D((2, 2)))
# C5: 卷积层（相当于全连接层）
model.add(Conv2D(120, (5, 5), activation='relu'))
# 展平
model.add(Flatten())
# F6: 全连接层
model.add(Dense(84, activation='relu'))
# Output: 全连接层
model.add(Dense(10, activation='softmax'))

# 编译模型
print("Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print("Training model...")
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 评估模型
print("Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 保存模型
print("Saving model...")
model.save('lenet5_model.h5')

# 绘制训练曲线
print("Plotting training curves...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LeNet-5 Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LeNet-5 Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('lenet5_training_curves.png')
print("Training curves saved as lenet5_training_curves.png")

print("LeNet-5 training completed!")