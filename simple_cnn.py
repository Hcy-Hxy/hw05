"""
极简 CNN 实现 - MNIST 手写数字识别
任务一：复现微信公众号文章中的极简 CNN 代码

代码来源：《计算机视觉》第10篇 - 极简卷积神经网络 CNN 识别手写数字
来源平台：微信公众号（计算机视觉）
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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

# 构建极简CNN模型
print("Building simple CNN model...")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# 编译模型
print("Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print("Training model...")
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 评估模型
print("Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 保存模型
print("Saving model...")
model.save('simple_cnn_model.h5')

# 绘制训练曲线
print("Plotting training curves...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved as training_curves.png")

print("Simple CNN training completed!")