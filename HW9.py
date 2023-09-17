import predictions as predictions
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Завантаження та підготовка даних
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Перетворення та нормалізація даних
train_images = train_images / 255.0
test_images = test_images / 255.0

# Побудова моделі нейронної мережі
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Вивід інформації про модель
model.summary()

# Тренування моделі
history = model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=20, batch_size=64, validation_split=0.2)

# Прогнозування класів для тестового набору даних
predictions = model.predict(test_images)

# Відображення прикладів тестових зображень та їхніх прогнозованих класів
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    plt.title(f"П: {predicted_label}, С: {true_label}")
    plt.axis('off')

plt.show()

# Відображення графіків навчання
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точність (навчання)')
plt.plot(history.history['val_accuracy'], label='Точність (валідація)')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Втрати (навчання)')
plt.plot(history.history['val_loss'], label='Втрати (валідація)')
plt.xlabel('Епохи')
plt.ylabel('Втрати')
plt.legend()

plt.show()

# Оцінка точності на тестовому наборі даних
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print("\nТочність на тестовому наборі даних:", test_acc)
