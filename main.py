import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Пути к папкам (оставить только apple, banana, avocado)
train_dir = "dataset/train/Fruit"
test_dir = "dataset/test/Fruit" 
valid_dir = "dataset/val/Fruit"

# Генераторы с нормализацией
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Генераторы из папок
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    classes=['Apple', 'Cucumber', 'Banana']  # фиксируем классы согласно папкам
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    classes=['Apple', 'Cucumber', 'Banana']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    classes=['Apple', 'Cucumber', 'Banana']
)

# Строим CNN модель
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 класса

# Компиляция
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение
history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=50
)

# Оценка на тестовых данных
score = model.evaluate(test_generator, steps=50)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Графики обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Сохраняем модель
model.save('fruit_classifier_apple_banana_avocado.h5')

# Проверим индексы классов
print("Class indices:", train_generator.class_indices)

# --- Пример предсказания на одном изображении ---
from tensorflow.keras.preprocessing import image

img_path = "dataset/test/Fruit/Apple/Golden-Delicious_001.jpg"  # путь к тестовой картинке
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
pred_class = np.argmax(prediction[0])
class_labels = ['Apple', 'Cucumber', 'Banana']

print("Prediction:", class_labels[pred_class], "| Probabilities:", prediction)