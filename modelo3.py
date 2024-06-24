import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Directorio donde están las imágenes
data_dir = 'bts_images'

# Filtrar archivos que no sean directorios y excluir .DS_Store
class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith('.')]
print(f"Classes found: {class_names}, Count: {len(class_names)}")

# Hiperparámetros
img_width, img_height = 224, 224
batch_size = 32
epochs = 50
num_classes = len(class_names)  # Número de clases según las carpetas encontradas

# Generador de datos de entrenamiento con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,  # Aumentamos el rango de rotación
    width_shift_range=0.2,  # Aumentamos el rango de desplazamiento horizontal
    height_shift_range=0.2,  # Aumentamos el rango de desplazamiento vertical
    horizontal_flip=True,
    validation_split=0.2
)

# Generadores de datos para entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=class_names  # Especificamos las clases encontradas
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=class_names  # Especificamos las clases encontradas
)

# Definición del modelo
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint(
    'models/bts_model3.keras', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Guardar el modelo final
model.save('models/bts_model_final3.keras')
