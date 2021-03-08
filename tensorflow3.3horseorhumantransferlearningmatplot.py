#loss: 0.0228 - accuracy: 0.9951 - val_loss: 6.3855e-04 - val_accuracy: 1.0000

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
weight= 'C:/Users/CVPR/Desktop/tensorflow/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_model= InceptionV3(input_shape=(150,150,3),
                              include_top= False,
                              weights= None)
pretrained_model.load_weights(weight)
for layer in pretrained_model.layers:
    layer.trainable= False
last_layer= pretrained_model.get_layer('mixed7')
last_output= last_layer.output
x= keras.layers.Flatten()(last_output)
x =keras.layers.Dense(1024,activation='relu')(x)
x= keras.layers.Dropout(0.2)(x)
x= keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(pretrained_model.input,x)
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen= ImageDataGenerator(rescale=1/255,rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validator_datagen= ImageDataGenerator(rescale=1/255)
train_generator=  train_datagen.flow_from_directory(
    'D:/datasets/horse-or-human',
    target_size=(150,150),
    batch_size= 20,
    class_mode= 'binary'
)
validator_generator=  validator_datagen.flow_from_directory(
           'D:/datasets/validation-horse-or-human',
            target_size=(150,150),
            batch_size= 20,
            class_mode= 'binary' )
history= model.fit(train_generator, epochs=10, validation_data=validator_generator, verbose=1)
import matplotlib.pyplot as plt
accuracy= history.history['accuracy']
valid_accuracy= history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs= range(len(accuracy))
plt.plot(epochs,accuracy, 'red', label= 'accuracy')
plt.plot(epochs,valid_accuracy,'g',label= 'valid_accuracy')
plt.title('training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
keras.layers.I
