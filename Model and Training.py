import glob
from keras.models import Model
from keras.layers import Flatten,Dense,Activation
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

vgg=VGG16(input_shape=(256,256,3),include_top=False,weights='imagenet')

for layer in vgg.layers:
    layer.trainable=False

x=Flatten()(vgg.output)

prediction=Dense(2,activation='softmax')(x)

model=Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])


train_datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

train=train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/train',
                                        target_size=(256,256),
                                        class_mode='categorical',
                                        batch_size=32)


test=train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test',
                                        target_size=(256,256),
                                        class_mode='categorical',
                                        batch_size=32)

h=model.fit_generator(train,
                     validation_data=test,
                     epochs=10,
                     steps_per_epoch=len(train),
                     validation_steps=len(test),
                     verbose=1)

model.save('..kaggle/output/working/save.hdf5')


import matplotlib.pyplot as plt
plt.plot(h.history['loss'],label='train_loss')
plt.plot(h.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()
plt.plot(h.history['accuracy'],label='train_acc')
plt.plot(h.history['val_accuracy'],label='Validation acc')
plt.legend()
plt.show()

