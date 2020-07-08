from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import glob
import numpy as np


model=load_model('model.hdf5')
paths=glob.glob('*.jpeg')  #The directory where the images are stored
print(len(paths))
for path in paths:
    img=image.load_img(path,target_size=(256,256))
    x=img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    classes=model.predict_(img_data)
    predict=model.predict_classes(img_data)
    if predict==0:
    	print('Everything is Normal...\n Accuracy:{np.round(classes[0][0]*100,2)}%')
    else:
    	print('Pneumonia Detected...\n Accuracy:{np.round(classes[0][1]*100},2)}%')
    	
