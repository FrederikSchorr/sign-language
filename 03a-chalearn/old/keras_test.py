from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
import numpy as np

model = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False)
model.summary()

img_path = "03a-chalearn/data/frame/train/c001/M_00071/frame-001.jpg"
pilFrame = image.load_img(img_path, target_size=(224, 224))
arFrame = image.img_to_array(pilFrame)

liX = []
liX.append(arFrame)
liX.append(arFrame)
liX.append(arFrame)

arX = np.array(liX)
print(arX.shape)
arX = preprocess_input(arX)
print(arX.shape)

arFeatures = model.predict(arX)
print(arFeatures.shape)