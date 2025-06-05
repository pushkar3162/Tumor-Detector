import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
model = load_model('BrainTumour10Epochs.h5')

image = cv2.imread('C:\\Users\\pushk\\Desktop\\dipproject1\\pred\\pred2.jpg')
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)
#print(img)
predictions = np.expand_dims(img, axis=0)

# Get the predicted class label
result = model.predict(predictions)
print(result)