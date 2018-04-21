import csv
import cv2
import numpy as np
lines=[]
with open("generated_data1/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

images=[]
measurements=[]
for line in lines:
    source_path=line[0]
    filename=source_path.split("/")[-1]
    if filename[0]=='c':
        current_path="generated_data1/IMG/"+filename

        image=cv2.imread(current_path)
        images.append(image)
        measurement=float(line[3])
        measurements.append(measurement)
    else:
        continue

augmented_images,augmented_measurements=[],[]

for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda
from keras.layers.convolutional import Convolution2D,Cropping2D


model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(24,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(1))




#model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True)
model.save('augmentedmodel.h5')
