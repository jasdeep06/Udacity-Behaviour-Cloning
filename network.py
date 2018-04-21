import csv
import cv2
import numpy as np
lines=[]
#reading log file and storing the lines from it.
with open("generated_data1/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)
csv_lines=[]
for line in lines:
    source_path=line[0]
    filename=source_path.split("/")[-1]
    #only files from centre camera
    if filename[0]=='c':
        csv_lines.append(line)


from sklearn.model_selection import train_test_split
train_csvlines,validation_csv_lines=train_test_split(csv_lines)
from  sklearn.utils import shuffle

#generator function to avoid holding preprocessed data in memory
def generator(data,batch_size=32):
    total_samples=len(data)
    while 1:
        for i in range(0,total_samples,batch_size):
            batch_samples=data[i:i+batch_size]

            images = []
            measurements = []

            for sample in batch_samples:
                fname = sample[0].split("/")[-1]
                current_path="generated_data1/IMG/"+fname
                image=cv2.imread(current_path)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                measurement=float(sample[3])
                images.append(image)
                measurements.append(measurement)

            X_train=np.array(images)
            y_train=np.array(measurements)
            yield shuffle(X_train,y_train)


train_generator=generator(train_csvlines)
validation_generator=generator(validation_csv_lines)

from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda,Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D
epoch=10

model=Sequential()
model.add(Lambda(lambda x:x/127.5-1.,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Convolution2D(24,3,3,activation="relu"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(1))




model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator,samples_per_epoch=len(train_csvlines),validation_data=validation_generator,nb_val_samples=len(validation_csv_lines),nb_epoch=epoch)

model.save('model.h5')

