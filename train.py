"""Importarea pachetelor"""
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

"""Se citesc datele din tabelul csv
"""

df = pd.read_csv('/content/drive/MyDrive/LICENTA/age_gender.csv')
df = df.drop('img_name', axis=1)

#declararea variabilelor de tip list -> care vor prelua datele din tabel
images = []
age = []
gender = []

#popularea cu datele din tabel pt varsta si gen
for index, row in df.iterrows():
    age.append(np.array(row[0]))
    gender.append(np.array(row[1]))

y = df.drop('pixels', axis=1)

#pentru varsta clasificarea se imparte in 5 categorii
y["age"] = pd.cut(y["age"],bins=[0,3,18,45,64,116],labels=["0","1","2","3","4"])

#in variabila x vom avea doar coloana pixels, celelalte sunt eliminate
columns = ['age', 'gender']
x = df.drop(columns, axis=1)
img_height = int(np.sqrt(len(x['pixels'][0].split(" "))))
img_width = int(np.sqrt(len(x['pixels'][0].split(" "))))

#Reformatarrea datelor pt pixels
x = pd.Series(x['pixels'])
x = x.apply(lambda x:x.split(' '))
x = x.apply(lambda x:np.array(list(map(lambda z:np.int32(z), x))))
x = np.array(x)
x = np.stack(np.array(x), axis=0)
x = x.reshape(-1, 48, 48, 1)

age_matrix = np.array(y['age'])
gender_matrix = np.array(y['gender'])
age = to_categorical(age_matrix, num_classes = 5)
gender = to_categorical(y["gender"], num_classes = 2)


#Normalizarea datelor
x = x / 255
x = x / 255

datagen = ImageDataGenerator(
        featurewise_center = False,
    # set input mean to 0 over the dataset
       samplewise_center = False,
    # set each sample mean to 0
       featurewise_std_normalization = False,
    # divide inputs by std of the dataset
       samplewise_std_normalization=False,
    # divide each input by its std
       zca_whitening=False,
    # dimesion reduction
       rotation_range=5,
    # randomly rotate images in the range 5 degrees
       zoom_range = 0.1,
    # Randomly zoom image 10%
       width_shift_range=0.1,
    # randomly shift images horizontally 10%
       height_shift_range=0.1,
    # randomly shift images vertically 10%
       horizontal_flip=False,
    # randomly flip images
        vertical_flip=False  # randomly flip images
)

datagen.fit(x)

# Genul
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(x, gender, test_size=0.3, random_state=42)

# Varsta
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(x, age, test_size=0.3, random_state=42)

print(X_train_gender.shape, X_train_age.shape)


#constructia modelului si al layerelor
def my_model(num_classes, activation, loss):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation=activation))

    model.compile(optimizer='Adam',
                  loss=loss,
                  metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(patience=10,
                               min_delta=0.001,
                               restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                           patience = 2,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr = 0.00001)

epochs = 15 
# pentru rezultate mai bune, se mareste nr de epoci
batch_size = 64

#modelul pentru varsta
model_age = my_model(5,'softmax','categorical_crossentropy')
history_age = model_age.fit(X_train_age, y_train_age, batch_size=batch_size,
                              epochs = epochs, validation_data = (X_test_age,y_test_age),
                            steps_per_epoch= X_train_age.shape[0] // batch_size,
                            callbacks= [early_stopping,
                            learning_rate_reduction])

fig = px.line(
    history_age.history, y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'loss'},
    title='Training History')
#se afiseaza un grafic pentru a vedea diferentele de precizie pentru varsta
fig.show()
loss, acc = model_age.evaluate(X_test_age, y_test_age, verbose=0)
#se afiseaza valorile pentru pierdere si acuratete la varsta
print('Test loss: {}'.format(loss))
print('Test Accuracy: {}'.format(acc))


model_gender = my_model(2, "sigmoid", "binary_crossentropy")
history_gender = model_gender.fit(X_train_gender, y_train_gender,
                                 batch_size = batch_size,
                                 epochs = epochs,
                                 validation_data = (X_test_gender, y_test_gender),
                                 steps_per_epoch = X_train_gender.shape[0] // batch_size, callbacks=[early_stopping,learning_rate_reduction])
fx = px.line(
    history_gender.history, y=["loss", "val_loss"],
    labels = {'index':'epoch', 'value':'loss'},
    title = 'Training History')

#se afiseaza un grafic pentru a vedea diferentele de precizie pentru gen
fig.show()

#se afiseaza valorile pentru pierdere si acuratete la gen
loss, acc = model_gender.evaluate(X_test_gender, y_test_gender, verbose=0)
print("Test loss: {}".format(loss))
print("Test Accuracy: {}".format(acc))

"salvarea modelelor antrenate pe drive"
model_age.save("/content/drive/MyDrive/Models_Age/")
model_gender.save("/content/drive/MyDrive/Models_Gender/")