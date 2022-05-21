import cv2
from cv2 import imshow, imwrite
#Pachetul Pillow pentru redimensionarea imaginilor (mai putini pixeli)
#2304 -> 48pixels x 48pixels = 2304
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#Se initializeaza camera prin metoda VideoCapture din opencv
cam = cv2.VideoCapture(0)


#Se citeste rezultatul cu metoda read().
result, image = cam.read()

if result:
    #afiseaza
    imshow("test", image)


    #salveaza
    imwrite("test.png", image)

else:
    print("EROARE")

#deschid imaginea creata
image = cv2.imread('test.png')

#convertesc in grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#folosesc un model antrenat pentru a detecta fata
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#cod pentru conturul fetei si taierea fetei din poza
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    faces = image[y:y + h, x:x + w]
    cv2.imshow("face",faces)
    cv2.imwrite('face.jpg', faces)

# Afisare imagine cropp-uita
cv2.imwrite('detcted.jpg', image)
cv2.imshow('img', image)
cv2.waitKey()

#se modifica imaginea in alb negru si se afiseaza informatii despre ea
face_img = Image.open('face.jpg').convert("L")
new_image = face_img.resize((48,48))
print(new_image.mode)
print(new_image.size)

#citesc datele imaginii
image_data = new_image.getdata()
#aici extrag pixelii din imagine care sunt prelucrati pentru a putea face detectia cu ei
pixel_val = np.array(image_data)
test = []
test.append(pixel_val)
test = np.stack(np.array(test), axis=0)
test = test.reshape(-1,48,48,1)
test = test / 255
test = test / 255


#pixelii sunt convertiti din list in string
lin_img = pixel_val.flatten()
pixel_list = lin_img.tolist()
pixel_str_list = map(str, pixel_list)
img_str = ' '.join(pixel_str_list)

#se salveaza si imaginea crop-uita 48*48 pixels
new_image.save("final-face.jpg")

#se afiseaza imaginea nou creata
image = np.fromstring(img_str, sep = ' ', dtype = int).reshape((48,48))
plt.title("Imagine testare")
plt.imshow(image, cmap ="gray")
plt.show()


#deschid modelele deja antrenate pentru a face predictia pe imagine

import tensorflow as tf

#predictia de varsta
valori = ['0-3','3-18','18-45','45-64','64-99']
age_model = tf.keras.models.load_model("Models_Age/")
result = list(age_model.predict(test)[0])
max_value = max(result)
index = result.index(max_value)
print(valori[index])

#predictia de gen
gender_model = tf.keras.models.load_model("Models_Gender/")
result_gender = gender_model.predict(test)[0]
if result_gender[0] > result_gender[1]:
    print("Masculin")
else:
    print("Feminin")
