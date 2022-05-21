# import librarii
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import self as self
from matplotlib.axes import Axes

# set_style metoda din pachetul seaborn care seteaza stilul de afisare al graficelor
sns.set_style('darkgrid')

#se ignora anumite warning-uri
import warnings
warnings.filterwarnings('ignore')

#citirea tabelului csv si afisarea informatiilor despre ce contine
df = pd.read_csv("age_gender.csv")
df = df.drop('img_name', axis=1)
df.info()

#declararea variabilelor de tip list care vor retine valorile din tabel
images = []
age = []
gender = []

#popularea cu datele din tabel pt varsta si gen
for index, row in df.iterrows():
    age.append(np.array(row[0]))
    gender.append(np.array(row[1]))

# renuntam pentru moment la coloana pixels, pentru a afisa cateva grafice despre celelalte coloane
y = df.drop('pixels', axis=1)

#variabile ajutatoare folosite pentru grafice
plot_titles = ["Repartizarea dupa varsta", "Repartizarea dupa gen"]
plot_xaxis = ['Varsta', 'Genul']
ct = 0

#parcurgere pentru afisarea repartizarii dupa grupe de varsta si gen
for i in y.columns:
    plt.figure(figsize=(15,7))
    g = sns.countplot(y[i], palette="icefire")
    plt.title(plot_titles[ct])
    plt.xlabel(plot_xaxis[ct])
    plt.ylabel("Numar")
    if i=="gender":
        plt.xticks([0,1],['masculin','feminin'])
    plt.show()
    ct += 1

#impartirea pe grupe de varsta si afisarea graficului
y["age"] = pd.cut(y["age"],bins=[0,3,18,45,64,116],labels=["0-3","3-18","18-45","45-64","64-116"])
plt.figure(figsize=(15,7))
g = sns.countplot(y["age"], palette="icefire")
plt.xlabel("Grupa de varsta")
plt.ylabel("Numar")
plt.title("Repartizarea pe categorii de varsta")
plt.show()


#acum incepe prelucrarea coloanei pixels.
columns = ['age', 'gender']
x = df.drop(columns, axis=1)
print(x.head())
#dimensiunile imaginilor sunt 48*48 pixels => 2304 valori pentru fiecare inregistrare
num_pixels = len(x['pixels'][0].split(" "))
img_height = int(np.sqrt(len(x['pixels'][0].split(" "))))
img_width = int(np.sqrt(len(x['pixels'][0].split(" "))))
print(num_pixels, img_height, img_width)

#Rearanjarea datelor pt coloana pixels pentru a putea fi folosite la invatare
X = pd.Series(x['pixels'])
X = X.apply(lambda x:x.split(' '))
X = X.apply(lambda x:np.array(list(map(lambda z:np.int(z), x))))
X = np.array(X)
X = np.stack(np.array(X), axis=0)
X = X.reshape(-1, 48, 48, 1)


#functie care afiseaza ca string genul, in loc de cifra
def get_gender(param):
    if param== 1:
        return "Feminin"
    return "Masculin"

#Vedere de ansamblu imagini
plt.figure(figsize=(16,16))
for i,a in zip(np.random.randint(0, 23705, 25), range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i])
    plt.xlabel(
    "Categorie varsta: "+str(y['age'].iloc[i])+
    "Genul:"+get_gender(y['gender'].iloc[i]))
plt.show()


#Vedere 1 singura imagine, cea de la index 1.
index = 1
image = np.fromstring(df['pixels'][index], sep = ' ', dtype = np.uint8).reshape((48,48))
plt.title("Image" + str(index))
plt.imshow(image, cmap ="gray")
plt.show()
print(image.shape)
