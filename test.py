import os
import numpy as np
import cv2
import face_recognition
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

def preprocess(img):
    # top, right, bottom, left
    (t, r, b, l) = face_recognition.face_locations(img)[0]
    # crop image
    face_img = img[t:b, l:r]
    # resize 
    face_img = cv2.resize(face_img, (224, 224))
    # encode
    encode = face_recognition.face_encodings(face_img)[0]

    return encode, (t, r, b, l)

labels = os.listdir('Dataset/')

model = pickle.load(open('models/svm-96.model', 'rb'))
X_test = np.load('train_data/X_test.npy')
y_test = np.load('train_data/y_test.npy')

pred = model.predict(X_test)
print(accuracy_score(pred, y_test))

img = cv2.imread('random/BillGates.jpg')
encode, (t,r,b,l) = preprocess(img)

pred = model.predict([encode])
person = labels[pred[0]]
print("Is it {} ?".format(person))

cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
cv2.putText(img, person, (l-10, t), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
cv2.imshow("Prediction: " + person + "?", img)
