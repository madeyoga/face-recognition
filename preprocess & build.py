import os
import numpy as np
import cv2
import face_recognition

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pickle

path = 'Dataset/'

labels = os.listdir(path)

print("label count: ", len(labels))

def preprocess(img):
    # top, right, bottom, left
    (t, r, b, l) = face_recognition.face_locations(img)[0]
    # crop image
    face_img = img[t:b, l:r]
    # resize 
    face_img = cv2.resize(face_img, (224, 224))
    # encode
    encode = face_recognition.face_encodings(face_img)[0]

    return encode

X = []
y = []
for i, label in enumerate(labels):
    img_filenames = os.listdir('{}{}/'.format(path, label))
    for filename in img_filenames:
        filepath = '{}{}/{}'.format(path, label, filename)
        img = cv2.imread(filepath)
        
        # Ignore if not found face in image
        try:
            encode = preprocess(img)
        except Exception as e:
            print(e, ":", label, filename)
            continue
        
        X.append(encode)
        y.append(i)

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

svc_model = svm.SVC(gamma='scale')
svc_model.fit(X_train, y_train)

## Train Accuracy
pred = svc_model.predict(X_train)
train_acc = accuracy_score(y_train, pred)
print("Training Accuracy: ", train_acc)

## Test Accuracy
pred = svc_model.predict(X_test)
test_acc = accuracy_score(y_test, pred)
print("Test Accuracy: ", test_acc)

model_name = 'svm-{}.model'.format(str(int(test_acc*100)))
pickle.dump(svc_model, open(model_name, 'wb'))


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

## Train Accuracy
pred = model.predict(X_train)
train_acc = accuracy_score(y_train, pred)
print("Training Accuracy: ", train_acc)

## Test Accuracy
pred = model.predict(X_test)
test_acc = accuracy_score(y_test, pred)
print("Test Accuracy: ", test_acc)

model_name = 'knn-{}.model'.format(str(int(test_acc*100)))
pickle.dump(svc_model, open(model_name, 'wb'))
