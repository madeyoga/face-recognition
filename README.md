# face-recognition
Train SVM &amp; KNN model for face recognition with the help of "The world's simplest facial recognition api for Python and the command line"

## Dependencies
- [opencv-python](https://pypi.org/project/opencv-python/)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [face_recognition](https://github.com/ageitgey/face_recognition)

## Dataset
- [LFW Face Database](http://vis-www.cs.umass.edu/lfw/)
- Used for training: 40 labels, 10 samples each.

## `preprocess & build.py` Output 
- SVM model gets 96% test accuracy
- KNN model gets 98% test accuracy

## Test on Random Image from Google
*Using SVM model*

![BillGates2](https://github.com/madeyoga/face-recognition/blob/master/output/output2.png)
