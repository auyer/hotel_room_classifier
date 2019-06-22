from keras.models import model_from_json

import os
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


import argparse

parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR', default=".",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture Json file (default: resnet18)')
parser.add_argument('--num_classes',default=2, type=int, help='num of class in the model')

parser.add_argument('--weights', '-w', metavar='weights', default='resnet18_best.pth.tar',
                    help='model architecture:')
args = parser.parse_args()
# th architecture to use
print(args.arch)
# arch = 'resnet18'

json_file = open(args.arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(args.weights)
# loaded_model

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        args.data,
        color_mode="rgb",
        target_size=(224, 224),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

nb_samples = len(test_generator.filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)
guesses = np.argmax(predict, axis=1)
classes = ("hotel", "nohotel")

TP = 0
FP = 0
TN = 0
FN = 0
TOTAL = 0

true_list = []
hip_list = []

for i, pred in enumerate(predict):
    try:
        # import pdb; pdb.set_trace()
        hip_list.append(pred[0])
        print('{:.3f} -> {}'.format(pred[0], classes[0]))
        if test_generator.classes[i]==0 :
            true_list.append(1)
            if guesses[i] == 0 :
                TP+=1
            else:
                FN += 1
        if test_generator.classes[i]==1 :
            true_list.append(0)
            if guesses[i] == 1:
                TN+=1
            else:
                FP += 1
        TOTAL +=1
    except Exception as e:
        print(e)
            
print("\nAvg correct Hit:\n")
print("TP {} ".format(TP))
print("FP {} ".format(FP))
print("TN {} ".format(TN))
print("FN {} ".format(FN))
print("TRUE successfull images: {} ".format(TOTAL))
print(TOTAL - (TP + FP + TN + FN ))
print("{}&{}&{}&{}&{}&{} &{}&{}".format(TP,FP,TN,FN, (float(TP)/float(TP+FN)) ,(float(FP)/float(FP+TN)),float(TN+TP)/float(TOTAL), float(TP)/float(TP+FP)))

fpr, tpr, thresholds = roc_curve(true_list, hip_list)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('rocK_{}.png'.format(args.arch))