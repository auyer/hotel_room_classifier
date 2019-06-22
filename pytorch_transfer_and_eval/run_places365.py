# PlacesCNN for scene classification
#
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import argparse

import wideresnet
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR', default=".",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--num_classes',default=365, type=int, help='num of class in the model')

args = parser.parse_args()
# th architecture to use
arch = args.arch
# arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

if args.arch.lower().startswith('wideresnet'):
    # a customized resnet model with last feature map size as 14x14 for better class activation mapping
    model  = wideresnet.resnet18(num_classes=365)
else:
    model = models.__dict__[arch](num_classes=args.num_classes)
# model.cuda()
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
# model = torch.nn.DataParallel(model)
model.cpu()
model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

correct_list = ["bedchamber", "bedroom", "berth", "childs_room", "dorm_room", "hospital_room", "hotel_room", "youth_hostel"]
# correct_list = ["hotel_room"]


TP = 0
FP = 0
TN = 0
FN = 0
true_list = []
hip_list = []
hotel_hip_list = []

TP5 = 0
FP5 = 0
TN5 = 0
FN5 = 0
hip_list5 = []

TP10 = 0
FP10 = 0
TN10 = 0
FN10 = 0
hip_list10 = []

TOTAL = 0
# load the test image
for file in os.listdir(args.data):
    if file.endswith(".jpg"):
        try:
            img_name = os.path.join(args.data, file)

            img = Image.open(img_name)
            input_img = V(centre_crop(img).unsqueeze(0))

            # forward pass
            logit = model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            positive_confidence = []
            for option in correct_list:
                positive_confidence.append(idx[classes.index(option)])
            hip_list.append(max(positive_confidence))
            hotel_hip_list.append(idx[classes.index("hotel_room")])
            print('{} prediction on {}'.format(arch,img_name))
            # output the prediction
            guesses5 = []
            guesses10 = []
            g5sum = 0
            for i in range(0, 5):
                # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                guesses5.append(classes[idx[i]])
                if classes[idx[i]] in correct_list:
                    g5sum += probs[i]
            if g5sum > 1: g5sum =1
            hip_list5.append(g5sum)
            g10sum = 0
            for i in range(0, 10):
                # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                guesses10.append(classes[idx[i]])
                if classes[idx[i]] in correct_list:
                    g10sum += probs[i]
            if g10sum > 1: g10sum =1
            hip_list10.append(g10sum)
            if file.startswith("nohotel"):
                true_list.append(0)
                if classes[idx[0]] not in correct_list:
                    TN+=1
                else:
                    FP += 1
                
                if len(set(guesses5) & set(correct_list)) > 0:
                    TN5+=1
                else:
                    FP5 += 1
                if len(set(guesses10) & set(correct_list)) > 0:
                    TN10+=1
                else:
                    FP10 += 1

            elif file.startswith("hotel"):
                true_list.append(1)
                if classes[idx[0]] in correct_list:
                    TP+=1
                else:
                    FN += 1

                if len(set(guesses5) & set(correct_list)) > 0:
                    TP5+=1
                else:
                    FN5 += 1
                if len(set(guesses10) & set(correct_list)) > 0:
                    TP10+=1
                else:
                    FN10 += 1
            TOTAL +=1
        except Exception as e:
            print(e)

print("\nAvg correct Hit:\n")
print("TP {} ".format(TP))
print("FP {} ".format(FP))
print("TN {} ".format(TN))
print("FN {} ".format(FN))

print("{}&{}&{}&{}&{}&{}".format(TP,FP,TN,FN, (float(TP)/float(TP+FN)) ,(float(FP)/float(FP+TN))))

print("\nAvg correct Hit TOP 5:\n")
print("TP5 {} ".format(TP5))
print("FP5 {} ".format(FP5))
print("TN5 {} ".format(TN5))
print("FN5 {} ".format(FN5))

print("{}&{}&{}&{}&{}&{}".format(TP5,FP5,TN5,FN5, (float(TP5)/float(TP5+FN5)) ,(float(FP5)/float(FP5+TN5))))

print("\nAvg correct Hit TOP 10:\n")
print("TP10 {} ".format(TP10))
print("FP10 {} ".format(FP10))
print("TN10 {} ".format(TN10))
print("FN10 {} ".format(FN10))

print("{}&{}&{}&{}&{}&{}".format(TP10,FP10,TN10,FN10, (float(TP10)/float(TP10+FN10)) ,(float(FP10)/float(FP10+TN10))))

print("Total successfull images: {} ".format(TOTAL))
print(TOTAL - (TP + FP + TN + FN ))

fpr, tpr, thresholds = roc_curve(true_list, hip_list)
roc_auc = auc(fpr, tpr)

fprh, tprh, thresholds = roc_curve(true_list, hotel_hip_list)
roc_auc_h = auc(fprh, tprh)

fpr5, tpr5, thresholds = roc_curve(true_list, hip_list5)
roc_auc5 = auc(fpr5, tpr5)

fpr10, tpr10, thresholds = roc_curve(true_list, hip_list10)
roc_auc10 = auc(fpr10, tpr10)

plt.figure()
plt.plot(fpr, tpr, color='magenta', lw=3, label='ROC curve MaxPred (area = %0.2f)' % roc_auc)
plt.plot(fprh, tprh, color='darkorange', lw=3, label='ROC curve hotel_room (area = %0.2f)' % roc_auc_h)
plt.plot(fpr5, tpr5, color='aqua', lw=2, label='ROC curve Top5 Sum (area = %0.2f)' % roc_auc5)
plt.plot(fpr10, tpr10, color='purple', lw=1, label='ROC curve Top10 Sum (area = %0.2f)' % roc_auc10)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc365_{}.png'.format(args.arch))