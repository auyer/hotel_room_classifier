# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)
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
parser.add_argument('--num_classes',default=2, type=int, help='num of class in the model')

parser.add_argument('--weights', '-w', metavar='weights', default='resnet18_best.pth.tar',
                    help='model architecture:')
args = parser.parse_args()
# th architecture to use
arch = args.arch
# arch = 'resnet18'

# load the pre-trained weights
model_file = args.weights
if args.arch.lower().startswith('wideresnet'):
    # a customized resnet model with last feature map size as 14x14 for better class activation mapping
    model  = wideresnet.resnet18(num_classes=args.num_classes)
else:
    model = models.__dict__[arch](num_classes=args.num_classes)
# model = models.__dict__[arch](num_classes=args.num_classes)
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
file_name = 'categories_places2hotels.txt'
if not os.access(file_name, os.W_OK):
    print("categories file missing")
    # synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    # os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][1:])
classes = tuple(classes)

TP = 0
FP = 0
TN = 0
FN = 0
TOTAL = 0

true_list = []
hip_list = []


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
            hip_list.append(h_x[0])
            probs, idx = h_x.sort(0, True)

            print('{} prediction on {}'.format(arch,img_name))
            # output the prediction
            for i in range(0, 2):
                print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            if file.startswith("nohotel"):
                true_list.append(0)
                if classes[idx[0]] == "nohotel":
                    TN+=1
                else:
                    FP += 1

            elif file.startswith("hotel"):
                true_list.append(1)
                if classes[idx[0]] == "hotel":
                    TP+=1
                else:
                    FN += 1
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
# roc_auc = auc((FP)/(FP+TN), (TP/(TP+FN)))

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_{}.png'.format(args.arch))