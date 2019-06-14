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

import argparse

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

model = models.__dict__[arch](num_classes=args.num_classes)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
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

correct_list = ["bedchamber", "bedroom", "berth", "childs_room", "dorm_room", "hospital_room", "hotel_room"]
top1 = 0
false1 = 0
top5 = 0
false5 = 0
top10 = 0
false10 =0
total_hotels = 0
total_nohotels = 0
# load the test image
for file in os.listdir(args.data):
    if file.endswith(".jpg"):
        if file.startswith("hotel"):
            try:
                img_name = os.path.join(args.data, file)

                img = Image.open(img_name)
                input_img = V(centre_crop(img).unsqueeze(0))

                # forward pass
                logit = model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)

                print('{} prediction on {}'.format(arch,img_name))
                # output the prediction
                guesses5 = []
                guesses10 = []
                for i in range(0, 5):
                    # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                    guesses5.append(classes[idx[i]])
                for i in range(0, 10):
                    # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                    guesses10.append(classes[idx[i]])

                if classes[idx[0]] in correct_list:
                    top1+=1
                if len(set(guesses5) & set(correct_list)) > 0:
                    top5+=1
                if len(set(guesses10) & set(correct_list)) > 0:
                    top10+=1
                total_hotels +=1
            except Exception as e:
                print(e)
        if file.startswith("nohotel"):
            try:
                img_name = os.path.join(args.data, file)

                img = Image.open(img_name)
                input_img = V(centre_crop(img).unsqueeze(0))

                # forward pass
                logit = model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)

                print('{} prediction on {}'.format(arch,img_name))
                # output the prediction
                guesses5 = []
                guesses10 = []
                for i in range(0, 5):
                    # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                    guesses5.append(classes[idx[i]])
                for i in range(0, 10):
                    # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                    guesses10.append(classes[idx[i]])

                if classes[idx[0]] in correct_list:
                    false1+=1
                if len(set(guesses5) & set(correct_list)) > 0:
                    false5+=1
                if len(set(guesses10) & set(correct_list)) > 0:
                    false10+=1
                total_nohotels +=1
            except Exception as e:
                print(e)
print("\nAvg correct Hit:\n")
print("top1 {} ".format(top1))
print("top5 {} ".format(top5))
print("top10 {} ".format(top10))
print("TRUE successfull images: {} ".format(total_hotels))


print("false1 {} ".format(false1))
print("false5 {} ".format(false5))
print("false10 {} ".format(false10))
print("FALSE successfull images: {} ".format(total_nohotels))
