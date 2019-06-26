# Hotel Room Classifier

This repository has the tools necessary to create a dataset of hotel rooms from images found in trave-based social networks.

It uses the  [Places2 Database](http://places2.csail.mit.edu), and  [Hotels-50K](https://github.com/GWUvision/Hotels-50K) datasets.

The script to train direclty on the two-classes intermediary dataset is in the [keras_train_and_eval folder](/keras_train_and_eval), while the tranfer-learning scripts are in the [pytorch_transfer_and_eval folder](/pytorch_transfer_and_eval).

The [tools folder](/tools) contains the scripts used to adjust the original datasets.