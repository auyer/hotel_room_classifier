import tensorflow as tf
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import backend as K
import functools
from keras.optimizers import SGD
from keras.callbacks import TensorBoard , ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
# K.set_image_dim_ordering('th')
from classification_models.resnet import ResNet50 #, preprocess_input


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.intra_op_parallelism_threads = 28
config.inter_op_parallelism_threads = 28
tf.ConfigProto().gpu_options. allocator_type = 'BFC'
sess = tf.Session(config=config)
K.set_session(sess)

num_of_classes = 2
batch_size = 64
epochs = 10
# train_images = 183488
# val_images = 900
train_images = 1979722
val_images = 73200
train_steps_per_epoch = train_images / batch_size
val_steps_per_epoch = val_images / batch_size

# create the base pre-trained model
# model = InceptionResNetV2(weights=None, include_top=True, classes=num_of_classes)
# model = VGG16(weights=None, include_top=True, classes=num_of_classes)
model = ResNet50((224, 224, 3), weights=None, classes=num_of_classes)
# model = ResNet50(weights=None, include_top=True, classes=num_of_classes)
print(model.summary())

# compile the model (should be done *after* setting layers to non-trainable)
sgd = SGD(lr=0.1, momentum=0.9, decay=1e-4)

# top5_acc = functools.partial(top_k_categorical_accuracy, k=5)

# top5_acc.__name__ = 'top5_acc'

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

print(model.summary())
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
   #  shear_range = 0.2, # random application of shearing
    # zoom_range = 0.2,
    horizontal_flip = True
    )

train_generator = train_datagen.flow_from_directory(
    directory=r"/home/auyer/raid/bkp.places2hotels_unified/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    directory=r"/home/auyer/raid/bkp.places2hotels_unified/val/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=42
)
len(validation_generator.filenames)

# # train the model on the new data for a few epochs
# logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = TensorBoard(write_graph=True,histogram_freq=1 ,batch_size=batch_size, write_images=True)

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit_generator(train_generator,
                    use_multiprocessing=True,
                    workers=28,
                    steps_per_epoch = train_steps_per_epoch,
                    epochs = epochs,
                    validation_data = validation_generator,
                    validation_steps = val_steps_per_epoch,
                    callbacks=[checkpoint , EarlyStopping(patience=3, restore_best_weights=True)])
