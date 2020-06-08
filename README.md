# DeepLab V3+ custom dataset implementation
Train your own custom dataset on DeepLab V3+ in an easy way

## Warnings
This implementation currently works only for the detection of 2 classes (for example: an object and the background).

## Make your dataset

1- Get a large amount of images of the object you want to segment (JPEG format).

2- Cut out the objects with the tool you want (photoshop, ...) and save it in png format. Please note: __the png cutout must have the same name and size as the original jpg image__.

3- Place the original images (jpg) in DATA/MyModel/Raw/JPEGImages and the cropped images (png) in DATA/MyModel/Raw/Segmentation.

## Train your dataset

1- Clone the [official DeepLab repo](https://github.com/tensorflow/models/tree/master/research/deeplab) and copy all files of this repo in models/research with replacing (you can delete all the folders except "slim" and "deeplab"). 

2- Edit model_training.sh :
- TRAIN_BATCH_SIZE : number of images per batch. An hugh value can take a lot of memory during training, depending of the TRAIN_CROP_SIZE. Ajust this value compared to your GPU memory.
- TRAIN_CROP_SIZE : preprocessing image size for training. The larger the size, the more precise the segmentation.
- TRAIN_VAL_RATIO : percentage of the number of training images compared to the number of evaluation images.
- NUM_OF_CLASSES : number of segmentation classes.
- MODEL_VARIANT : refer to the official DeepLab repo for more information.
- NUM_ITERATIONS : number of training steps.

For the other training parameters, look into deeplab/train.py file or the [official DeepLab repo](https://github.com/tensorflow/models/tree/master/research/deeplab). 

3- Download a pretrained model from [DeepLab model zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) and place all the .ckpt file into DATA/MyModel/Model/init_model

If you don't want to keep pretrained weights, set --last_layers_contain_logits_only=false.

4- Run model_training.sh. 

5- At the end, model checkpoint will be stored in DATA/MyModel/Model/train_log and the frozen inference graph (for production use) in DATA/MyModel/Model/frozen_graph. 

## Background removal

For automatic background removal, refer to this repo.





