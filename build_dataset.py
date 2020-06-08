from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import glob
from PIL import Image

import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', './Raw', 'Data for dataset construction')
flags.DEFINE_string('output_path', './Dataset', 'Data output for dataset construction')
flags.DEFINE_integer('train_crop_size',513,'Image crop size')
flags.DEFINE_float('train_val_ratio',3/4,'Ratio for train and eval')


class DatasetBuild:

    def __init__(self, inputPath, outputPath, cropSize, trainValRatio):
        self._inputPath = inputPath
        self._outputPath = outputPath
        self._cropSize = cropSize
        self._images = glob.glob(self._inputPath+"/JPEGImages/*.jpg")
        self._segmentations = glob.glob(self._inputPath+"/Segmentation/*.png")
        self._trainValRatio = trainValRatio
        print(str(len(self._images)) + " images loaded.")
        print(str(len(self._segmentations)) + " segmentations loaded.")
        self.processImages()
        self.processImagesSet()
        self.generateTfRecords()
        print("Dataset build done.")

    def processImages(self):
        print("Processing images...")
        counter = 0
        for segmentation in self._segmentations:
            print("Image: " +str(counter))
            with open(segmentation,'rb') as file:
                img=Image.open(file)
                width, height = img.size
                ratio = self._cropSize/max(width,height)
                img = img.resize((int(width*ratio),int(height*ratio)))
                img = np.array(img)
                mask = np.zeros((img.shape[0],img.shape[1]))
                indicesFg = np.where(img[:,:,3] != 0)
                mask[indicesFg[0],indicesFg[1]] = 1
                mask = Image.fromarray(mask)
                mask = mask.convert("L")
                name = os.path.basename(file.name)[:-4]
                mask.save(self._outputPath+"/Segmentation/"+str(name)+".png")
                print("Mask saved")
                try:
                    with open(self._inputPath+"/JPEGImages/"+str(name)+".jpg",'rb') as file:
                        img = Image.open(file)
                        img = img.resize((int(width*ratio),int(height*ratio)))
                        img.save(self._outputPath+"/JPEGImages/"+str(name)+".jpg")
                except FileNotFoundError:
                    print("JPEGImage " + str(name)+" not found" )

            counter = counter +1
        print("Processing done")

    def processImagesSet(self):
        names=[]

        if(len(self._segmentations) >= len(self._images)):
            processQueue = self._images
        else:
            processQueue = self._segmentations
        for image in processQueue:
            with open(image,'rb') as file:
                print(file.name)
                name = os.path.basename(file.name)
                names.append(name[0:len(name)-4])
        train = open(self._outputPath+"/ImageSets/train.txt","w")
        val = open(self._outputPath+"/ImageSets/val.txt","w")
        trainval = open(self._outputPath+"/ImageSets/trainval.txt","w")

        maxTrain = int(self._trainValRatio*len(names))

        for i in range(0,maxTrain):
            train.write(names[i] + "\n")
            trainval.write(names[i] + "\n")

        for i in range(maxTrain,len(names)):
            val.write(names[i] + "\n")
            trainval.write(names[i] + "\n")

    def getEvalSplitLength(self):
        return int((1-self._trainValRatio)*len(self._images))

    def getTrainSplitLength(self):
        return int(self._trainValRatio*len(self._images))

    def generateTfRecords(self):
        datasetSplits = tf.gfile.Glob(os.path.join(self._outputPath+"/ImageSets", '*.txt'))
        for datasetSplit in datasetSplits:
            self.convertDataset(datasetSplit)
    
    def convertDataset(self,datasetSplit):
        
        dataset = os.path.basename(datasetSplit)[:-4]
        sys.stdout.write('Processing ' + dataset)
        filenames = [x.strip('\n') for x in open(datasetSplit, 'r')]
        numImages = len(filenames)
        numPerShard = int(math.ceil(numImages / 4))

        imageReader = build_data.ImageReader('jpeg', channels=3)
        labelReader = build_data.ImageReader('png', channels=1)

        for shardId in range(4):
            outputFilename = os.path.join(
                self._outputPath+"/tfrecord",
                '%s-%05d-of-%05d.tfrecord' % (dataset, shardId, 4))
            with tf.python_io.TFRecordWriter(outputFilename) as tfrecordWriter:
                startIdx = shardId * numPerShard
                endIdx = min((shardId + 1) * numPerShard, numImages)
                for i in range(startIdx, endIdx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        i + 1, len(filenames), shardId))
                    sys.stdout.flush()
                    imageFilename = os.path.join(self._outputPath+"/JPEGImages", filenames[i] + '.jpg')
                    imageData = tf.gfile.GFile(imageFilename, 'rb').read()
                    height, width = imageReader.read_image_dims(imageData)
                    segFilename = os.path.join(
                        self._outputPath+"/Segmentation",
                        filenames[i] + '.png')
                    segData = tf.gfile.GFile(segFilename, 'rb').read()
                    segHeight, seg_width = labelReader.read_image_dims(segData)
                    if height != segHeight or width != seg_width:
                        raise RuntimeError('Shape mismatched between image and label.')
                    example = build_data.image_seg_to_tfexample(
                        imageData, filenames[i], height, width, segData)
                    tfrecordWriter.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()

datasetBuilder = DatasetBuild(FLAGS.input_path,FLAGS.output_path,FLAGS.train_crop_size,FLAGS.train_val_ratio)



