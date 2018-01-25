#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import numpy.random as npr
import argparse
import os
import utils as ayg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allTest', type=int, default=1)
    parser.add_argument('--numAllTestImgs', type=int, default=1000)
    parser.add_argument('--numTrainImgs', type=int, default=200000)
    parser.add_argument('--allImagesPath', type=str, default='/path/to/all/dataset/folder/')
    parser.add_argument('--testImagesPath', type=str, default='/path/to/test/images/folder/')

    args = parser.parse_args()

    npr.seed(args.seed)

    save = os.path.expanduser(args.save)

    numTrain = args.numTrainImgs
    numTestImgs = args.numAllTestImgs

    with tf.Session(graph=tf.Graph()) as sess:
        export_dir = "{}/model/celebA".format(args.save)
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        loaded_graph = tf.get_default_graph()

        inputTensorXName = model.signature_def['predict_images'].inputs['x'].name
        inputTensorX = loaded_graph.get_tensor_by_name(inputTensorXName)
        inputTensory0Name = model.signature_def['predict_images'].inputs['y0'].name
        inputTensory0 = loaded_graph.get_tensor_by_name(inputTensory0Name)
        outputTensorName = model.signature_def['predict_images'].outputs['yn'].name
        outputTensor = loaded_graph.get_tensor_by_name(outputTensorName)

        I = npr.randint(numTrain, size=10000)
        _, newTrainY = ayg.createBatch(I, path=args.allImagesPath)
        meanY = np.mean(newTrainY, axis=0)

        if args.allTest == 1:
            I = npr.randint(low=numTrain+1, high=numTrain+numTestImgs, size=numTestImgs)
            valXBatch, valYBatch = ayg.createBatch(I, path=args.allImagesPath)
        else:
            valXBatch, valYBatch, numTestImgs = ayg.createBatchSpec(path=args.testImagesPath)
        y0 = np.expand_dims(meanY, axis=0).repeat(numTestImgs, axis=0)

        resImg = sess.run(outputTensor, {inputTensorX: valXBatch, inputTensory0: y0})
        cw = 10
        if numTestImgs < 10:
            cw = numTestImgs % 10
        ayg.saveImgs(valXBatch, resImg, valYBatch, "{}/results/res".format(args.save), colWidth=cw)


if __name__=='__main__':
    main()
