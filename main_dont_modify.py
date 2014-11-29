# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 19:14:32 2014
main
@author: Acer
"""
from theano.tensor import shared_randomstreams
import numpy
import time
import pickle
import cv2
import random
import threading
import theano
import theano.tensor as T
import math
import ctypes
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_modify import LogisticRegression
from mlp import HiddenLayer

fingerpool=[[],[],[],[]]
gesturepool=[[],[],[],[]]
img=0
started=False

'''
开启检测手掌和检测手指模块
'''
detectfinger=True
detectgesture=True
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class fingerdetect(threading.Thread):#手指检测器
    def __init__(self):
        threading.Thread.__init__(self)
        rng = numpy.random.RandomState(23455)
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.nkerns=[6, 16]
        print '... building finger the model'
        self.layer0_input = self.x.reshape((1, 1, 40, 40))
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(1, 1, 40, 40),
            filter_shape=(self.nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(1, self.nkerns[0], 18, 18),
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        self.layer2_input = self.layer1.output.flatten(2)
    
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=self.nkerns[1] * 7 * 7,
            n_out=500,
            activation=T.tanh
        )
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=2)
        print 'loading pickle'
    
        with open('f_W_layer0.pickle', 'rb') as f:
            self.layer0.W.set_value(pickle.load(f))
        with open('f_B_layer0.pickle', 'rb') as f:
            self.layer0.b.set_value(pickle.load(f))
        with open('f_W_layer1.pickle', 'rb') as f:
            self.layer1.W.set_value(pickle.load(f))
        with open('f_B_layer1.pickle', 'rb') as f:
            self.layer1.b.set_value(pickle.load(f))
        with open('f_W_layer2.pickle', 'rb') as f:
            self.layer2.W.set_value(pickle.load(f))
        with open('f_B_layer2.pickle', 'rb') as f:
            self.layer2.b.set_value(pickle.load(f))
        with open('f_W_layer3.pickle', 'rb') as f:
            self.layer3.W.set_value(pickle.load(f))
        with open('f_B_layer3.pickle', 'rb') as f:
            self.layer3.b.set_value(pickle.load(f))
    
        print 'finger pickles loaded'
        self.classfy=theano.function(inputs=[self.x],outputs=self.layer3.p_y_given_x)

        print 'finger function compiled'
    def run(self):
        global fingerpool
        global img
        global started
        while True:
            if started:
                self.localfingerpool=[[],[],[],[]]#清空
                self.a=img.copy()
                self.c=cv2.cvtColor(self.a,7)
                self.picsize=self.c.shape
                self.k=40       
                while self.k==40:
                    self.j=0
                    while self.j<self.picsize[1]/2:
                        self.i=0
                        while self.i<self.picsize[0]:
                            if (self.k+self.j<self.picsize[1]/2) and (self.k+self.i<self.picsize[0]):
                                self.pred=self.classfy(cv2.resize(self.c[self.i:self.i+self.k,self.j:self.j+self.k],(40,40))/256.0)[0][1]
                                if self.pred>=0.5:
                                    print 'got finger'
                                    self.localfingerpool[0].append(self.j)
                                    self.localfingerpool[1].append(self.i)
                                    self.localfingerpool[2].append(self.k)
                                    self.localfingerpool[3]=time.time()
                                    fingerpool=self.localfingerpool
                            self.i+=20
                        self.j+=20
                    self.k+=20

            else :
                time.sleep(0.01)

class gesturedetect(threading.Thread):#手指检测器
    def __init__(self):
        threading.Thread.__init__(self)
        rng = numpy.random.RandomState(23455)
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.nkerns=[6, 16]
        print '... building gesture the model'
        self.layer0_input = self.x.reshape((1, 1, 20, 20))
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(1, 1, 20, 20),
            filter_shape=(self.nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(1, self.nkerns[0], 8, 8),
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        self.layer2_input = self.layer1.output.flatten(2)
    
        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=self.nkerns[1] * 2 * 2,
            n_out=500,
            activation=T.tanh
        )
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=2)
        print 'loading pickle'
    
        with open('g_W_layer0.pickle', 'rb') as f:
            self.layer0.W.set_value(pickle.load(f))
        with open('g_B_layer0.pickle', 'rb') as f:
            self.layer0.b.set_value(pickle.load(f))
        with open('g_W_layer1.pickle', 'rb') as f:
            self.layer1.W.set_value(pickle.load(f))
        with open('g_B_layer1.pickle', 'rb') as f:
            self.layer1.b.set_value(pickle.load(f))
        with open('g_W_layer2.pickle', 'rb') as f:
            self.layer2.W.set_value(pickle.load(f))
        with open('g_B_layer2.pickle', 'rb') as f:
            self.layer2.b.set_value(pickle.load(f))
        with open('g_W_layer3.pickle', 'rb') as f:
            self.layer3.W.set_value(pickle.load(f))
        with open('g_B_layer3.pickle', 'rb') as f:
            self.layer3.b.set_value(pickle.load(f))
    
        print 'gesture pickles loaded'
        self.classfy=theano.function(inputs=[self.x],outputs=self.layer3.p_y_given_x)

        print 'gesture function compiled'
    def run(self):
        global gesturepool
        global img
        global started
        while True:
            self.looptime=time.time()
            if started:
                self.localgesturepool=[[],[],[],[]]#清空
                self.a=img.copy()
                self.c=cv2.cvtColor(self.a,7)
                self.picsize=self.c.shape
                self.k=100       
                while self.k<=160:
                    self.j=0
                    while self.j<self.picsize[1]/2:
                        self.i=0
                        while self.i<self.picsize[0]:
                            if (self.k+self.j<self.picsize[1]/2) and (self.k+self.i<self.picsize[0]):
                                self.pred=self.classfy(cv2.resize(self.c[self.i:self.i+self.k,self.j:self.j+self.k],(20,20))/256.0)[0][1]
                                if self.pred>=0.5:
                                    print 'got gesture'
                                    self.localgesturepool[0].append(self.j)
                                    self.localgesturepool[1].append(self.i)
                                    self.localgesturepool[2].append(self.k)
                                    self.localgesturepool[3]=time.time()
                                    gesturepool=self.localgesturepool
                                    break
                            self.i+=20
                        self.j+=10
                    self.k+=20

            else :
                time.sleep(0.01)
            print 'gesture spent time %f'%(time.time()-self.looptime)


def plotandmovemouse():#数据处理及绘图及移动鼠标
    global fingerpool
    global gesturepool
    global img
    if fingerpool != [[],[],[],[]] and (time.time()-fingerpool[3])<1:
        
        cv2.rectangle(img,(fingerpool[0][0],fingerpool[1][0]),(fingerpool[0][0]+fingerpool[2][0],fingerpool[1][0]+fingerpool[2][0]),cv2.cv.RGB(255,0,0))
    if gesturepool != [[],[],[],[]] and (time.time()-gesturepool[3])<1:
        ctypes.windll.user32.SetCursorPos(int((gesturepool[0][0]+gesturepool[2][0]*0.5)/320.0*1366.0), int((gesturepool[1][0]+gesturepool[2][0]*0.5)/480.0*768.0))#1366x768 
        cv2.rectangle(img,(gesturepool[0][0],gesturepool[1][0]),(gesturepool[0][0]+gesturepool[2][0],gesturepool[1][0]+gesturepool[2][0]),cv2.cv.RGB(0,255,0))

def mainloop():
    cv2.namedWindow('main',cv2.CV_WINDOW_AUTOSIZE)   
    capture=cv2.VideoCapture(0)
    #f=fingerdetect()
    g=gesturedetect()
    #f.setDaemon(True)
    g.setDaemon(True)
    #f.start()
    g.start()
    global started
    global img
    while True:
        a,img=capture.read()
        started=True
        plotandmovemouse()
        cv2.imshow('main',img)
        key=cv2.waitKey(1)
        if key==27:
            break
    capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    mainloop()