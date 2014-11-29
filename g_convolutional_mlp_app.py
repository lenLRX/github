"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cv2
import time
import pickle
import numpy
import random

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_modify import LogisticRegression
from mlp import HiddenLayer



def load_data(dataset):
    f=open('img1.txt')
    f2=open('img2.txt')
    dataorigin=[]
    while 1:
        line=f.readline()
        if not line:
            break
        dataorigin.append([map(int,line.split()),1])
    while 1:
        line2=f2.readline()
        if not line2:
            break
        dataorigin.append([map(int,line2.split()),0])
    f.close()
    f2.close()
    print '...file loaded'
    train_set=[[],[]]
    for i in range(len(dataorigin)):
        train_set[0].append(dataorigin[i][0])
        train_set[1].append(dataorigin[i][1])
    for i in range(len(train_set[0])):
        for j in range(len(train_set[0][0])):
            train_set[0][i][j]=train_set[0][i][j]/256.0
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    #test_set_x, test_set_y = shared_dataset(test_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y)]
    return rval





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


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[6, 16], batch_size=1):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    start_time=time.time()    
    
    cv2.namedWindow('len',cv2.CV_WINDOW_AUTOSIZE)
    #cv2.namedWindow('detect',cv2.CV_WINDOW_AUTOSIZE)    
    capture=cv2.VideoCapture(0)   
    
    rng = numpy.random.RandomState(23455)


    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    #n_valid_batches /= batch_size
    #n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 20, 20))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 20, 20),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 8, 8),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 2 * 2,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    print 'loading pickle'

    with open('g_W_layer0.pickle', 'rb') as f:
        layer0.W.set_value(pickle.load(f))
    with open('g_B_layer0.pickle', 'rb') as f:
        layer0.b.set_value(pickle.load(f))
    with open('g_W_layer1.pickle', 'rb') as f:
        layer1.W.set_value(pickle.load(f))
    with open('g_B_layer1.pickle', 'rb') as f:
        layer1.b.set_value(pickle.load(f))
    with open('g_W_layer2.pickle', 'rb') as f:
        layer2.W.set_value(pickle.load(f))
    with open('g_B_layer2.pickle', 'rb') as f:
        layer2.b.set_value(pickle.load(f))
    with open('g_W_layer3.pickle', 'rb') as f:
        layer3.W.set_value(pickle.load(f))
    with open('g_B_layer3.pickle', 'rb') as f:
        layer3.b.set_value(pickle.load(f))

    print 'pickles loaded'
    classfy=theano.function(inputs=[x],outputs=layer3.p_y_given_x)
    #reshape=theano.function(inputs=[x],outputs=layer0_input)
    print 'function compiled'
    
    def detect(raw_pic,to_print,factor=2,minsize=30):
        picsize=raw_pic.shape
        print picsize
        count=0
        countmax=0
        imax=jmax=kmax=0
        count_time=0
        '''while (i<picsize[0]):
            j=0
            while (j<picsize[1]):
                k=minsize
                while (k+i<picsize[0]) and (k+j<picsize[1]):
                    print i,j,k+i,k+j
                    output=cv2.resize(raw_pic[i:i+k,j:j+k],(20,20))
                    #print cv2.imwrite('C:\\deeplearning\\imsave\\a%f.jpg'%random.random(),output)
                    draw(raw_pic,to_print,output,i,j,k,count)
                    k=k+20
                    count_time=count_time+1
                    time.sleep(0.2)
                j+=20
            i+=20
        print i
        print count_time'''
        k=100       
        while k<=200:
            j=0
            while j<=picsize[1]:
                i=0
                while i<=picsize[0]:
                    #print i,j,k
                    if (k+j<=picsize[1]) and (k+i<=picsize[0]):
                        output=cv2.resize(raw_pic[i:i+k,j:j+k],(20,20))
                        #print cv2.imwrite('C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\nonfaces\\a39%f.jpg'%random.random(),output)
                        count=draw(raw_pic,to_print,output,i,j,k,count)
                        if count>=countmax and count>0:
                            imax=i
                            jmax=j
                            kmax=k
                            countmax=count
                        count_time=count_time+1
                    i+=20
                j+=20
            k+=20
        print count_time,countmax
        if countmax>0:
            cv2.rectangle(to_print,(jmax,imax),(jmax+kmax,imax+kmax),0)
            '''print imax,jmax,kmax
            r=raw_pic[imax:imax+kmax,jmax:jmax+kmax]
            cv2.imshow('detect',r)
            cv2.waitKey(1)
            r=cv2.resize(r,(20,20))
            print cv2.imwrite('C:\\deeplearning\\neg_gesturesample\\a40%f.jpg'%random.random(),r)'''
    def draw(raw_pic,to_print,d_pic,i,j,k,count):
        pred=classfy(d_pic/256.0)[0][1]     
        if pred>0.01:
            print 'got%f'%pred
            print i,j,k
            r=raw_pic[i:i+k,j:j+k]
            cv2.imshow('detect',r)
            cv2.waitKey(1)
            r=cv2.resize(r,(20,20))
            print cv2.imwrite('C:\\deeplearning\\neg_gesturesample\\a41%f.jpg'%random.random(),r)
            #cv2.imshow('len',to_print
            #cv2.waitKey(1)
            #print 'detected%d'%count
            return count+1
        else :
            return 0
        '''cv2.rectangle(to_print,(j,i),(j+k,i+k),255)
        cv2.imshow('len',to_print)
        cv2.waitKey(1)'''
        
    while True:
        a,b=capture.read()
        c=cv2.cvtColor(b,7)
        #d=cv2.equalizeHist(c)
        e=c.copy()
        #cv2.imshow('len',e)
        #key=cv2.waitKey(1)
        a1=time.time()
        detect(c,e)
        a2=time.time()
        print a2-a1
        cv2.imshow('len',e)
        cv2.waitKey(1)
        #print 'stop!!!!'
        #time.sleep(1000)
        key=cv2.waitKey(1)
        if key==27:
            break
    capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
    