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
import os
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_modify import LogisticRegression
from mlp import HiddenLayer
from pic2pickle import getbuffer
def load_data(dataset):
    '''f=open('C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\faces\\info.txt')
    f2=open('C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\nonfaces\\bg.txt')
    dataorigin=[]
    i=0
    j=0
    os.chdir('C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\faces')
    for i in xrange(29*100):
        print '\b...loading %f'%((i+1)/(29*100.0))
        line=f.readline()
        if not line:
            break    
        line=line.replace('\n','',1)
        img=cv2.cvtColor(cv2.imread(line),7)
        assert img.any()
        dataorigin.append([img,1])
    os.chdir('C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\nonfaces')
    for i in xrange(935*100):
        print '\b...loading %f'%((i+1)/(935*100.0))
        print i
        line2=f2.readline()
        print line2
        if not line2:
            break
        line2=line2.replace('\n','',1)
        img=cv2.cvtColor(cv2.imread(line2),7)
        assert img.any()        
        dataorigin.append([img,0])        
    f.close()
    f2.close()'''
    gesture_pos_indexfile='C:\\deeplearning\\gesturesample\\index.txt'
    gesture_neg_indexfile='C:\\deeplearning\\neg_gesturesample\\index.txt'
    face_pos_indexfile='C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\faces\\info.txt'
    face_neg_indexfile='C:\\Users\\Acer\\Documents\\Visual Studio 2010\\Projects\\face\\mit\\nonfaces\\bg.txt'    
    face_pos_rawbuffer=getbuffer(face_pos_indexfile,'face_pos.pickle',2900,0)
    face_neg_rawbuffer=getbuffer(face_neg_indexfile,'face_neg.pickle',53500,0)
    gesture_pos_rawbuffer=getbuffer(gesture_pos_indexfile,'gesture_pos.pickle',929,1)
    gesture_neg_rawbuffer=getbuffer(gesture_neg_indexfile,'gesture_neg.pickle',8829,0)
    print '...file loaded'
    train_set=[[],[],[]]    
    for i in range(len(gesture_pos_rawbuffer[0])):
        train_set[0].append(gesture_pos_rawbuffer[0][i])
        train_set[1].append(gesture_pos_rawbuffer[1][i])
        train_set[2].append(0)
    for i in range(len(gesture_neg_rawbuffer[0])):
        train_set[0].append(gesture_neg_rawbuffer[0][i])
        train_set[1].append(gesture_neg_rawbuffer[1][i])
        train_set[2].append(0)
    for i in range(len(face_pos_rawbuffer[0])):
        train_set[0].append(face_pos_rawbuffer[0][i])
        train_set[1].append(face_pos_rawbuffer[1][i])
        train_set[2].append(0)
    for i in range(len(face_neg_rawbuffer[0])):
        train_set[0].append(face_neg_rawbuffer[0][i])
        train_set[1].append(face_neg_rawbuffer[1][i])
        train_set[2].append(0)

    for i in range(len(train_set[0])):
        train_set[0][i]=train_set[0][i]/256.0
    '''
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
    '''
    rval= train_set
    #rval = (train_set_x, train_set_y)
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


def evaluate_lenet5(learning_rate=0.00005, n_epochs=200,
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
    
    cv2.namedWindow('running_CNN',0)
    IMG=numpy.zeros((480,640))
    cv2.imshow('running_CNN',IMG)    
    
    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)
    
    train_set_x, train_set_y, skip = datasets
    '''print train_set_y
    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    #n_valid_batches /= batch_size
    #n_test_batches /= batch_size'''

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

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)# it was neg likelyhood  dont forget it
    #test=layer1.input
    # create a function to compute the mistakes that are made by the model
    '''test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )'''

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [x,y],
        outputs=[cost,layer3.errors(y)],
        updates=updates,
    )
    os.chdir('C:\\deeplearning')

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

    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    '''patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))'''
    full_error=1    
    #while full_error>0.01 : 
    if 1:                    
        avg_error=1
        j=0
        avg_error=1.0
        avg_cost=1.0
        max_error=1
        max_errorindex=0
        shouldskip=True
        #while True:
        #print skip
        while avg_error>0.0 or max_error>0.01:
            j=j+1
            lmax_error=max_error
            max_error=0.0
            lavg_cost=avg_cost
            lavg_error=avg_error
            if j==2:
                full_error=avg_error
            error_sum=0.0
            cost_sum=0.0
            if j%20==0:
                shouldskip=False
            else :
                shouldskip=True
            #shouldskip=True
            print 'round%d'%j
            for i in xrange((535+29)*100+929+8829):
                
                if shouldskip and skip[i]==1:
                    print 'skip'
                #elif skip[i]==2:
                #    print 'skip'
                else :
                    print 'round%d'%j
                    print 'going to process %d batch '%i
                    print 'lastime max error%f'%lmax_error
                    print max_errorindex
                    minibatch_avg_cost , the_error = train_model(train_set_x[i],[train_set_y[i]])
                    if minibatch_avg_cost==0 :
                        skip[i]=1
                    else :
                        skip[i]=0
                    #if minibatch_avg_cost==0:
                        #skip[i]=2
                    if max_error<minibatch_avg_cost:
                        max_error=minibatch_avg_cost
                        max_errorindex=i
                    error_sum=error_sum+the_error
                    avg_error=error_sum/(i+1)
                    cost_sum=cost_sum+minibatch_avg_cost
                    avg_cost=cost_sum/(i+1)
                    print 'error: %f'%the_error
                    print 'cost :%f'%minibatch_avg_cost
                    print 'done for %d'%(i+1)
                    print 'avg_cost%f'%avg_cost
                    print 'avg_error%f'%avg_error
                    print 'lasttime avg_cost%f'%lavg_cost
                    print 'lasttime avg_error%f'%lavg_error
                
            key=cv2.waitKey(1000)#exit and save
            if key==27:
                break
    cv2.destroyAllWindows()
    
    with open('g_W_layer0.pickle', 'wb') as f:
        pickle.dump(layer0.W.get_value(),f)
    with open('g_B_layer0.pickle', 'wb') as f:
        pickle.dump(layer0.b.get_value(),f)
        
    with open('g_W_layer1.pickle', 'wb') as f:
        pickle.dump(layer1.W.get_value(),f)
    with open('g_B_layer1.pickle', 'wb') as f:
        pickle.dump(layer1.b.get_value(),f)

    with open('g_W_layer2.pickle', 'wb') as f:
        pickle.dump(layer2.W.get_value(),f)
    with open('g_B_layer2.pickle', 'wb') as f:
        pickle.dump(layer2.b.get_value(),f) 
        
    with open('g_W_layer3.pickle', 'wb') as f:
        pickle.dump(layer3.W.get_value(),f)
    with open('g_B_layer3.pickle', 'wb') as f:
        pickle.dump(layer3.b.get_value(),f)
    '''W_layer0=open('W_layer0.txt','w')
    W_layer0_data=layer0.W.get_value()
    for i in range(len(W_layer0_data)):
        W_layer0.write(str(W_layer0_data[i]))
        W_layer0.write(' ')
    W_layer0.close()
    
    B_layer0=open('B_layer0.txt','w')
    B_layer0_data=layer0.b.get_value()
    for i in range(len(B_layer0_data)):
        B_layer0.write(str(B_layer0_data[i]))
        B_layer0.write(' ')
    B_layer0.close()
    
    
    W_layer1=open('W_layer1.txt','w')
    W_layer1_data=layer1.W.get_value()
    for i in range(len(W_layer1_data)):
        W_layer1.write(str(W_layer1_data[i]))
        W_layer1.write(' ')
    W_layer1.close()
    
    B_layer1=open('B_layer1.txt','w')
    B_layer1_data=layer1.b.get_value()
    for i in range(len(B_layer1_data)):
        B_layer1.write(str(B_layer1_data[i]))
        B_layer1.write(' ')
    B_layer1.close()
    
    W_layer2=open('W_layer2.txt','w')
    W_layer2_data=layer2.W.get_value()
    for i in range(len(W_layer2_data)):
        W_layer2.write(str(W_layer2_data[i]))
        W_layer2.write(' ')
    W_layer2.close()
    
    B_layer2=open('B_layer2.txt','w')
    B_layer2_data=layer2.b.get_value()
    for i in range(len(B_layer2_data)):
        B_layer2.write(str(B_layer2_data[i]))
        B_layer2.write(' ')
    B_layer2.close()
    
    W_layer3=open('W_layer3.txt','w')
    W_layer3_data=layer3.W.get_value()
    for i in range(len(W_layer3_data)):
        W_layer3.write(str(W_layer3_data[i]))
        W_layer3.write(' ')
    W_layer3.close()
    
    B_layer3=open('B_layer3.txt','w')
    B_layer3_data=layer3.b.get_value()
    for i in range(len(B_layer3_data)):
        B_layer3.write(str(B_layer3_data[i]))
        B_layer3.write(' ')
    B_layer3.close()'''
    
    end_time=time.time()
    print 'ran for %.1fs' %((end_time - start_time))
    print 'error %f'% avg_error
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
    