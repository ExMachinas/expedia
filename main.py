# Main Application
# Author: dujung
#
import six.moves.cPickle as pickle
import numpy as np                              # numpy
from DataReader import DataFactory
from DataReader import DTR                      # DateTime Reverse Function.
from DataReader import MatrixStack

def main(useKeras = True):
    print("main(Keras=%s)...."%("Yes" if useKeras else "No"))

    # prepare dataset to run
    if useKeras:
        dataset = prepare_dataset_for_keras()
    else:
        dataset = prepare_dataset()

    # run sgd-optimization with given dataset.
    if dataset is not None:
        #sgd_optimization_mnist(dataset=dataset)
        #test_mlp(dataset=dataset)
        #test_keras(dataset)
        test_keras_CNN(dataset)
    else:
        raise("dataset is not loaded!")

    print("finished....")

# load data-files from cached file if possible.
def do_load_file(reload=False):
    fact = DataFactory.load(reload)
    print(fact)

    dest = fact.get('destination')
    if dest is not None:
        print("---------------------------- : destination")
        print(dest.header())
        print('> count=%d'%(dest.count()))

    train = fact.get('train')
    if train is not None:
        print("---------------------------- : train")
        print(train.header())
        print('> train=%d'%(dest.count()))

    test = fact.get('test')
    if test is not None:
        print("---------------------------- : test")
        print(test.header())
        print('> test=%d'%(dest.count()))

    #! step1. build-up lookuptable for destination.
    ret = dest.build_map()
    return ret

# reset load to release memory.
def do_reset_load():
    fact = DataFactory.load()
    fact.reset_all()
    return True

# transform test-date to temporal matrix-stack array.
def prepare_transform_train():
    from DataReader import TransTrain00
    # load transformer.
    mstack = TransTrain00()
    if not mstack.load_from_file():
        do_load_file()
        mstack.transform()
        #mstack.transform(force=True)
        mstack.test()
        do_reset_load()

    return mstack

# safe-loading dataset file.
def prepare_dataset(filename = "data/dataset-00.dat"):
    # import os.path
    # if os.path.isfile(filename):
    #     return None

    # build dataset from mstack.

    print("=====================================")
    print("Start: Data Conversion to Matrix file")
    print("=====================================")
    #do_load_file()             # load data.
    #do_load_file(True)          # force to reload
    #do_transform_train()        # transform data.
    mstack = prepare_transform_train()
    print("mstack.count = "+str(mstack.count()))

    # prepare train/validation set (90%, 10%)
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    count = mstack.count()
    for i in range(count):
        (x, y) = (mstack._matrix_list[i], mstack._matrix_list_y[i])
        if i % 10 != 2:
            train_x.append(x)
            train_y.append(y)
            #train_y += y
        else:
            valid_x.append(x)
            valid_y.append(y)
            #valid_y += y

    train_x = np.vstack(train_x)
    train_y = np.concatenate(train_y)
    valid_x = np.vstack(valid_x)
    valid_y = np.concatenate(valid_y)

    print("train_x.count = %d, train_y.count = %d"%(len(train_x), len(train_y)))
    print("valid_x.count = %d, valid_y.count = %d"%(len(valid_x), len(valid_y)))
    print('-------------- test data')
    prnt_idx = 5
    print('x[%d]='%(prnt_idx), train_x[prnt_idx])
    print('y[%d]='%(prnt_idx), train_y[prnt_idx])

    # # ok now save this data into file..
    # f = open(filename, 'wb')
    # try:
    #     pickle.dump((train_x, train_y, valid_x, valid_y), f, protocol=pickle.HIGHEST_PROTOCOL)
    # except:
    #     print('%s: failed to save file'%(filename))
    #     try:
    #         import os
    #         os.remove(filename)
    #     except:
    #         return (train_x, train_y, valid_x, valid_y)
    #     return (train_x, train_y, valid_x, valid_y)
    # finally:
    #     f.close()
    # print('%s: saved to file :'%(filename))
    return (train_x, train_y, valid_x, valid_y)


# safe-loading dataset file.(X, OneHat(Y))
def prepare_dataset_for_keras(filename = "data/dataset-ks.dat"):
    print("prepare_dataset_for_keras(%s)...."%(filename))
    print("=====================================")
    print("Start: Data Loading from Matrix file")
    print("=====================================")

    from keras.utils import np_utils

    mstack = prepare_transform_train()
    print("> mstack.count = "+str(mstack.count()))

    # prepare train/validation set (90%, 10%)
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []

    train_y_v = []
    valid_y_v = []

    count = mstack.count()
    for i in range(count):
        (x, y) = (mstack._matrix_list[i], mstack._matrix_list_y[i])
        y2 = np_utils.to_categorical(y, 100)
        y2 = y2.astype('int32')       # convert to int32

        if i % 10 != 2:
            train_x.append(x)
            train_y.append(y2)
            train_y_v += y
        else:
            valid_x.append(x)
            valid_y.append(y2)
            valid_y_v += y

    print("> v-stack ......")
    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    valid_x = np.vstack(valid_x)
    valid_y = np.vstack(valid_y)

    #train_y_v = train_y_v.astype('int32')
    #valid_y_v = valid_y_v.astype('int32')

    def print_stat(arr):
        from itertools import groupby
        #import copy
        #import pprint
        #arr = copy.deepcopy(arr)
        arr.sort()
        grps = ((k, len(list(g))) for k, g in groupby(arr))        # group counting.
        stat = np.fromiter(grps, dtype='u2,u2')
        print (stat)
        #pprint.pprint(stat)

    print("> train_y stat ......")
    print_stat(train_y_v)
    print("> valid_y stat ......")
    print_stat(valid_y_v)

    print("> train_x.count = %d, train_y.count = %d"%(len(train_x), len(train_y)))
    print("> valid_x.count = %d, valid_y.count = %d"%(len(valid_x), len(valid_y)))
    print('-------------- test data')
    for prnt_idx in [5]:            # Y must be 92 at 5th.
        print('> x[%d]='%(prnt_idx), train_x[prnt_idx])
        print('> y[%d]='%(prnt_idx), train_y[prnt_idx])
        print('> max-index[%d]=%d'%(prnt_idx, int(T.argmax(train_y[prnt_idx]).eval())))

    return (train_x, train_y, valid_x, valid_y)

'''
------------------------------------------------------------------------------------
-- logistic-sgd
------------------------------------------------------------------------------------
'''
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        rng = numpy.random.RandomState(1234)
        value = rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )

        self.W = theano.shared(
            #value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            value=value,
            #value=numpy.random.rand((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(value=numpy.zeros((n_out,),dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset, use_shared=True):
    print('... loading data (%s - type:%s)'%(dataset if type(dataset) is str else '', type(dataset)))

    # load dataset if tuple. (train_x, train_y, valid_x, valid_y)
    if type(dataset) is tuple:
        (train_x, train_y, valid_x, valid_y) = dataset
    elif type(dataset) is str:
        f = open(dataset, 'rb')
        try:
           (train_x, train_y, valid_x, valid_y) = pickle.load(f)
        finally:
            f.close()
    else:
        print('WARN! unknown dataset type:%s'%(type(dataset)))
        return False

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    if use_shared:
        test_set_x, test_set_y = shared_dataset(([],[]))
        valid_set_x, valid_set_y = shared_dataset((valid_x, valid_y))
        train_set_x, train_set_y = shared_dataset((train_x, train_y))
    else:
        test_set_x, test_set_y = ([],[])
        valid_set_x, valid_set_y = (valid_x, valid_y)
        train_set_x, train_set_y = (train_x, train_y)


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.000000001, n_epochs=1000,
                           dataset='data/dataset-00.dat',
                           batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    x_s = train_set_x.get_value(borrow=True).shape[1];
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    classifier = LogisticRegression(input=x, n_in=x_s, n_out=100)
    cost = classifier.negative_log_likelihood(y)
    # test_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.errors(y),
    #     givens={
    #         x: test_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: test_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        #outputs=classifier.negative_log_likelihood(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    # n_epochs = 1
    # n_train_batches = 10
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # print('minibatch_avg_cost %f %%' %(minibatch_avg_cost))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    '''
                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    '''
                    print('Best Model: epoch %i, minibatch %i/%i, load %f %% '%(epoch, minibatch_index + 1,
                                                                        minibatch_index, best_validation_loss*100))

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time)))


'''
------------------------------------------------------------------------------------
-- mlp
------------------------------------------------------------------------------------
'''
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input

def test_mlp(learning_rate=0.0001, L1_reg=0.00, L2_reg=0.001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=600000, n_hidden=100):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]

    x_s = train_set_x.get_value(borrow=True).shape[1];
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng, input=x, n_in=x_s,
        n_hidden=n_hidden, n_out=100
    )
    cost = (classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr)

    # test_model = theano.function(
    #     inputs=[index],
    #     outputs=classifier.errors(y),
    #     givens={
    #         x: test_set_x[index * batch_size:(index + 1) * batch_size],
    #         y: test_set_y[index * batch_size:(index + 1) * batch_size]
    #     }
    # )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    print("n_train_batches = %d"%(n_train_batches))
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            if minibatch_index % 100 == 0:
                print('minibatch_avg_cost: %f'%(minibatch_avg_cost))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # # test it on the test set
                    # test_losses = [test_model(i) for i
                    #                in range(n_test_batches)]
                    # test_score = numpy.mean(test_losses)
                    #
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #        'best model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))

            if False and patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

def predict():
    return

'''
------------------------------------------------------------------------------------
-- keras vanilla model
------------------------------------------------------------------------------------
'''
def test_keras(dataset, batch_size=600000):
    print('test_keras(batch_size=%d)...'%(batch_size))
    #from __future__ import print_function
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils

    # load dataset.
    datasets = load_data(dataset, use_shared=False)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # definitions
    in_dim = train_set_x.shape[1]
    mid_dim = in_dim
    out_dim = 100 if train_set_y.ndim == 1 else train_set_y.shape[1]
    nb_epoch = 250

    print('> train_x samples = %d, ndim=%d'%(train_set_x.shape[0], train_set_x.ndim), train_set_x.shape)
    print('> train_y samples = %d, ndim=%d'%(train_set_y.shape[0], train_set_y.ndim), train_set_y.shape)

    # convert to unit vector if not initiaize.
    print('> train_set_y.shape=',train_set_y.shape)
    if train_set_y.ndim == 1:
        print('> > try to make categorical by %d'%(out_dim))
        # at first, convert to int32
        train_set_y = train_set_y.astype('int32')
        valid_set_y = valid_set_y.astype('int32')
        # convert class vectors to binary class matrices
        train_set_y = np_utils.to_categorical(train_set_y, out_dim)
        valid_set_y = np_utils.to_categorical(valid_set_y, out_dim)


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    model = Sequential()
    model.add(Dense(output_dim=mid_dim, input_dim=in_dim))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(mid_dim))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=out_dim))
    model.add(Activation("softmax"))

    model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01))


    #model.fit(train_set_x, train_set_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_data=(valid_set_x, valid_set_x))
    model.fit(train_set_x, train_set_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    #model.train_on_batch(X_batch, Y_batch)
    #loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=batch_size)

    score = model.evaluate(valid_set_x, valid_set_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return

'''
------------------------------------------------------------------------------------
-- keras vanilla model 2 with Convolution
------------------------------------------------------------------------------------
'''
def test_keras_CNN(dataset, batch_size=60000):
    print('test_keras_CNN(batch_size=%d)...'%(batch_size))
    #from __future__ import print_function
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Convolution1D, Convolution2D, Lambda, MaxPooling1D, MaxPooling2D, Flatten
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils
    from keras import backend as K

    # load dataset.
    datasets = load_data(dataset, use_shared=False)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # definitions
    in_dim = train_set_x.shape[1]
    mid_dim = 100   #in_dim
    out_dim = 100 if train_set_y.ndim == 1 else train_set_y.shape[1]
    nb_epoch = 200

    print('> train_x samples = %d, ndim=%d'%(train_set_x.shape[0], train_set_x.ndim), train_set_x.shape)
    print('> train_y samples = %d, ndim=%d'%(train_set_y.shape[0], train_set_y.ndim), train_set_y.shape)

    # convert to unit vector if not initiaize.
    print('> train_set_y.shape=',train_set_y.shape)
    if train_set_y.ndim == 1:
        print('> > try to make categorical by %d'%(out_dim))
        # at first, convert to int32
        train_set_y = train_set_y.astype('int32')
        valid_set_y = valid_set_y.astype('int32')
        # convert class vectors to binary class matrices
        train_set_y = np_utils.to_categorical(train_set_y, out_dim)
        valid_set_y = np_utils.to_categorical(valid_set_y, out_dim)

    use_2D = False

    if use_2D:
        print('.... reshape-2D x')     # shape of vector x must be 164 = 4*41
        train_set_x = train_set_x.reshape(train_set_x.shape[0],1, 4, 41)
        valid_set_x = valid_set_x.reshape(valid_set_x.shape[0],1, 4, 41)
    else:
        print('.... reshape-1D x')     # shape of vector x must be 164 = 4*41
        train_set_x = train_set_x.reshape(train_set_x.shape[0],4, 41)
        valid_set_x = valid_set_x.reshape(valid_set_x.shape[0],4, 41)
        mid_dim = 128


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    print('input-x.shape', train_set_x.shape)

    def max_1d(X):
       return K.max(X, axis=1)

    model = Sequential()
    if use_2D:
        model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu', input_shape=(1, 4, 41)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
    else:
        #model.add(Convolution1D(32, 3, border_mode='valid', activation='relu', input_length=938108, input_dim=in_dim))
        #model.add(Convolution1D(32, 3, border_mode='valid', activation='relu', input_shape=(1, 164)))
        model.add(Convolution1D(32, 3, border_mode='valid', activation='relu', input_shape=(4, 41), subsample_length=1))
        #model.add(Lambda(max_1d, output_shape=(164,)))
        model.add(MaxPooling1D(pool_length=2))
        model.add(Flatten())

    model.add(Dense(mid_dim))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(mid_dim))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=out_dim))
    model.add(Activation("softmax"))

    model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print('> optimizer=', 'categorical_crossentropy', 'adam')

    #model.fit(train_set_x, train_set_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_data=(valid_set_x, valid_set_x))
    model.fit(train_set_x, train_set_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    #model.train_on_batch(X_batch, Y_batch)
    #loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=batch_size)

    score = model.evaluate(valid_set_x, valid_set_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return

#############################
# Self Test Main.
if __name__ == '__main__':
    main()