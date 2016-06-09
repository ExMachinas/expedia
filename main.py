# Main Application
# Author: dujung
#
import six.moves.cPickle as pickle
import numpy as np                              # numpy
from DataReader import DataFactory
from DataReader import DTR                      # DateTime Reverse Function.
from DataReader import MatrixStack

def main():
    print("main()....")

    # prepare dataset to run
    dataset = prepare_dataset()

    # run sgd-optimization with given dataset.
    if dataset is not None:
        sgd_optimization_mnist(dataset=dataset)
    else:
        sgd_optimization_mnist()

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
    import os.path
    if os.path.isfile(filename):
        return None

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
            #train_y.append(y)
            train_y += y
        else:
            valid_x.append(x)
            #valid_y.append(y)
            valid_y += y

    train_x = np.vstack(train_x)
    #train_y = np.concatenate(train_y)
    valid_x = np.vstack(valid_x)
    #valid_y = np.concatenate(valid_y)

    print("train_x.count = %d, train_y.count = %d"%(len(train_x), len(train_y)))
    print("valid_x.count = %d, valid_y.count = %d"%(len(valid_x), len(valid_y)))

    # ok now save this data into file..
    f = open(filename, 'wb')
    try:
        pickle.dump((train_x, train_y, valid_x, valid_y), f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('%s: failed to save file'%(filename))
        try:
            import os
            os.remove(filename)
        except:
            return (train_x, train_y, valid_x, valid_y)
        return (train_x, train_y, valid_x, valid_y)
    finally:
        f.close()
    print('%s: saved to file :'%(filename))
    return (train_x, train_y, valid_x, valid_y)

'''
------------------------------------------------------------------------------------
-- logistic-sgd
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
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
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

def load_data(dataset):
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

    test_set_x, test_set_y = shared_dataset(([],[]))
    valid_set_x, valid_set_y = shared_dataset((valid_x, valid_y))
    train_set_x, train_set_y = shared_dataset((train_x, train_y))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
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
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
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
                    print('Best Model: epoch %i, minibatch %i/%i, load %f %% '%(epoch, minibatch_index + 1, minibatch_index, best_validation_loss))

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


def predict():
    return

#############################
# Self Test Main.
if __name__ == '__main__':
    main()