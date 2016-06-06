# Main Application
# Author: dujung
#

import numpy as np                              # numpy
from DataReader import DataFactory
from DataReader import DTR                      # DateTime Reverse Function.
from DataReader import MatrixStack

def main():
    print("main()....")

    print("======================================")
    print("Start: Data Conversion to Matrix file")
    print("======================================")
    do_load_file()             # load data.
    #do_load_file(True)          # force to reload
    do_transform_train()        # transform data.

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
    dest.build_map()

# transform test-date to temporal matrix-stack array.
def do_transform_train():
    from DataReader import TransTrain00
    # load transformer.
    mstack = TransTrain00()
    mstack.transform()
    #mstack.transform(force=True)
    mstack.test()

# Self Test Main.
if __name__ == '__main__':
    main()