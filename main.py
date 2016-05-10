# Main Application
# Author: dujung
#

import DataReader

def test_main():
    print("hello test-main")

    print("======================================")
    print("Start: Data Conversion to Matrix file")
    from DataReader import DataFactory
    fact = DataFactory.load()
    print("======================================")

# Self Test Main.
if __name__ == '__main__':
    test_main()