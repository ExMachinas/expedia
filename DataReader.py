# Data File Reader
# Author: dujung@kaist.ac.kr
#
import unittest
import gzip, csv, io
import dateutil.parser as dp
import numpy as np          # for matrix handling.

##############################
# class: GzCsvReader
# - for reading ~.csv.gz file in data folder.
class GzCsvReader:
    def __init__(self, filename):
        gzfile = 'data/'+filename
        self._gzfile = io.BufferedReader(gzip.open(gzfile, "r"))
        self._reader = csv.reader(io.TextIOWrapper(self._gzfile, newline=""))
        self._header = self._reader.next()        # header information.

    # header()
    def header(self):
        return self._header

    # next
    def next(self):
        return self._reader.next()

    # iterate support like (for x in self)
    def __iter__(self):
        return self._reader.__iter__()

    # close file.
    def close(self):
        self._gzfile.close()

##############################
# Common Definitions
B = lambda x:True if int(x) != 0 else False     # boolean
I = lambda x:int(x) if x != '' else 0           # integer
F = lambda x:float(x) if x != '' else 0         # float
S = lambda x:x                                  # string
D = lambda x:dp.parse(x + ' 12:00:00')          # date (yyyy-MM-dd)
T = lambda x:x                                  # time (hh:mm:ss)
DT = lambda x:dp.parse(x)                       # date-time (yyyy-MM-dd hh:mm:ss)


##############################
# class: DataSheet
# - Abstract data-sheet (convert all string-valued value to int, float or datatime)
class DataSheet(GzCsvReader):
    def __init__(self, filename):
        #super(self.__class__, self).__init__(filename)             - SEEMS NOT WORK IN VER 2.x
        GzCsvReader.__init__(self, filename)

        # filter function for each column data.
        self._filters = [self.def_filter(i,name) for i,name in enumerate(self._header)]

        # flag to auto-convert
        self._is_conv = True

    # move next()
    def next(self):
        #list = super(self.__class__, self).next()
        list = GzCsvReader.next(self)
        #out = [self.conv(x) for x in list]
        if self._is_conv:
            out = [self.conv(i,v) for i,v in enumerate(list)]
        else:
            out = list
        return out

    # find out the target filter function.
    def find_filter(self, col):
        return self._filters[col]

    # define filter by col-index
    def def_filter(self, col, name):
        print(':[%d] %s '%(col, name))
        i = lambda x: int(x)
        s = lambda x: x
        return i if col < 1 else s

    # execute conversion by id & value,
    def conv(self, i, v):
        try:
            f = self.find_filter(i)
            return f(v) if f else v
        except:
            print('ERR! convert colume %s - "%s" '%(i,str(v)))
            return v

    # print next row with column name
    def next_print(self):
        row = self.next()
        print('------------------------------------------------------')
        for i,name in enumerate(self._header):
            if isinstance(row[i], basestring):
                print("[%02d] %30s = '%s'"%(i, self._header[i], str(row[i])))
            else:
                print("[%02d] %30s = %s"%(i, self._header[i], str(row[i])))

    # populate all data into matrix.
    def populate_(self):
        matrix = []
        #for i in range(0, 99):    matrix.append(self.next())
        for row in self: matrix.append(row)
        self._matrix = np.array(matrix)

    # populate all data into matrix.
    def populate(self):
        list = self.next()
        matrix = np.array(list)
        #for i in range(0, 10): matrix = np.vstack((matrix, self.next()))
        #for row in self: matrix = np.vstack((matrix, row))
        for i,row in enumerate(self):
            matrix = np.vstack((matrix, row))               # array push
            if(i > 200): break

        self._matrix = matrix

    # find-out all value for column
    def cols(self, name):
        try:
            i = self._header.index(name)
            return self._matrix[:,i]
        except:
            return []



##############################
# class: SubmissionSheet
# - submisssion data type handling.
class SubmissionSheet(DataSheet):
    def __init__(self):
        DataSheet.__init__(self, "sample_submission.csv.gz")

    #! override def_filter()
    def def_filter(self, col, name):
        #print('> [%d] %s '%(col, name))
        #define filter-LUT
        LUT = {'id':lambda x:int(x)}
        try:
            return LUT[name]
        except:
            return lambda x:x


##############################
# class: DestinationSheet
# - Destination data type handling.
class DestinationSheet(DataSheet):
    def __init__(self):
        DataSheet.__init__(self, "destinations.csv.gz")

    #! override def_filter()
    def def_filter(self, col, name):
        #print('> [%d] %s '%(col, name))
        #define filter-LUT
        LUT = {'srch_destination_id':lambda x:int(x)}
        try:
            return LUT[name]
        except:
            return (lambda x:float(x)) if 1>0 else (lambda x:x)


##############################
# class: TestSheet
# - Test data handling.
class TestSheet(DataSheet):
    def __init__(self):
        DataSheet.__init__(self, "test.csv.gz")

    #! override def_filter()
    def def_filter(self, col, name):
        LUT = {'id':I, 'date_time':DT, 'site_name':I ,'posa_continent':I
               ,'user_location_country':I, 'user_location_region':I, 'user_location_city':I,'orig_destination_distance':F
               ,'user_id':I, 'is_mobile':B, 'is_package':B, 'channel':I
               ,'srch_ci':D, 'srch_co':D, 'srch_adults_cnt':I,'srch_children_cnt':I,'srch_rm_cnt':I
               ,'srch_destination_id':I,'srch_destination_type_id':I,'is_booking':B
               ,'cnt':I,'hotel_continent':I,'hotel_country':I,'hotel_market':I,'hotel_cluster':I}
        # LUT = {}
        try:
            return LUT[name]
        except:
            return lambda x:x


##############################
# class: TrainSheet is same as TestSheet
# - Train data handling.
# ['date_time', 'site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster']
class TrainSheet(DataSheet):
    def __init__(self):
        DataSheet.__init__(self, "train.csv.gz")

    #! override def_filter()
    def def_filter(self, col, name):
        LUT = {'id':I, 'date_time':DT, 'site_name':I ,'posa_continent':I
               ,'user_location_country':I, 'user_location_region':I, 'user_location_city':I,'orig_destination_distance':F
               ,'user_id':I, 'is_mobile':B, 'is_package':B, 'channel':I
               ,'srch_ci':D, 'srch_co':D, 'srch_adults_cnt':I,'srch_children_cnt':I,'srch_rm_cnt':I
               ,'srch_destination_id':I,'srch_destination_type_id':I,'is_booking':B
               ,'cnt':I,'hotel_continent':I,'hotel_country':I,'hotel_market':I,'hotel_cluster':I}
        # LUT = {}
        try:
            return LUT[name]
        except:
            return lambda x:x


##############################
# Unit Test Class.
class TestReader(unittest.TestCase):
    def test_sheet(self):
        print('test_sheet()...')
        test_DataReader()


##############################
# Unit Test Function.
def test_DataReader(max=5, min=0):
    gzfile="sample_submission.csv.gz"
    print ("hello test DataReader --- ")
    #dr = GzCsvReader(gzfile)
    #dr = DataSheet(gzfile)
    #dr = SubmissionSheet()
    #dr = DestinationSheet()
    #dr = TestSheet()
    dr = TrainSheet()

    # print header first
    print(dr.header())

    # enumerate by next()
    #for i in range(min,10):
    #    print(dr.next())
    #    #dr.next_print()

    dr.populate()
    #print (dr._matrix)
    def print_col(dd, name):
        from itertools import groupby
        print ("--- " + name)
        cols = dd.cols(name)
        cols.sort()
        grps = ((k, len(list(g))) for k, g in groupby(cols))            # grouping
        index = np.fromiter(grps, dtype='a8,u2')                        # a8 string len=8
        print ("> count="+str(index.shape[0])+", min="+str(cols.min())+", max="+str(cols.max())+" --- ")
        #print (cols)
        print (index)
        # group : see http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance

    #print_col(dr, "hotel_cluster")
    print_col(dr, "site_name")
    print_col(dr, "user_location_country")
    print_col(dr, "orig_destination_distance")
    print_col(dr, "srch_destination_id")
    print_col(dr, "srch_destination_type_id")
    print_col(dr, "hotel_continent")
    print_col(dr, "hotel_country")
    print_col(dr, "hotel_market")
    print_col(dr, "hotel_cluster")

    x = print_col
    x(dr, "srch_rm_cnt")

##############################
# Self Test Main.
if __name__ == '__main__':
    unittest.main()