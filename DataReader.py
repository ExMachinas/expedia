# Data File Reader
# Author: dujung@kaist.ac.kr
#
import unittest
import gzip, csv, io
#import dateutil.parser as dp
import numpy as np          # for matrix handling.
from datetime import datetime, timedelta

##############################
# class: GzCsvReader
# - for reading ~.csv.gz file in data folder.
class GzCsvReader:
    def __init__(self, filename):
        self._filename = filename
        self._gzname = 'data/'+filename
        self._gzfile = io.BufferedReader(gzip.open(self._gzname, "r"))
        self._reader = csv.reader(io.TextIOWrapper(self._gzfile, newline=""))
        self._header = self._reader.next()        # header information.
        self.build_header_map()

    # header()
    def header(self):
        return self._header

    # make column map index.
    def build_header_map(self):
        #class ColInd( object ):
        #    pass
        #ret = ColInd()
        ret = self
        for i, name in enumerate(self._header):
            #print('> %s=%d'%(name, i))
            setattr(ret, name, i)
        return ret

    # next
    def next(self):
        try:
            return self._reader.next()
        except:
            return None

    # iterate support like (for x in self)
    def __iter__(self):
        return self._reader.__iter__()

    # close file.
    def close(self):
        self._gzfile.close()

##############################
# Common Data Conversion Func
#B = lambda x:True if int(x) != 0 else False     # boolean (better than int)
B = lambda x:1 if int(x) != 0 else 0            # boolean
I = lambda x:int(x) if x != '' else 0           # integer
F = lambda x:float(x) if x != '' else 0         # float
S = lambda x:x                                  # string
#D = lambda x:dp.parse(x + ' 12:00:00')          # date (yyyy-MM-dd)
T = lambda x:x                                  # time (hh:mm:ss)
#DT = lambda x:dp.parse(x)                       # date-time (yyyy-MM-dd hh:mm:ss) !WARN! EASY BUT TOO SLOW FUNCTION.

#------------------------------
# IMPROVE DATETIME PARSER.
Dd = lambda x: [int(i) for i in x.split('-')]    # date (yyyy-MM-dd)
Dt = lambda x: [int(i) for i in x.split(':')]    # time (hh:mm:ss)

EPOCH = datetime(1970, 1, 1) # use POSIX epoch

def DT(x):            # DT Datetime parser to seconds since EPOCH
    if x=='': return 0
    try:
        y = x.split(' ')
        d,t = [Dd(y[0]), Dt(y[1])]
        t0 = datetime(d[0], d[1], d[2], t[0], t[1], t[2])
        td = (t0 - EPOCH)           # timedelta
        return td.total_seconds()
    except:
        return -1                               # -1 means invalid-time-data.

def D(x):
    if x == '': return 0
    return DT(x + ' 12:00:00')

def DTR(timestamp):             # DT Reverse from second to Datetme.
    utc_time = EPOCH + timedelta(seconds=int(timestamp))
    return utc_time

MAX_ROW = -1                                     # maximum number of row to be read from csv (-1 means the unlimited)

#------------------------------
# time-usage print
import time
def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        func(*args, **kwargs)
        end_ts = time.time()
        print("%s - elapsed time: %f" % (func.func_name, end_ts - beg_ts))
    return wrapper

##############################
# class: MatrixStack
# - to handle large size of matrix in stack for cut/merge
MATSTACK_GRP_SIZE = 10              # 255
class MatrixStack():
    def __init__(self, name = None):
        self._name = name              # used in file saving.
        self._matrix = []              # to save group of matrix in array.
        self._matrix_256 = None
        self._count = 0                # number of rows in matrix.

    # push single array ..........................................
    def push(self, list, dtype=np.float32):
        np_list = np.array(list, dtype)

        # stack-up into matrix_255
        if self._matrix_256 is None:
            self._matrix_256 = np_list
        else:
            self._matrix_256 = np.vstack((self._matrix_256, np_list))

        # increase count
        self._count += 1

        # check if
        if self._count%MATSTACK_GRP_SIZE == 0:        # matrix_256 must be full.
            self.pack()

    # the count of array in _matrix (# of group of 256 matrix)
    def count(self):
        return self._count/MATSTACK_GRP_SIZE

    # get np.array() matrix from to .............................
    def get(self, start, end=None):
        max = self.count()
        end = max if end is None else end
        matrix = None
        for i in range(start, end):
            m = self._matrix[i] if i < max else None
            if m is None:
                break
            if matrix is None:
                matrix = m
            else:
                matrix = np.vstack((matrix, m))
        return matrix

    # clear buffer...............................................
    def reset(self):
        self._matrix = []
        self._matrix_256 = None
        self._count = 0

    # pack _matrix_256 into _matrix..............................
    def pack(self):
        if self._matrix_256 is not None:
            self._matrix.append(self._matrix_256)
            self._matrix_256 = None

    # get filename from given name
    def as_filename(self, name = None):
        name = name if name else self._name
        name = name if name else "def"
        filename = "data/mstack-"+name+".dat"
        return filename

    # save matrix into file......................................
    # @return True if saved
    def save_to_file(self, name=None):
        filename = self.as_filename(name)
        self.pack()
        if self._matrix is not None:
            from six.moves import cPickle
            f = open(filename, 'wb')
            cPickle.dump((self._count, self._matrix), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            print('matrix-stack: saved to file :'+filename+', count='+str(self._count))
            return True
        else:
            return False

    # load matrix object from file...............................
    # @return True if load (and file exists)
    def load_from_file(self, name=None):
        import os.path
        filename = self.as_filename(name)
        self.reset()
        matrix = None
        count = 0

        # check if file exists.
        if os.path.isfile(filename):
            from six.moves import cPickle
            f = open(filename, 'rb')
            (count, matrix) = cPickle.load(f)
            f.close()
        else:
            return False

        # check if data loaded.
        if matrix is not None:
            self._count = count
            self._matrix = matrix
            print('matrix-stack: loaded from file :'+filename+", count="+str(count))
            return True
        else:
            return False

    # test data itself ..........................................
    @time_usage
    def test(self, print_deep=False):
        # print current matrix status.
        cnt = lambda x: len(x) if x is not None else 0
        print('mstack[%s] count=%d, matrix.size=%d, mat256.size=%d '%(self._name, self._count, cnt(self._matrix), cnt(self._matrix_256)))
        if print_deep:
            for i,m in enumerate(self._matrix):
                print('--------- : [%d/%d]'%(i, cnt(self._matrix)))
                print(m[0:10,0:9])
            if self._matrix_256 is not None:
                print('--------- : [last]')
                print(self._matrix_256[0:10,0:9])


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

    # move next(), and return the converted [] array.
    def next(self):
        #list = super(self.__class__, self).next()
        list = GzCsvReader.next(self)
        if list is None:
            return []
        else:
            return self.filter(list)

    def filter(self, list):
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
            raise
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
    def populate(self, force_reload = False):
        return self.populate_4(force_reload)

    # populate all data into matrix.
    # Time - 3k => 1.2s ,4k => 2s, 5k => 3.8s
    def populate_1(self):
        list = self.next()
        matrix = np.array(list)
        for i,row in enumerate(self):
            matrix = np.vstack((matrix, row))               # array push
            if(MAX_ROW > 0 and i > MAX_ROW): break

        self._matrix = matrix
        return True

    # populate all data into matrix. (stack up every 100 list)
    # Time2 - 5k => 0.24s
    def populate_2(self):
        list = self.next()
        matrix = np.array(list)
        matrix_100 = np.array([])
        for i,row in enumerate(self):
            if(MAX_ROW > 0 and i > MAX_ROW): break
            if i%100 == 0:
                if matrix_100.size > 0:
                    matrix = np.vstack((matrix, matrix_100))      # array push
                matrix_100 = np.array(row)
                continue
            matrix_100 = np.vstack((matrix_100, row))               # array push

        if matrix_100.size > 0:
            matrix = np.vstack((matrix, matrix_100))      # array push

        self._matrix = matrix
        return True

    # populate all data into matrix. (stack up every 1000 list)
    # Time3 - 5k => 0.66s (at 1000), 0.22s at 200, 0.32 at 500
    # Time3 - 10k => 9.9s (1024), 6.43s (512), 6.36 (256)
    def populate_3(self):
        list = self.next()
        matrix = np.array(list, dtype=np.float32)
        matrix_256 = np.array([])
        for i,row in enumerate(self):
            list = self.filter(row)
            if(MAX_ROW > 0 and i > MAX_ROW): break
            if i%256 == 0:
                if matrix_256.size > 0:
                    matrix = np.vstack((matrix, matrix_256))      # array push
                    print("Rows:%d"%(matrix.shape[0]))
                    print(list)
                matrix_256 = np.array(list, dtype=np.float32)
                continue
            list_1 = np.array(list, dtype=np.float32)
            matrix_256 = np.vstack((matrix_256, list_1))               # array push

        if matrix_256.size > 0:
            matrix = np.vstack((matrix, matrix_256))      # array push

        self._matrix = matrix
        return True

    # populate#4 - save intermitent file every 1M lines. then rebuild.
    # Time: 290ms for 2048 data-read.
    @time_usage
    def populate_4(self, force_build = False):
        import os.path
        # merge back splited file into one file with vstack.
        @time_usage
        def merge_split_to_matrix(thiz, max):
            #- rebuild whole matrix from temp-file.
            matrix = None
            for fid in range(0, max):
                filename = "data/%s-%04d"%(thiz._filename, fid)
                is_file = os.path.isfile(filename)
                if not is_file: break
                thiz.load_from_file(filename)
                print("> [%d] loaded matrix.count = %d "%(fid, len(thiz._matrix)))
                if matrix is None:
                    matrix = thiz._matrix
                else:
                    matrix = np.vstack((matrix, thiz._matrix))               # array push input matrix

            print(">> Total matrix Count = %d"%(len(matrix)))
            thiz._matrix = matrix
            return matrix

        # check if the temp-file exists already.
        if not force_build:
            filename = "data/%s-%04d"%(self._filename, 0)
            is_file = os.path.isfile(filename)
            if is_file:
                print("INFO - start populating from cached files : "+filename)
                merge_split_to_matrix(self, 999)
                return True

        ############
        # Build temp files.
        RCOUNT = 256
        PACK_ROW = RCOUNT*1024*4        # 4k * 256 = 1M

        # read rows
        def read_rows(count):
            mat = None
            for i in range(0, count):
                row = self.next()
                row = np.array(row, dtype=np.float32)   # convert to float
                if row.size < 1: break;     # it must be EOL
                if mat is None:
                    mat = row
                else:
                    mat = np.vstack((mat, row))
            return mat

        # enumerate all rows.
        matrix = None
        next_id = 0
        i = 0
        while(True):
            rows = read_rows(RCOUNT)
            if rows is None: break          # must be EOL

            # increase row number
            i += rows.shape[0]

            # check max-row.
            if(MAX_ROW > 0 and i >= MAX_ROW):
                # init or push next-list
                if matrix is None:
                    matrix = rows
                else:
                    matrix = np.vstack((matrix, rows))               # array push
                break

            # print status every 1k
            if i%(RCOUNT*4) == 0:
                print("Rows: %d"%(i))

            # do every 1M lines
            if i%PACK_ROW == 0:
                if matrix is not None:
                    #save to temp file.
                    filename = "data/%s-%04d"%(self._filename, next_id)
                    self._matrix = matrix
                    self.save_to_file(filename)
                    next_id = next_id + 1
                    matrix = None
                # save current-list to matrix
                matrix = rows
                # next-loop
                continue

            # init or push next-list
            if matrix is None:
                matrix = rows
            else:
                matrix = np.vstack((matrix, rows))               # array push

        # for the remained data.
        if matrix is not None:
            #save to temp file.
            filename = "data/%s-%04d"%(self._filename, next_id)
            self._matrix = matrix
            self.save_to_file(filename)
            next_id = next_id + 1
            matrix = None

        # merge all temp-file to single
        merge_split_to_matrix(self, next_id)

        return True

    # find-out all value for column
    def cols(self, name):
        try:
            i = self._header.index(name)
            return self._matrix[:,i]
        except:
            return np.array([])

    # find-out all value for row
    def rows(self, line):
        try:
            return self._matrix[line]
        except:
            return np.array([])

    # save matrix to file
    def save_to_file(self, filename=None):
        filename = filename if filename else (self._filename + ".dat")
        matrix = self._matrix if hasattr(self, '_matrix') else None
        try:
            #! by using cPickle
            if matrix is not None:
                from six.moves import cPickle
                f = open(filename, 'wb')
                cPickle.dump(matrix, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
            print('> saved to file :'+filename)
            #end of cPickle
        except:
            print('WARN! failed to save to file :'+filename)
        return filename

    # load matrix object from file.
    def load_from_file(self, filename=None):
        import os.path
        filename = filename if filename else (self._filename + ".dat")
        matrix = None
        #! by using cPickle
        if os.path.isfile(filename):
            from six.moves import cPickle
            f = open(filename, 'rb')
            matrix = cPickle.load(f)
            f.close()
        #end of cPickle
        if matrix is not None:
            self._matrix = matrix
        print('> loaded from file :'+filename)
        return filename

    # clear all matrix data.
    def clear(self):
        self._matrix = np.array([])


    # count of rows in matrix
    def count(self):
        cnt = lambda x: len(x) if x is not None else 0
        return cnt(self._matrix)

    # auto-loading (or populating & save back to file from matrix)
    # Time Measure: 1.62s -> 0.08s with 1k data.
    def load_auto(self, force_populate = False):
        import os.path
        fname = "data/"+self._filename+".0.dat"
        print("INFO - load-auto : "+fname)
        #- if there is no 0.dat file, then start populate.
        is_file = os.path.isfile(fname)
        if not is_file or force_populate:
            print("INFO - started populating from gz-file"+self._gzname)
            self.populate(force_populate)
            is_file = False

        #- ok! now save back to file if not found.
        if not is_file:
            self.save_to_file(fname)
        else:
            self.load_from_file(fname)

        return fname

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
        self._map = None
        self._missed = {}           # not-found count of destination_id

    #! override def_filter()
    def def_filter(self, col, name):
        #print('> [%d] %s '%(col, name))
        #define filter-LUT
        LUT = {'srch_destination_id':lambda x:int(x)}
        try:
            return LUT[name]
        except:
            return (lambda x:float(x)) if 1>0 else (lambda x:x)

    #! build internal map from array.
    def build_map(self, rebuild = False):
        if ((not rebuild) and self._map is not None):
            return self._map
        map_dest = {}
        for m in self._matrix:
            map_dest[m[0]] = m.tolist()

        self._map = None if len(map_dest) < 1 else map_dest

    #! lookup dest_id from map
    def lookup(self, dest_id):
        try:
            return self._map[dest_id]
        except:
            if self._missed.has_key(dest_id):
                self._missed[dest_id] = self._missed[dest_id] + 1
            else:
                self._missed[dest_id] = 0

            if self._missed[dest_id] % 100 == 0:
                print('- WARN! dest not found id:%d (missed %d)'%(dest_id, self._missed[dest_id]+1))
            return None


##############################
# class: TestSheet
# - Test data handling.
# Rows:2,528,001
# [2528001, datetime.datetime(2015, 11, 13, 7, 29, 43), 2, 3, 66, 348, 18487, 251.7068, 1198021, False, False, 10, datetime.datetime(2015, 11, 25, 12, 0), datetime.datetime(2015, 11, 29, 12, 0), 1, 0, 1, 9524, 1, 2, 50, 561]
# --- hotel_market
# [27 1540 699 ..., 628 905 1490]
# > count=2115, min=0, max=2117 ---
# [('0', 55) ('1', 222) ('2', 17208) ..., ('2115', 37) ('2116', 72)
#  ('2117', 634)]
class TestSheet(DataSheet):
    def __init__(self):
        DataSheet.__init__(self, "test.csv.gz")

    #! override def_filter()
    def def_filter(self, col, name):
        LUT = {'id':I, 'date_time':DT, 'site_name':I ,'posa_continent':I
               ,'user_location_country':I, 'user_location_region':I, 'user_location_city':I,'orig_destination_distance':F
               ,'user_id':I, 'is_mobile':B, 'is_package':B, 'channel':I
               ,'srch_ci':D, 'srch_co':D, 'srch_adults_cnt':I,'srch_children_cnt':I,'srch_rm_cnt':I
               ,'srch_destination_id':I,'srch_destination_type_id':I
               ,'hotel_continent':I,'hotel_country':I,'hotel_market':I}
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
# Factory Class to load all required data-sheet
class DataFactory():
    instance = None
    def __init__(self, reload):
        print("make DataFactory()")
        self._reload = reload
        self.init_sheets()

    def init_sheets(self):
        map = {}
        #TODO:XENI - not yet use submission
        #map['submission'] = SubmissionSheet()
        map['destination'] = DestinationSheet()
        map['test'] = TestSheet()
        map['train'] = TrainSheet()
        self._map = map

        for k,o in map.iteritems():
            print("--------------------------------")
            print("Loading: "+str(k)+" -> "+str(o))
            if self._reload:
                o.load_auto(True)           # force to reload data.
            else:
                o.load_auto()               # normal loading.

    # get DataReader instance for the given name
    def get(self, name):
        return self._map[name]

    #@staticmethod
    @classmethod
    def load(cls, reload=False):
        #global instance
        if cls.instance is None:
            cls.instance = DataFactory(reload)
        return cls.instance

'''
------------------------------------------------------------------------------------------------------------------------
Transform Class
- transform each row of traint/test data to vector
'''
##############################
# class: TransTrain Case00
class TransTrain00(MatrixStack):
    def __init__(self):
        MatrixStack.__init__(self, "train00")

    # transform test-date to temporal matrix-stack array.
    def transform(self, train = None, dest = None, force = False):
        print("TransTrain00.transform(force=%s)...."%("True" if force else "False"))
        fact = DataFactory.load()
        train = train if train else fact.get('train')
        dest = dest if dest else fact.get('destination')

        # transform the input date to array [msec, week, holiday?]
        # @arg dmsec date-second since EPOCH (see DT() function)
        def trans_date(dsec, isSeason = False):
            ret = []
            d = DTR(dsec)
            #weekday : Monday is 0 and Sunday is 6
            HOLIDAY = [0,0,0,0,0.5,1,1]
            SEASON = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            SEASON_KEY = [0,0,1,1,1,2,2,2,3,3,3,0]      # season key by month.
            #ret.extend([d.year, d.month, d.day, d.weekday(), HOLIDAY[d.weekday()], d.hour])
            ret += [d.year, HOLIDAY[d.weekday()]]
            if isSeason:
                ret += SEASON[SEASON_KEY[d.month-1]]
            return ret

        def diff_date(co, ci):
            dx = [(co - ci)/(60*60*24)]
            #if True: print(str(DTR(co))+" - "+str(DTR(ci))+" = "+str(dx))
            return dx

        # transform each row of train/test data to single array.
        def trans_train_row(i, R, train):
            row = []
            row = [i]           #for test in order to track row number.
            # date-time (only save the 1st click-date, and difference from now)
            row += trans_date(R[train.date_time]) + trans_date(R[train.srch_ci], True) # + trans_date(R[train.srch_co], True)
            row += diff_date(R[train.srch_ci], R[train.date_time])     # ci - time (in day)
            row += diff_date(R[train.srch_co], R[train.srch_ci])      # co - ci (in day)
            # count
            row += [R[train.channel], R[train.is_mobile], R[train.is_package]]
            row += [R[train.srch_adults_cnt], R[train.srch_children_cnt], R[train.srch_rm_cnt]]
            # locations
            row += [R[train.posa_continent], R[train.user_location_country], R[train.user_location_region], R[train.user_location_city]]
            row += [R[train.srch_destination_type_id], R[train.hotel_continent], R[train.hotel_country], R[train.hotel_market]]
            dest_row = dest.lookup(R[train.srch_destination_id])
            #TODO:XENI - dest_row can be None, for now ignore this case (TODO IMPROVE)
            if dest_row is None:
                return None
            return row

        @time_usage
        def run_transform(mstack):
            #test-case : enumerate each set.
            for i,R in enumerate(train._matrix):
                row = trans_train_row(i, R, train)
                if row is None: continue
                mstack.push(row, np.int32)
                #print(str(DTR(R[train.date_time]))+':'+str(row))

        # MatrixStack
        #mstack = MatrixStack()
        mstack = self

        if force:
            run_transform(mstack)
            mstack.save_to_file()
        else:      # if failed to load
            loaded = self.load_from_file()
            if not loaded:
                run_transform(mstack)
                mstack.save_to_file()

        return True

'''
------------------------------------------------------------------------------------------------------------------------
Test Functions to verify each function method.
'''
##############################
# Unit Test Class.
class TestReader(unittest.TestCase):
    def test_sheet(self):
        print('test_sheet()...')
        test_timestamp()
        #test_DataReader()
        #test_Factory()
        test_matstack()

#print (dr._matrix)
def print_col(dd, name):
    from itertools import groupby
    print ("--- " + name)
    cols = dd.cols(name)
    print ('Count:'+str(cols.size))
    print (cols)
    if cols.size < 1:
        print (">WARN! - empty ");
        return
    cols.sort()
    grps = ((k, len(list(g))) for k, g in groupby(cols))            # grouping
    index = np.fromiter(grps, dtype='a8,u2')                        # a8 string len=8
    print ("> count="+str(index.shape[0])+", min="+str(cols.min())+", max="+str(cols.max())+" --- ")
    #print (cols)
    print (index)
    # group : see http://stackoverflow.com/questions/4651683/numpy-grouping-using-itertools-groupby-performance

##############################
# Unit Test Function.
# - test populate() and load() function
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

    # for quick debugging.
    global MAX_ROW
    MAX_ROW = 2500

    # enumerate by next()
    #for i in range(min,10):
    #    print(dr.next())
    #    #dr.next_print()

    #! use load_auto()
    #dr.populate()

    #print_col(dr, "hotel_cluster")
    #print_col(dr, "site_name")
    #print_col(dr, "user_location_country")
    #print_col(dr, "orig_destination_distance")
    #print_col(dr, "srch_destination_id")
    print_col(dr, "srch_destination_type_id")
    #print_col(dr, "hotel_continent")
    #print_col(dr, "hotel_country")
    print_col(dr, "hotel_market")
    #print_col(dr, "hotel_cluster")

    #save to file.
    dr.save_to_file("data/test.dat")

    #clear matrix data. => it must be empty [] array.
    dr.clear()
    print_col(dr, "hotel_market")

    #load back from file.  => it must print same result before saving.
    dr.load_from_file("data/test.dat")
    print_col(dr, "hotel_market")

    #ok! auto-load preliminary data (which was converted from original gz file, then saved back to file)
    #dr.load_auto(True)
    dr.load_auto()

    #print again.
    print_col(dr, "hotel_market")

# test : timestamp
def test_timestamp():
    # test - timestamp conversion
    t1 = "2014-02-27 17:44:32"
    t2 = DTR(DT(t1))
    print(t1 + ' == ' + str(t2))

# test : matrix-stack (to read/write matrix from/into files)
def test_matstack():
    print("============================ : test_matstack()")
    fact = DataFactory.load()
    print("---------------------------- : train")
    train = fact.get('train')
    print(train.header())
    print("---------------------------- : destination")
    dest = fact.get('destination')
    print(dest.header())

    #! step1. build-up lookuptable for destination.
    dest.build_map()

    # transform the input date to array [msec, week, holiday?]
    # @arg dmsec date-second since EPOCH (see DT() function)
    def trans_date(dsec, isSeason = False):
        ret = []
        d = DTR(dsec)
        #weekday : Monday is 0 and Sunday is 6
        HOLIDAY = [0,0,0,0,0.5,1,1]
        SEASON = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        SEASON_KEY = [0,0,1,1,1,2,2,2,3,3,3,0]      # season key by month.
        #ret.extend([d.year, d.month, d.day, d.weekday(), HOLIDAY[d.weekday()], d.hour])
        ret += [d.year, HOLIDAY[d.weekday()]]
        if isSeason:
            ret += SEASON[SEASON_KEY[d.month-1]]
        return ret

    def diff_date(co, ci):
        dx = [(co - ci)/(60*60*24)]
        #if True: print(str(DTR(co))+" - "+str(DTR(ci))+" = "+str(dx))
        return dx

    # transform each row of train/test data to single array.
    def trans_train_row(i, R, train):
        row = []
        row = [i]           #for test in order to track row number.
        # date-time (only save the 1st click-date, and difference from now)
        row += trans_date(R[train.date_time]) + trans_date(R[train.srch_ci], True) # + trans_date(R[train.srch_co], True)
        row += diff_date(R[train.srch_ci], R[train.date_time])     # ci - time (in day)
        row += diff_date(R[train.srch_co], R[train.srch_ci])      # co - ci (in day)
        # count
        row += [R[train.channel], R[train.is_mobile], R[train.is_package]]
        row += [R[train.srch_adults_cnt], R[train.srch_children_cnt], R[train.srch_rm_cnt]]
        # locations
        row += [R[train.posa_continent], R[train.user_location_country], R[train.user_location_region], R[train.user_location_city]]
        row += [R[train.srch_destination_type_id], R[train.hotel_continent], R[train.hotel_country], R[train.hotel_market]]
        dest_row = dest.lookup(R[train.srch_destination_id])
        #TODO:XENI - dest_row can be None, for now ignore this case (TODO IMPROVE)
        if dest_row is None:
            return None
        return row

    mstack = MatrixStack()

    #test-case : enumerate each set.
    for i,R in enumerate(train._matrix):
        row = trans_train_row(i, R, train)
        if row is None: continue
        mstack.push(row, np.int32)
        print(str(DTR(R[train.date_time]))+':'+str(row))

    mstack.test()
    mstack.save_to_file()
    mstack.reset()
    mstack.test()
    mstack.load_from_file()
    mstack.test()
    m2 = mstack.get(6)
    print('---------: mstack.get(6)')
    print(m2)
    count = mstack.count()
    print('count='+str(count))

# test : factory
def test_Factory():
    fact = DataFactory.load()
    print(fact)

    dest = fact.get('destination')
    print("---------------------------- : destination")
    print(dest.header())
    #print(dest.rows(10))
    #print_col(dest, 'srch_destination_id')

    train = fact.get('train')
    print("---------------------------- : train")
    print(train.header())
    print_col(train, 'hotel_cluster')
    #print_col(train, 'srch_destination_id')
    #print_col(train, 'channel')

    test = fact.get('test')
    print("---------------------------- : test")
    print(test.header())
    #print_col(test, 'hotel_cluster')
    #print_col(test, 'srch_destination_id')

    #! step1. build-up lookuptable for destination.
    dest.build_map()

    print('index of d142='+str(dest.d142))
    print('index of train.srch_destination_id='+str(train.srch_destination_id))
    print('index of test.date_time='+str(test.date_time))
    print('index of train.user_location_country='+str(train.user_location_country))

    srch_destination_id = train.cols('srch_destination_id')
    for i in srch_destination_id[10:20]:
        #print('search[id: %d]'%(i))
        d = dest.lookup(i)
        if d is None:
            print('- WARN! not found dest_id:%d'%(i))


    # print('-------- m2')
    # print(m2)

##############################
# Self Test Main.
if __name__ == '__main__':
    unittest.main()