from sklearn.svm import SVC
import argparse
import numpy
import pprint
class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        
        f.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='c value')
    parser.add_argument('--C', type=int, default=1,
                        help="Number of nearest points to use")
    parser.add_argument('--kernel', type=str, default= "linear" ,
                        help="kernales")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    i = 0
    data.train_y = data.train_y[:args.limit]
    data.train_x = data.train_x[:args.limit]
    print len(data.train_y)
    
    x = numpy.nditer(data.train_y,flags=['f_index'])
    badI = list()
    while not x.finished:
        print x.index
        if not (x[0] == 3 or x[0] ==8):
            print x[0]
            badI.append(x.index)
        x.iternext()
    
    data.train_y = numpy.delete(data.train_y,badI)
    data.train_x = numpy.delete(data.train_x,badI)
    print data.train_y


        



