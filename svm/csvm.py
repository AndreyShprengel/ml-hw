from sklearn.svm import SVC
import argparse
import numpy
import pprint
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
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
    
    print "lenx " + str(len(data.train_x))
    
    x = numpy.nditer(data.train_y,flags=['f_index'])
    badI = list()
    while not x.finished:
        if not (x[0] == 3 or x[0] ==8):
            badI.append(x.index)
        x.iternext()
    
    data.train_y = numpy.delete(data.train_y,badI)
    data.train_x = numpy.delete(data.train_x,badI,0)
    print "leny " + str(len(data.test_y))
    x = numpy.nditer(data.test_y,flags=['f_index'])
    badI = list()
    while not x.finished:
        if not (x[0] == 3 or x[0] ==8):

            badI.append(x.index)
        x.iternext()
    print "leny " + str(len(data.test_y))

    data.test_y = numpy.delete(data.test_y,badI)
    data.test_x = numpy.delete(data.test_x,badI,0)
    pprint.pprint(data.train_x)
    pprint.pprint(data.train_y)

    f = open("data1.txt", "a")
    
    clf = SVC(C = 11, kernel = 'linear') 
    clf.fit(data.train_x, data.train_y)
    numRight = 0 
    for i in range(len(data.test_x)):
        if clf.predict(data.test_x[i]) == data.test_y[i]:
           numRight += 1
    acc = float(numRight)/len(data.test_x)
    print  "accuracy: " + str(acc)
    sv = clf.support_vectors_
    print "numsupvec" + str(len(sv))
    image_arr =  clf.support_vectors_[1] #the front parts of it are 3s, the rear ones are 8s 
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.imshow(numpy.reshape(image_arr, (28, 28)), cmap = cm.Greys_r)
    plt.show() 
       # f. write('linear' + ", " +str(acc) + "\n")
    image_arr =  clf.support_vectors_[len(clf.support_vectors_)-1] #the front parts of it are 3s, the rear ones are 8s 
    plt.imshow(numpy.reshape(image_arr, (28, 28)), cmap = cm.Greys_r)
    plt.show() 




