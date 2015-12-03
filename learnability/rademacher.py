from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi, tan
import itertools

from numpy import array
from numpy.linalg import norm

from bst import BST

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


class Classifier:
    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))
        
        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"
        summ = 0
        for i in range(len(data)):
            summ += labels[i] * (1 if self.classify(data[i]) else -1)
        return float(summ)/(len(data))


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
            (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """
    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.

    """
    def __init__(self, start_x, start_y, end_x, end_y):
        """

        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
            (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
            (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
            (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
            (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    thetas = set()
    
    #produce angles, remove divby 0 errors.
    for x in dataset:
        if x[0] == 0:
            x += .00000000001
        thetas.add(atan(x[1]/x[0]))

    thetas = list(thetas)
    thetas.sort()
   
    
    planes = list()
    classifications =  list()
    #create first
    classifier_theta = thetas[0] - .00000001
    y = tan(classifier_theta)
    posHH = OriginPlaneHypothesis(1, y)
    negHH = OriginPlaneHypothesis(-1,-y)
    planes.append(posHH)
    planes.append(negHH)
    classifications.append([posHH.classify(x) for x in dataset])
    classifications.append([negHH.classify(x) for x in dataset])
    for i in range(len(thetas)-1):
        classifier_theta = (thetas[i]+thetas[i+1])/2 + pi/2
        y = tan(classifier_theta)
        #test if already classify data
        
        posHH = OriginPlaneHypothesis(1, y)
        negHH = OriginPlaneHypothesis(-1,-y)
        posClass = [posHH.classify(x) for x in dataset]
        negClass = [negHH.classify(x) for x in dataset]
        
        if (negHH not in classifications):
            planes.append(negHH)
            classifications.append(negClass)
        if (posHH not in classifications):
            planes.append(posHH)
            classifications.append(posClass)
  

    return iter(planes)

def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    
    
    # Complete this for extra credit
    return


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    Args:
      dataset: The dataset to use to generate hypotheses
    """
    n = len(dataset)
    squares = list()
    classifications =  list()
    #all neg hypothsis

    rec = AxisAlignedRectangle(0, 0, 0, 0)
    classifications.append([rec.classify(x) for x in dataset])
    squares.append(rec)

    #all single point hypothses
    for point in dataset:
        rec = AxisAlignedRectangle(point[0],point[1],point[0],point[1])
        classifications.append([rec.classify(x) for x in dataset])
        squares.append(rec)
    for x in range(n,1,-1):
        for y in list(itertools.combinations(dataset, x)):
 
            xMax = max(y, key=lambda item:item[0])[0]
            xMin = min(y, key=lambda item:item[0])[0]
            yMax = max(y, key=lambda item:item[1])[1]
            yMin = min(y, key=lambda item:item[1])[1]
             
            rec = AxisAlignedRectangle(xMin, yMin, xMax, yMax)
            clsf = [rec.classify(x) for x in dataset]
            #check if classifier exists
            if clsf not in classifications:
                classifications.append([rec.classify(x) for x in dataset])
                squares.append(rec)
            
    print classifications
            
    # TODO: complete this function
    return iter(squares)


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    Args:
      number: The number of coin tosses to perform

      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in xrange(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator

      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset

      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """

    hyps = list(hypothesis_generator(dataset))
    samples = list(coin_tosses(num_samples * len(dataset), random_seed))
    samples = [samples[i:i+len(dataset)] for i in range(0, len(samples), len(dataset))]
    print samples
    ex = 0
    for x in range(0,num_samples):
        labels = samples[x]
        ex += max([hyp.correlation(
            dataset, labels) for hyp in hyps])

        
    # TODO: complete this function
    return ex/num_samples


if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
