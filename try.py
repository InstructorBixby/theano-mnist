import sys
import gzip
import cPickle
import numpy

# no theano in here, its two matrix multiplies, numpy can do that

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()
trX = d[0][0]
trY = d[0][1].astype('int32')
teX = d[2][0]
teY = d[2][1].astype('int32')

W1, b1, W2, b2 = cPickle.load(open('/home/panic/Desktop/mnist/model.pkl'))

h = numpy.tanh(numpy.dot(teX, W1) + b1)
a = numpy.dot(h, W2) + b2
ex = numpy.exp(a - a.max(axis=1).reshape(-1, 1))
p = ex / ex.sum(axis=1).reshape(-1, 1)
pr = p.argmax(axis=1)

print (pr != teY).mean()

if len(sys.argv) > 1:
    i = int(sys.argv[1])
    print 'said', pr[i], 'actually', teY[i], p[i][pr[i]]
