import gzip
import cPickle
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

# http://deeplearning.net/tutorial/lenet.html

LR = 0.1
B = 500
K0 = 20
K1 = 50

r = numpy.random.RandomState(23455)

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()
trX = d[0][0]
trY = d[0][1].astype('int32')
teX = d[2][0]
teY = d[2][1].astype('int32')

def mkw(sh, fi, fo):
    z = numpy.sqrt(6.0 / (fi + fo))
    return theano.shared(numpy.asarray(r.uniform(-z, z, sh),
                                       dtype=theano.config.floatX))

x = T.matrix('x')
y = T.ivector('y')
inp = x.reshape((B, 1, 28, 28))

# 5x5 conv -> 24x24, pool 2 -> 12x12
W0 = mkw((K0, 1, 5, 5), 25, K0*25/4.0)
b0 = theano.shared(numpy.zeros((K0,), dtype=theano.config.floatX))
c0 = conv.conv2d(inp, W0, filter_shape=(K0, 1, 5, 5),
                 image_shape=(B, 1, 28, 28))
p0 = downsample.max_pool_2d(c0, (2, 2), ignore_border=True)
o0 = T.tanh(p0 + b0.dimshuffle('x', 0, 'x', 'x'))

# 5x5 conv -> 8x8, pool 2 -> 4x4
W1 = mkw((K1, K0, 5, 5), K0*25, K1*25/4.0)
b1 = theano.shared(numpy.zeros((K1,), dtype=theano.config.floatX))
c1 = conv.conv2d(o0, W1, filter_shape=(K1, K0, 5, 5),
                 image_shape=(B, K0, 12, 12))
p1 = downsample.max_pool_2d(c1, (2, 2), ignore_border=True)
o1 = T.tanh(p1 + b1.dimshuffle('x', 0, 'x', 'x'))

fl = o1.flatten(2)

W2 = mkw((K1*16, 500), K1*16, 500)
b2 = theano.shared(numpy.zeros((500,), dtype=theano.config.floatX))
hh = T.tanh(T.dot(fl, W2) + b2)

W3 = theano.shared(numpy.zeros((500, 10), dtype=theano.config.floatX))
b3 = theano.shared(numpy.zeros((10,), dtype=theano.config.floatX))
p = T.nnet.softmax(T.dot(hh, W3) + b3)
pred = T.argmax(p, axis=1)
cost = -T.mean(T.log(p)[T.arange(y.shape[0]), y])

ps = [W0, b0, W1, b1, W2, b2, W3, b3]
gs = T.grad(cost, ps)
up = []
i = 0
while i < len(ps):
    up.append((ps[i], ps[i] - LR*gs[i]))
    i = i + 1

train = theano.function([x, y], cost, updates=up)
err = theano.function([x, y], T.mean(T.neq(pred, y)))

nb = trX.shape[0] / B
e = 0
while e < 200:
    i = 0
    while i < nb:
        train(trX[i*B:i*B+B], trY[i*B:i*B+B])
        i = i + 1
    # err only takes exactly B rows because image_shape is fixed, so
    # this is the first 500 test images only, NOT the real test error
    print e, err(teX[0:B], teY[0:B])
    e = e + 1
