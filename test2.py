import gzip
import cPickle
import numpy
import theano
import theano.tensor as T

# http://deeplearning.net/tutorial/logreg.html

LR = 0.13
# LR = 1.0    # cost goes to nan
# LR = 0.01   # works but takes forever

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()
trX = d[0][0]
trY = d[0][1].astype('int32')
teX = d[1][0]
teY = d[1][1].astype('int32')

W = theano.shared(numpy.zeros((784, 10), dtype=theano.config.floatX))
b = theano.shared(numpy.zeros((10,), dtype=theano.config.floatX))

x = T.matrix('x')
y = T.ivector('y')

p = T.nnet.softmax(T.dot(x, W) + b)
pred = T.argmax(p, axis=1)
cost = -T.mean(T.log(p)[T.arange(y.shape[0]), y])

gW = T.grad(cost, W)
gb = T.grad(cost, b)

train = theano.function([x, y], cost, updates=[(W, W - LR*gW), (b, b - LR*gb)])
err = theano.function([x, y], T.mean(T.neq(pred, y)))

B = 600
nb = trX.shape[0] / B

e = 0
while e < 50:
    i = 0
    while i < nb:
        c = train(trX[i*B:i*B+B], trY[i*B:i*B+B])
        i = i + 1
    print e, c, err(teX, teY)
    e = e + 1
print "DONE"
