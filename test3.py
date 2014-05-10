import gzip
import cPickle
import numpy
import theano
import theano.tensor as T

# http://deeplearning.net/tutorial/mlp.html

LR = 0.1
B = 20
NH = 1000

r = numpy.random.RandomState(1234)

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()
trX = d[0][0]
trY = d[0][1].astype('int32')
teX = d[1][0]
teY = d[1][1].astype('int32')

W1 = theano.shared(numpy.asarray(r.uniform(-0.01, 0.01, (784, NH)),
                                 dtype=theano.config.floatX))
b1 = theano.shared(numpy.zeros((NH,), dtype=theano.config.floatX))
W2 = theano.shared(numpy.asarray(r.uniform(-0.01, 0.01, (NH, 10)),
                                 dtype=theano.config.floatX))
b2 = theano.shared(numpy.zeros((10,), dtype=theano.config.floatX))

x = T.matrix('x')
y = T.ivector('y')

h = T.tanh(T.dot(x, W1) + b1)
p = T.nnet.softmax(T.dot(h, W2) + b2)
pred = T.argmax(p, axis=1)
cost = -T.mean(T.log(p)[T.arange(y.shape[0]), y])

ps = [W1, b1, W2, b2]
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
while e < 100:
    i = 0
    while i < nb:
        train(trX[i*B:i*B+B], trY[i*B:i*B+B])
        i = i + 1
    print e, err(teX, teY)
    e = e + 1
print "DONE"
