import time
import theano
import theano.tensor as T

# how long does the compile actually take

t = time.time()
x = T.matrix('x')
y = T.nnet.softmax(T.dot(x, x.T))
f = theano.function([x], y)
print time.time() - t
# 28 seconds. thats gcc.
