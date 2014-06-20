import gzip
import cPickle
import numpy

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
pr = a.argmax(axis=1)

m = numpy.zeros((10, 10))

i = 0
while i < len(teY):
    m[teY[i]][pr[i]] = m[teY[i]][pr[i]] + 1
    i = i + 1

j = 0
while j < 10:
    k = 0
    line = str(j) + ' '
    while k < 10:
        line = line + str(int(m[j][k])) + ' '
        k = k + 1
    print line
    j = j + 1
