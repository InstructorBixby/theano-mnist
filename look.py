import sys
import gzip
import cPickle
import numpy
from PIL import Image

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()
trX = d[0][0]
trY = d[0][1]

i = int(sys.argv[1])
a = trX[i].reshape(28, 28) * 255
im = Image.fromarray(a.astype(numpy.uint8))
im = im.resize((112, 112))
im.save('/home/panic/Desktop/mnist/d' + str(i) + '.png')
print trY[i]
print 'saved'
