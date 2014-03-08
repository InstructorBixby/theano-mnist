import gzip
import cPickle

# mnist.pkl.gz off deeplearning.net/data/mnist/
# its a tuple of 3 and each one is (images, labels)

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()

print len(d)
print d[0][0].shape
print d[0][1].shape
print d[0][0].min(), d[0][0].max()
print d[0][1][0:10]
print "DONE"
