mnist with theano
=================

doing the tutorials off http://deeplearning.net/tutorial/

get mnist.pkl.gz with get.sh first. all the paths are hardcoded to
/home/panic/Desktop/mnist/ sorry.

test.py      loads it and prints the shapes
look.py 7    writes d7.png so you can see a digit
test2.py     logistic regression, 7.4%, 3 min
test3.py     one hidden layer, 1.7%, 40 min
conv.py      convnet, FOURTEEN MINUTES PER EPOCH, dont bother
try.py       loads model.pkl and predicts
check.py     confusion matrix
graph.py     draws curve.txt

test3_old.py is the broken one that sits on 11%, dont run it.

needs numpy theano PIL matplotlib
