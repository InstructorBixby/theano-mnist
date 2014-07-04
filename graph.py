import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

xs = []
ys = []
for line in open('/home/panic/Desktop/mnist/curve.txt'):
    a = line.split()
    xs.append(int(a[0]))
    ys.append(float(a[1]))

plt.plot(xs, ys)
plt.xlabel('epoch')
plt.ylabel('test error')
plt.ylim(0, 0.08)
plt.grid(True)
plt.savefig('/home/panic/Desktop/mnist/curve.png')
print 'done'
