#!/home/emneuron/usr/local/anaconda2/bin/python
from sys import argv
from os.path import join
from scandir import scandir

import numpy as np

subdir = argv[1]
loss = argv[2]

h,z = [],[]
for d in scandir(subdir):
    v_path = join(d.path, loss + '_loss.npy')
    v = np.load(v_path)
    if loss == 'val':
        print v[:,0].min(), v[:,1].min()
        h.append(v[:,0].min())
        z.append(v[:,1].min())
    else:
        print v
        h.append(v[0])
        z.append(v[1])
print 'MEAN ' + loss + ' LOSS: h=%s, z=%s' % (np.mean(h), np.mean(z))
