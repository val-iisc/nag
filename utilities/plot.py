import numpy as np
file_name = 'log_loss_imagenet.txt'
lst = []
with open(file_name,'r') as f:
  for l in f.readlines():
    l = l.strip()
    tmp = np.asarray(l)
    lst.append(tmp)
lst = np.asarray(lst)
np.save('log_loss_imagenet.npy',lst)
    
