f = open('ilsvrc_train.txt').readlines()
for i,name in enumerate(f):
    if '_10' in name:
       g = open('ilsvrc_valid.txt','a')
       g.write(name)
       g.close()	

