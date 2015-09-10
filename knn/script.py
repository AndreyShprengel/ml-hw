import os
k = 3
for l in range(500 , 20000, 1000):
	os.system('python knn.py --limit ' + str(l) + ' --k  ' + str(k))
