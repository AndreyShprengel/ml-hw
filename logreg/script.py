import os
def drange(start, stop, step):
	
    r = start
    while r < stop:
		yield r
		r += step
x = drange( 1.0, 10.0, 1.0)
for i in x:
	os.system('python logreg.py --step ' + str(i) +  ' --passes 10')
	 
