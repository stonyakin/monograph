import numpy as np
import math
from datetime import datetime
import random

def norm(x):
	if(np.linalg.norm(x)>R):
		x=R*x/np.linalg.norm(x)
	return x

def f(x):
	# return sum(np.linalg.norm(np.subtract(x,i)) for i in Dots)/r+0.0001*sum(np.power(x,2))/2
	# maximum square distance
	global D
	T=[sum((x-i)**2) for i in Dots]
	D=np.argmax(T)
	return max(T)

def grad_f(x):
	# return sum(np.subtract(x,i)/np.linalg.norm(np.subtract(x,i)) for i in Dots)/r+0.0001*np.multiply(x,2)/2
	return (x-Dots[D])*2 # maximum square distance

n=1000 # dimension
r=100 # number of dots
mu=2
L0=2
R=1
M=10000+1
N=[i for i in range(0,M,5000)] # list of iterations
ln=len(N)-1
x0=np.array([1]*n)
ex=2 # Number of experiments
q=1
me=[0 for i in range(ln)]
mt=[0 for i in range(ln)]
mf=[0 for i in range(ln)]

while(q<=ex):
	Dots=np.random.randint(-1,1,size=(r,n))
	print('Experiment #',q,'\n')
	L=L0 # L_0
	L/=2 # L_1
	delta=1/250 # delta_0
	delta/=2 # delta_1
	x=norm(x0)
	p=1
	u=1
	s=0
	k=0
	j=1
	pp=[0]*M
	dd=[0]*M
	F=None
	start_time=datetime.now()
	while(k<N[-1]):
		fx=f(x)
		if(F==None or f(x)<F):
			F=f(x)
		gf=grad_f(x)
		x1=x-np.multiply(1/L,gf)
		t=x1-x
		if(f(x1)<=fx+np.dot(gf,t)+L/2*sum(np.power(t,2))+delta):
			pp[k]=1-mu/L
			dd[k]=delta
			x=x1
			k+=1
			L/=2
			delta/=2
			if(k in N):
				for i in range(N[j-1],N[j]):
					p*=pp[i]
				for i in range(N[j-1],N[j]-1):
					for m in range(i,N[j]):
						u*=pp[m]
					s+=dd[i]*u
				E=p*f(x0)+s+delta
				me[j-1]+=E
				end_time=datetime.now()
				tt=(end_time-start_time).seconds
				mt[j-1]+=tt
				mf[j-1]+=F
				print('Iterations:',N[j])
				print('Estimate:',E)
				print('F(x):',F)
				print('Time:',tt,'s')
				j+=1
		else:
			L*=2
			delta*=2
	print()
	for i in range(ln):
		print('Iterations:',N[i+1])
		print('Mean estimate (adaptive):',me[i]/q)
		print('Mean F(x) (adaptive):',mf[i]/q)
		print('Mean time (adaptive):',mt[i]/q,'s')
	q+=1
	print()
