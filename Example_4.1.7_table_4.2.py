import numpy as np
import math
from datetime import datetime
import random

def norm(x):
	if(np.linalg.norm(x)>R):
		x=R*x/np.linalg.norm(x)
	return x

def f(x):
	return sum(np.power(np.dot(A,x)-b,2))/2

def fd(x):
	return sum(np.power(np.dot(A,x)-b,2))/2-dh

def grad_f(x):
	return np.dot(At,np.dot(A,x)-b)/2

n=1000 # dimension
nz=10 # maximum of non-zero elements of the matrix
R=1
M=10000+1
N=[i for i in range(0,M,1000)] # list of iterations
ln=len(N)-1
d=1/100 # delta
D=1/4 # Delta
x0=np.array([0.1]*n)
ex=5 # Number of experiments
q=1
me=[[0,0] for i in range(ln)]
mt=[[0,0] for i in range(ln)]
mf=[[0,0] for i in range(ln)]
mfd=[[0,0] for i in range(ln)]

while(q<=ex):
	print('Experiment #',q,'\n')
	A=np.diagflat(np.random.randint(1,10,size=n))
	for kk in range(nz):
		while(True):
			ii=random.randint(0,n-1)
			jj=random.randint(0,n-1)
			if(A[ii][jj]==0 and ii!=jj):
				A[ii][jj]=random.randint(1,10)
				break
	# b=np.random.random(n) # random floats in the half-open interval [0.0, 1.0)
	b=np.random.randint(-10,11,size=n) # random integers from -10 (inclusive) to 11 (exclusive)
	# b=np.random.standard_gamma(1,size=n) # samples from a standard Gamma distribution (k=1, Theta=1)
	At=np.transpose(A)
	L0=max(np.linalg.eig(np.dot(At,A))[0]).real
	mu=min(np.linalg.eig(np.dot(At,A))[0]).real
	L=L0 # L_0
	L/=2 # L_1
	delta=1/200 # delta_0
	delta/=2 # delta_1
	Delta=1/200 # Delta_0
	Delta/=2 # Delta_1
	print('Adaptive')
	print('================')
	x=x0
	p=1
	u=1
	s=0
	k=0
	j=1
	pp=[0]*M
	dd=[0]*M
	F=None
	Fd=None
	start_time=datetime.now()
	while(k<N[-1]):
		dh=random.uniform(0,d) # delta_hat
		if(F==None or f(x)<F):
			F=f(x)
		if(Fd==None or fd(x)<Fd):
			Fd=fd(x)
		gf=grad_f(x)
		gfw=[random.uniform(gf[i]-D/math.sqrt(n),gf[i]+D/math.sqrt(n)) for i in range(n)]
		gw=np.linalg.norm(gfw)
		x1=x-np.multiply(1/L*(1-Delta/gw),gfw)
		t=x1-x
		if(fd(x1)<=fd(x)+np.dot(gfw,t)+L/2*sum(np.power(t,2))+Delta*np.linalg.norm(t)+delta):
			pp[k]=1-mu/L*((gw-Delta)/(gw+D))**2
			dd[k]=delta
			x=x1
			k+=1
			L/=2
			delta/=2
			Delta/=2
			if(k in N):
				for i in range(N[j-1],N[j]):
					p*=pp[i]
				for i in range(N[j-1],N[j]-1):
					for m in range(i,N[j]):
						u*=pp[m]
					s+=(d+dd[i])*u
				E=p*f(x0)+s+delta+d
				me[j-1][1]+=E
				end_time=datetime.now()
				tt=(end_time-start_time).seconds
				mt[j-1][1]+=tt
				mf[j-1][1]+=F
				mfd[j-1][1]+=Fd
				print('Iterations:',N[j])
				print('Estimate:',E)
				print('F(x):',F)
				print('Fd(x):',Fd)
				print('Time:',tt,'s')
				j+=1
		else:
			L*=2
			delta*=2
			Delta*=2
	print()
	print('Non-adaptive')
	print('================')
	x=x0
	p=1
	u=1
	s=0
	k=0
	j=1
	pp=[0]*M
	F=None
	start_time=datetime.now()
	while(k<N[-1]):
		if(F==None or f(x)<F):
			F=f(x)
		gf=grad_f(x)
		gfw=[random.uniform(gf[i]-D/math.sqrt(n),gf[i]+D/math.sqrt(n)) for i in range(n)]
		gw=np.linalg.norm(gfw)
		x1=x-np.multiply(1/L0*(1-D/gw),gfw)
		pp[k]=1-mu/L0*((gw-D)/(gw+D))**2
		x=x1
		k+=1
		if(k in N):
			for i in range(N[j-1],N[j]):
				p*=pp[i]
			for i in range(N[j-1],N[j]-1):
				for m in range(i,N[j]):
					u*=pp[m]
				s+=u
			E=p*f(x0)+2*d*(s+1)
			me[j-1][0]+=E
			end_time=datetime.now()
			tt=(end_time-start_time).seconds
			mt[j-1][0]+=tt
			mf[j-1][0]+=F
			print('Iterations:',N[j])
			print('Estimate:',E)
			print('F(x):',F)
			print('Time:',tt,'s')
			j+=1
	print()
	for i in range(ln):
		print('Iterations:',N[i+1])
		print('Mean estimate (adaptive):',round(me[i][1]/q,5))
		print('Mean F(x) (adaptive):',mf[i][1]/q)
		print('Mean Fd(x) (adaptive):',mfd[i][1]/q)
		print('Mean time (adaptive):',mt[i][1]/q,'s')
		print('Mean estimate (non-adaptive):',round(me[i][0]/q,5))
		print('Mean F(x) (non-adaptive):',mf[i][0]/q)
		print('Mean time (non-adaptive):',mt[i][0]/q,'s')
	q+=1
	print()
