import numpy as np
import math
from datetime import datetime
import random

def norm(x):
	if(np.linalg.norm(x)>1):
		x/=np.linalg.norm(x)
	# for i in range(n):
		# if(x[i]<=0):x[i]=1e-5
	return x

def G(g,x,tr):
	if(tr=='grad'):
		if(tg=='linear'):t=g
		if(tg=='abs'):t=np.multiply(g,np.sign(x))
		if(tg=='sqr'):t=np.multiply(g,np.multiply(2,x))
	if(tr=='g'):
		if(tg=='linear'):t=sum(np.multiply(g,x))+fm
		if(tg=='abs'):t=sum(np.multiply(g,np.abs(x)))+fm
		if(tg=='sqr'):t=sum(np.multiply(g,np.power(x,2)))+fm
	return t

# Modification.
def Grad_g1(x,alg):
	global M
	for gi in Coef:
		v=G(gi,x,'g')
		if(alg=='N'):
			M=np.linalg.norm(gi)
			v/=M
		if(v>e[i]):
			return G(gi,x,'grad')
	return [None]

# Standard.
def Grad_g(x,alg):
	global M
	tm=G(Coef[0],x,'g')
	gm=Coef[0]
	for gi in Coef:
		t=G(gi,x,'g')
		if(t>tm):
			tm=t
			gm=gi
	if(alg=='N'):
		M=np.linalg.norm(gm)
		tm/=M
	if(tm>e[i]):
		return G(gm,x,'grad')
	return [None]

def f(x):
	# return sum(np.linalg.norm(np.subtract(x,i)) for i in Dots)/r # Medium distance.
	# Maximum distance:
	global D
	T=[np.linalg.norm(np.subtract(x,i)) for i in Dots]
	D=np.argmax(T)
	return max(T)
	# return sum(math.sqrt(i) for i in x)/n # Square root.
	# Maximum distance (with rho):
	# global D,c
	# c=1
	# T=[]
	# for i in range(r):
		# dist=np.linalg.norm(np.subtract(x,Dots[i]))
		# if(dist<=Radius[i]):
			# T.append(rho*dist)
		# else:
			# T.append((rho-1)*Radius[i]+dist)
	# V=max(T)
	# D=np.argmax(T)
	# if(V<=Radius[D]):c=rho
	# return V

def Grad_f(x):
	# return sum(np.subtract(x,i)/np.linalg.norm(np.subtract(x,i)) for i in Dots)/r # Medium distance.
	return np.subtract(x,Dots[D])/np.linalg.norm(np.subtract(x,Dots[D])) # Maximum distance.
	# return [1/(2*n*math.sqrt(i)) for i in x] # Square root.	
	# Maximum distance (with rho):
	# T=np.subtract(x,Dots[D])
	# return c*T/np.linalg.norm(T)

def gen():
	t=[random.randint(-10,10) for i in range(n)]
	# return t/np.linalg.norm(t)*1.5
	return t

n=int(1e3) # Dimension.
m=20 # Number of constraints.
r=5 # Number of dots.
p=1
Coef=[[1 for i in range(n)] for j in range(m)]
Coef[0]=[1]*n
Coef[1]=[1]+[2]*(n-1)
Coef[2]=[1]+[3]*(n-1)
for j in range(3,m):
	Coef[j]=[1]+[i+p for i in range(1,n)]
	p+=1
tg='abs'
x0=norm([0.1]*n)
Th0s=2
e=[1/2,1/4,1/6,1/8]
E=['1/2','1/4','1/6','1/8']
# e=[1/2,1/4,1/6,1/8,1/10,1/12,1/14,1/16]
# E=['1/2','1/4','1/6','1/8','1/10','1/12','1/14','1/16']
fm=-1
l=len(e)
N=1 # Number of experiments.
mk=[[0,0] for i in range(l)]
mt=[[0,0] for i in range(l)]
rho=2
Radius=[1]*r

for j in range(N):
	print('Experiment #',j+1,'\r\n')
	Dots=[gen() for t in range(r)]
	# print('Algorithm 1')
	# print('================')
	# for i in range(l):
		# print('Epsilon =',E[i])
		# x=x0
		# k=0
		# I=0
		# S=0
		# start_time=datetime.now()
		# while(2*Th0s/e[i]**2>S+I):
			# grad_g=Grad_g(x,'I')
			# if(grad_g[0]==None):
				# fx=f(x)
				# grad_f=Grad_f(x)
				# M=np.linalg.norm(grad_f)
				# h=e[i]/M
				# y=norm(np.subtract(x,np.multiply(h,grad_f)))
				# # if(I==0 or f(X)>fx):
					# # X=x
				# I+=1
			# else:
				# Ms=sum(np.power(grad_g,2)) # Square norm.
				# h=e[i]/Ms
				# y=norm(np.subtract(x,np.multiply(h,grad_g)))
				# S+=1/Ms
			# x=y
			# k+=1
		# if(I!=0):
			# print('K =',k)
			# # print('Productive steps =',I)
			# # print('Ensure:',X)
		# else:
			# print("The set I is empty, k =",k)
		# end_time=datetime.now()
		# mk[i][0]+=k
		# t=(end_time-start_time).seconds
		# mt[i][0]+=t
		# print('Time:',t,'s')
		# print('------------------------------------')
	print('Algorithm 2')
	print('================')
	for i in range(l):
		print('Epsilon =',E[i])
		x=x0
		k=0
		I=0
		start_time=datetime.now()
		while(2*Th0s/e[i]**2>k):
			grad_g=Grad_g(x,'N')
			if(grad_g[0]==None):
				fx=f(x)
				grad_f=Grad_f(x)
				M=np.linalg.norm(grad_f)
				h=e[i]/M
				y=norm(np.subtract(x,np.multiply(h,grad_f)))
				if(I==0 or f(X)<fx):
					X=x
				I+=1
			else:
				h=e[i]/M
				y=norm(np.subtract(x,np.multiply(h,grad_g)))
			x=y
			k+=1
		if(I!=0):
			print('K =',k)
			# print('Productive steps =',I)
			print('f(X):',f(X))
			tt=[]
			for ii in Coef:
				tt.append(G(ii,X,'g'))
			print('Max G(X):',max(tt))
		else:
			print("The set I is empty, k =",k)
		end_time=datetime.now()
		mk[i][1]+=k
		t=(end_time-start_time).microseconds/1000
		# t=(end_time-start_time).seconds
		mt[i][1]+=t
		print('Time:',t,'ms')
		print('------------------------------------')
	print()
	for s in range(l):
		print('Epsilon =',E[s])
		print('Algorithm 1:')
		print('Mean K =',mk[s][0]/(j+1))
		print('Mean time:',mt[s][0]/(j+1),'s')
		print('Algorithm 2:')
		print('Mean K =',mk[s][1]/(j+1))
		print('Mean time:',mt[s][1]/(j+1),'ms')
	print('\r\n-----------------------\r\n')
