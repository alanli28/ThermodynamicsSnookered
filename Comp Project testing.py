#%%
from Comp_Proj_Balls import Ball
from Comp_Proj_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
#%%
ball1=Ball(100000000,10,[0,0],[0,0])
ball2=Ball(1,1,[0,9],[0,-1])
print(Ball.time_to_collision(ball1,ball2))
#ball1.vel()
# %%
#particles=[Ball(1000000000,10,[0,0],[0,0]),Ball(1,1,[0,0],[0,1])]
particles=[Ball(1000000000,10,[0,0],[0,0]),Ball(1,1,[0,2],[0,1]),Ball(1,1,[0,-2],[3,-2])]
s1= Simulation(particles)
s1.run(200,animate=True)
# %%
print(np.abs(-100))
# %%
import pylab as pl

f = pl.figure()
patch = pl.Circle(np.array([-10., -10.]), 2, ec='b', fc='None')
ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.add_patch(patch)

for i in range(-10, 10):
    patch.center = [i, i]
    pl.pause(0.001)
pl.show()
#%%
particles=[Ball(1000000000,10,[0,0],[0,0])]
for i in range (0,7):
    particles.append(Ball(1,1,[-8+2.5*i,0],2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1,1,[-8+2.5*i,-3],2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1,1,[-8+2.5*i,3],2*np.random.normal(0, 1, 2)))
for i in range (0,5):
    particles.append(Ball(1,1,[-6+2.5*i,6],2*np.random.normal(0, 1, 2)))
for i in range (0,4):
    particles.append(Ball(1,1,[-5+2.5*i,-6],2*np.random.normal(0, 1, 2)))
s1= Simulation(particles)
s1.histograms(200, 15)
#%%
particles=[Ball(1000000000,10,[0,0],[0,0])]
for i in range (0,10):
    particles.append(Ball(1,0.5,[-9+2*i,0],2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1,0.5,[-8+1.6*i,-3],2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1,0.5,[-8+1.6*i,3],2*np.random.normal(0, 1, 2)))
print(len(particles))
total_v= 0 
for i in range (1,31):
    total_v= total_v + (np.linalg.norm(particles[i].v))**2
    i=i+1
print("rms velocity",np.sqrt(total_v/30))
print("avg KE",0.5*total_v/30)
#%%
def maxwell(v,T):
    return (v/T)*np.exp((-v*v)/(2*T))
x=np.linspace(0,8,100)
y=106000*maxwell(x,0.5*total_v/30)
plt.plot(x,y)
print(max(y))
#plt.show()
#%%
#10balls
particles=[Ball(1000000000,10,[0,0],[0,0])]
for i in range (0,10):
    particles.append(Ball(1,0.5,[-9+2*i,0],2*np.random.normal(0, 1, 2)))
print(len(particles))
total_v= 0 
for i in range (1,11):
    total_v= total_v + (np.linalg.norm(particles[i].v))**2
    i=i+1
print("rms velocity",np.sqrt(total_v/10))
print("avg KE",0.5*total_v/10)
#%%
s1= Simulation(particles)
s1.ballspeed(15000,35)
#%%
#21 seconds to run
s1= Simulation(particles)
print("avg pressure", s1.getpressure(10,6))
#%%
print(np.arange(1,10,0.2))
#%%
#Task11C
avgke=[]
avgpressure=[]
for a in np.arange(1,8,1):
    particles=[Ball(1000000000,10,[0,0],[0,0])]
    for i in range (0,10):
        particles.append(Ball(1,0.01,[-9+2*i,0],2*np.random.normal(0, a, 2)))
        particles.append(Ball(1,0.01,[-8+1.6*i,-3],2*np.random.normal(0, a, 2)))
        particles.append(Ball(1,0.01,[-8+1.6*i,3],2*np.random.normal(0, a, 2)))
    total_v= 0 
    for i in range (1,31):
        total_v= total_v + (np.linalg.norm(particles[i].v))**2
        i=i+1
    avgke.append(0.5*total_v/30)
    avgp= Simulation(particles).getpressure(500,400)
    avgpressure.append(avgp)
    print("rms velocity",np.sqrt(total_v/30))
    print("avg KE",0.5*total_v/30)
    print("avg pressure after 900 collisions",avgp)
    print(" ")

#%%
radlist=[0.2, 0.3, 0.4, 0.5]
particles=[]
s1= Simulation(particles)
s1.pressure_temp(1, 11, 3, 500, 400, radlist)
#%%
radlist=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
particles=[]
s1= Simulation(particles)
s1.pressuretemp(1, 10, 3, 500, 400, radlist)
#%%
radlist=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
particles=[]
s1= Simulation(particles)
s1.vanderwaal(1, 10, 3, 500, 400, radlist)
#%%
particles=[]
s1= Simulation(particles)
s1.vanderwaal(1, 10, 3, 500, 400)
#%%
gradient=30/(np.pi*100)
xrange=np.arange(-300,300,1)
yrange=[]
for i in xrange:
    yrange.append(gradient*i)
plt.plot(xrange,yrange)
plt.plot(avgke,avgpressure,'o')
plt.show()
#%%
particles=[Ball(1000000000,10,[0,0],[0,0]),Ball(1,1,[0,0],[0,-1])]
#%%
s1= Simulation(particles)
s1.run(1000,animate=False)
#%%
particles=[Ball(1000000000,10,[0,0],[0,0]),Ball(10000,5,[0,0],[0,0]),Ball(1000,1,[1,0],[1,1])]
s1= Simulation(particles)
s1.run(100,animate=True)
#%%
print(np.random.normal(0, 1, 2))
#%%
print(np.exp(2))
#%%
print(type(Simulation(particles)))