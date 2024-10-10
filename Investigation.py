#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:47:50 2022

@author: alanli
"""
from Comp_Proj_Balls import Ball
from Comp_Proj_Simulation import Simulation
import numpy as np


#%%
#Task 9
particles = [Ball(1000000000, 10, [0, 0], [0, 0])]
for i in range(0, 7):
    particles.append(Ball(1, 0.5, [-8+2.5*i, 0], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.5, [-8+2.5*i, -3], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.5, [-8+2.5*i, 3], 2*np.random.normal(0, 1, 2)))
for i in range(0, 5):
    particles.append(Ball(1, 0.5, [-6+2.5*i, 6], 2*np.random.normal(0, 1, 2)))
for i in range(0, 4):
    particles.append(Ball(1, 0.5, [-5+2.5*i, -6], 2*np.random.normal(0, 1, 2)))
s1 = Simulation(particles)
s1.histograms(5000, 35)

#%%
#Task 11, System kinetic energy versus time
particles = [Ball(1000000000, 10, [0, 0], [0, 0])]
for i in range(0, 7):
    particles.append(Ball(1, 0.5, [-8+2.5*i, 0], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.5, [-8+2.5*i, -3], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.5, [-8+2.5*i, 3], 2*np.random.normal(0, 1, 2)))
for i in range(0, 5):
    particles.append(Ball(1, 0.5, [-6+2.5*i, 6], 2*np.random.normal(0, 1, 2)))
for i in range(0, 4):
    particles.append(Ball(1, 0.5, [-5+2.5*i, -6], 2*np.random.normal(0, 1, 2)))
s1 = Simulation(particles)
s1.KE_time(20000)

#%%
#Task 11, Momentum versus time
particles = [Ball(1000000000, 10, [0, 0], [0, 0])]
for i in range(0, 7):
    particles.append(Ball(1, 0.5, [-8+2.5*i, 0], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.5, [-8+2.5*i, -3], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.5, [-8+2.5*i, 3], 2*np.random.normal(0, 1, 2)))
for i in range(0, 5):
    particles.append(Ball(1, 0.5, [-6+2.5*i, 6], 2*np.random.normal(0, 1, 2)))
for i in range(0, 4):
    particles.append(Ball(1, 0.5, [-5+2.5*i, -6], 2*np.random.normal(0, 1, 2)))
s1 = Simulation(particles)
s1.momentum_time(20000)

#%%
#Task 11, Pressure versus temperature
radlist = [0.1]
s1 = Simulation(particles)
s1.pressure_temp(1, 10, 1, 500, 400, radlist)

#%%
#Task 12, ball radius versus equation of state
radlist=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
s1 = Simulation(particles)
s1.pressure_temp(1, 10, 3, 500, 400, radlist)

#%%
#Task 13, Histogram of ball speeds & comparison to theoretical prediction
particles = [Ball(1000000000, 10, [0, 0], [0, 0])]
for i in range(0, 7):
    particles.append(Ball(1, 0.1, [-8+2.5*i, 0], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.1, [-8+2.5*i, -3], 2*np.random.normal(0, 1, 2)))
    particles.append(Ball(1, 0.1, [-8+2.5*i, 3], 2*np.random.normal(0, 1, 2)))
for i in range(0, 5):
    particles.append(Ball(1, 0.1, [-6+2.5*i, 6], 2*np.random.normal(0, 1, 2)))
for i in range(0, 4):
    particles.append(Ball(1, 0.1, [-5+2.5*i, -6], 2*np.random.normal(0, 1, 2)))
s1 = Simulation(particles)
s1.ball_speed_and_prediction(20000, 35)

#%%
#Task 14, Fitting Van Der Waal's law, shows b against ball radius plot and b 
#against ballarea plot with a best fit line 
radlist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
particles = []
s1 = Simulation(particles)
s1.van_der_waal(1, 10, 3, 500, 400, radlist)
