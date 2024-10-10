#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:44:38 2022

@author: alanli
"""
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from Comp_Proj_Balls import Ball


class Simulation:

    def __init__(self, particles):
        '''


        Parameters
        ----------
        particles : list
            List containing ball data type elements.

        Returns
        -------
        None.

        '''
        self.p = particles
        self.dP = 0
        self.containerp = [0, 0]

    def get_pressure(self, num_frames, countframe):
        '''


        Parameters
        ----------
        num_frames : int
            Number of collisions.
        countframe : int
            Start calculating average after this collision.

        Returns
        -------
        float
            Average pressure after countframe.

        '''
        totaltime = 0
        pressure = 0
        pressurelist = []
        for frame in range(num_frames):
            totaltime = totaltime+self.time_to_next_collision()
            pressure = self.dP/(totaltime*np.pi*2*10)
            if frame >= countframe:
                pressurelist.append(pressure)
            self.next_collision()
        return sum(pressurelist)/(num_frames-countframe)

    def histograms(self, num_frames, num_bins):
        '''
        This method plots the histograms for task 9

        Parameters
        ----------
        num_frames : int
            Number of collisions.
        num_bins : int
            Number of bins for the histograms.

        Returns
        -------
        None.

        '''
        distfromcenter = []
        distfromeachother = []
        for frame in range(num_frames):
            for i in self.p[1:]:
                distfromcenter.append(np.linalg.norm(i.pos()))
            for i in range(1, len(self.p)):
                for j in range(i+1, len(self.p)):
                    distfromeachother.append\
                        (np.linalg.norm(np.abs(self.p[i].pos()-self.p[j].\
                                               pos())))
            self.next_collision()
        plt.figure()
        plt.hist(distfromcenter, num_bins)
        plt.xlabel('Ball distance from container center')
        plt.ylabel('Frequency')
        plt.style.use('seaborn')
        radius = self.p[1].rad()
        mass = self.p[1].mas()
        plt.savefig(f'a 30(m={mass}, R={radius}, {num_bins}bins.jpg', dpi\
                    =500)

        plt.figure()
        plt.hist(distfromeachother, num_bins)
        plt.xlabel('Ball distance from each other')
        plt.ylabel('Frequency')
        plt.style.use('seaborn')
        plt.savefig(f'b 30(m = {mass}, R = {radius}, {num_bins}bins.jpg', dpi\
                    =500)

    def KE_time(self, num_frames):
        '''
        This method makes the KE vs time plot for task 11

        Parameters
        ----------
        num_frames : int
            Number of collisions.

        Returns
        -------
        None.

        '''
        totaltime = 0
        keplot = []
        timeplot = []
        for frame in range(num_frames):
            timeplot.append(totaltime)
            totaltime = totaltime+self.time_to_next_collision()
            keplot.append(self.kinetic_energy())
            self.next_collision()
        plt.figure()
        plt.plot(timeplot, keplot)
        plt.xlabel('Time')
        plt.ylabel('System KE')
        plt.style.use('seaborn')
        plt.savefig('System KE vs time', dpi=500)

    def momentum_time(self, num_frames):
        '''
        This method makes the momentum vs time plot for task 11

        Parameters
        ----------
        num_frames : int
            Number of collisions.

        Returns
        -------
        None.

        '''
        totaltime = 0
        timeplot = []
        momentumplot = []
        for frame in range(num_frames):
            timeplot.append(totaltime)
            totaltime = totaltime+self.time_to_next_collision()
            totalmomentum = 0
            for i in self.p:
                totalmomentum = totalmomentum+i.mas()*i.vel()
            momentumplot.append(np.linalg.norm(totalmomentum))
            self.next_collision()
        plt.figure()
        plt.plot(timeplot, momentumplot)
        plt.xlabel('Time')
        plt.ylabel('System Momentum')
        plt.style.use('seaborn')
        plt.savefig('System Momentum vs time', dpi=500)

    def pressure_temp(self, sigmastart, sigmaend, sigint, num_frames,\
                      countframes, radlist):
        '''
        This method makes the pressure vs temperature plot for task 11 and
        the plot that illustrates the equation of the state for different
        ball radii in task 12.

        Parameters
        ----------
        sigmastart : float
            The starting sigma value for the gaussian distribution of ball
            velocity.
        sigmaend : float
            The ending sigma value for the gaussian distribution of ball
            velocity.
        sigint : int
            The step for the sigma value.
        num_frames : int
            Number of collisions.
        countframe : int
            Start calculating average after this collision.
        radlist : list
            The list of ball radii.

        Returns
        -------
        None.

        '''
#Constant R plots can be made by using radlist with one elment
        idealgradient = 30/(np.pi*100)
        gradientdiff = []
        for r in radlist:
            avgke = []
            avgpressure = []
            for a in np.arange(sigmastart, sigmaend, sigint):
                particles = [Ball(1000000000, 10, [0, 0], [0, 0])]
                for i in range(0, 7):
                    particles.append(Ball(1, r, [-8+2.5*i, 0], 2*np.random.\
                                          normal(0, a, 2)))
                    particles.append(Ball(1, r, [-8+2.5*i, -3], 2*np.random.\
                                          normal(0, a, 2)))
                    particles.append(Ball(1, r, [-8+2.5*i, 3], 2*np.random.\
                                          normal(0, a, 2)))
                for i in range(0, 5):
                    particles.append(Ball(1, r, [-6+2.5*i, 6], 2*np.random.\
                                          normal(0, a, 2)))
                for i in range(0, 4):
                    particles.append(Ball(1, r, [-5+2.5*i, -6], 2*np.random.\
                                          normal(0, a, 2)))
                total_v = 0
                for i in range(1, 31):
                    total_v = total_v + (np.linalg.norm(particles[i].vel()))**2
                    i = i+1
                avgke.append(0.5*total_v/30)
                avgp = Simulation(particles).get_pressure(num_frames,
                                                          countframes)
                avgpressure.append(avgp)
                print("rms velocity", np.sqrt(total_v/30))
                print("avg KE", 0.5*total_v/30)
                print(f'avg pressure after {countframes} collisions', avgp)
                print(" ")

            fit, cov = np.polyfit(avgke, avgpressure, 1, True)
            gradientdiff.append(fit-idealgradient)
            xrange = np.arange(-100, 500, 1)
            yrange = []
            for i in xrange:
                yrange.append(idealgradient*i)

            plt.plot(avgke, fit*np.array(avgke), label=f'Radius = {r}')
            plt.plot(avgke, avgpressure, 'P', markersize=5)
            plt.xlabel('Temperature/$k_B$')
            plt.ylabel('Pressure')
            plt.xlim(-20, 200)
            plt.ylim(-5, 25)
        plt.plot(xrange, yrange, color='fuchsia', label='Ideal Gas Line')
        plt.legend()
        plt.savefig('P vs T different radius', dpi=1000)

        plt.figure()
        plt.plot(radlist, gradientdiff, 'P', markersize=5)
        plt.xlabel('Radius')
        plt.ylabel('Gradient difference from ideal gas')
        plt.savefig('Task 12 state', dpi=1000)

    def maxwell(self, v):
        total_v = 0
        for i in self.p[1:]:
            total_v = total_v + (np.linalg.norm(i.vel()))**2
        T = 0.5*total_v/30
        return (v/T)*np.exp((-v*v)/(2*T))

    def ball_speed_and_prediction(self, num_frames, num_bins):
        '''
        This method makes a histogram for the accumulative ball velocity and
        it fits the Maxwell-Boltzmann distribution curve onto the histogram.

        Parameters
        ----------
        num_frames : int
            Number of collisions.
        num_bins : int
            Number of bins for the histogram.

        Returns
        -------
        None.

        '''
        speedplot = []
        for frame in range(num_frames):
            for i in self.p[1:]:
                speedplot.append(np.linalg.norm(i.vel()))
            self.next_collision()

        plt.figure()
        height, bins, patches = plt.hist(speedplot, num_bins)
        plt.hist(speedplot, num_bins)

        x = np.linspace(0, 8, 100)
        y = (max(height)/max(self.maxwell(x)))*self.maxwell(x)
        plt.plot(x, y, label='Theoretical distribution')
        plt.xlabel('Speed')
        plt.ylabel('Frequency')
        plt.legend()
        plt.style.use('seaborn')
        plt.savefig('ball speed histogram.jpg', dpi=500)

    def van_der_waal(self, sigmastart, sigmaend, sigint, num_frames,\
                     countframe, radlist):
        '''
        This method makes a b against ball radius plot and a b against ball
        area plot with a best fit line.

        Parameters
        ----------
        sigmastart : float
            The starting sigma value for the gaussian distribution of ball
            velocity.
        sigmaend : float
            The ending sigma value for the gaussian distribution of ball
            velocity.
        sigint : int
            The step for the sigma value.
        num_frames : int
            Number of collisions.
        countframe : int
            Start calculating average after this collision.
        radlist : list
            The list of ball radii.

        Returns
        -------
        None

        '''
        idealgradient = 30/(np.pi*100)
        gradientdiff = []
        b_value = []
        ballarea = []
        for r in radlist:
            ballarea.append(np.pi*r*r)
            avgke = []
            avgpressure = []
            for a in np.arange(sigmastart, sigmaend, sigint):
                particles = [Ball(1000000000, 10, [0, 0], [0, 0])]
                for i in range(0, 7):
                    particles.append(Ball(1, r, [-8+2.5*i, 0], 2*np.random.\
                                          normal(0, a, 2)))
                    particles.append(Ball(1, r, [-8+2.5*i, -3], 2*np.random.\
                                          normal(0, a, 2)))
                    particles.append(Ball(1, r, [-8+2.5*i, 3], 2*np.random.\
                                          normal(0, a, 2)))
                for i in range(0, 5):
                    particles.append(Ball(1, r, [-6+2.5*i, 6], 2*np.random.\
                                          normal(0, a, 2)))
                for i in range(0, 4):
                    particles.append(Ball(1, r, [-5+2.5*i, -6], 2*np.random.\
                                          normal(0, a, 2)))
                total_v = 0
                for i in range(1, 31):
                    total_v = total_v + (np.linalg.norm(particles[i].vel()))**2
                    i = i+1
                avgke.append(0.5*total_v/30)
                avgp = Simulation(particles).get_pressure(num_frames,\
                                                          countframe)
                avgpressure.append(avgp)
                print("rms velocity", np.sqrt(total_v/30))
                print("avg KE", 0.5*total_v/30)
                print(f'avg pressure after {countframe} collisions', avgp)
                print(" ")

            def vdwEqn(T, b):
                return (30*T)/(np.pi*100-30*b)
            fit, cov = curve_fit(vdwEqn, avgke, avgpressure)
            b_value.append(fit)
            gradientdiff.append(30/(np.pi*100-30*fit)-idealgradient)
        fit, cov = np.polyfit(ballarea, b_value, 1, True)
        x_range = np.arange(0, 3.2, 0.1)
        y_range = []
        for i in x_range:
            y_range.append(i*fit)

        plt.figure()
        plt.plot(x_range, y_range, label=f'Best fit line, gradient = {fit}')
        plt.legend()
        plt.plot(ballarea, b_value, 'P', markersize=10)
        plt.xlabel('Ball area')
        plt.ylabel('b')
        plt.savefig(f'vanderwaal area {fit}.jpg', dpi=1000)

        plt.figure()
        plt.plot(radlist, b_value, 'P', markersize=10)
        plt.xlabel('Radius')
        plt.ylabel('b')
        plt.savefig(f'vanderwaal radius {fit}.jpg', dpi=1000)

    def run(self, num_frames, animate=False):
        '''
        This method runs the simulation and shows the animation of the
        collisions.

        Parameters
        ----------
        num_frames : int
            Number of collisions.
        animate : Boolean, optional
            Show animation. The default is False.

        Raises
        ------
        Exception
            When the number of ball reduce.

        Returns
        -------
        None.

        '''
        totaltime = 0
        if animate:
            pl.figure(figsize=[5, 5])
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self.p[0].get_patch())
            patch = []
            for i in range(1, len(self.p)):
                patch.append(self.p[i].get_patch())
                ax.add_patch(patch[i-1])
        pressureplot = []
        timeplot = []
        for frame in range(num_frames):
            if len(self.p) < len(self.p):
                raise Exception("Ball gone")
            print("Current time", totaltime)
            timeplot.append(totaltime)
            totaltime = totaltime+self.time_to_next_collision()
            print("Pressure", self.dP/(totaltime*np.pi*2*10))
            pressureplot.append(self.dP/(totaltime*np.pi*2*10))
            self.next_collision()
            if animate:
                path_diff = [0]*len(patch)
                patch_prev = [0]*len(patch)
                for i in range(0, len(patch)):
                    patch_prev[i] = patch[i].center
                    path_diff[i] = self.p[i+1].pos()-patch[i].center
                for i in np.linspace(0, 1, 101):
                    pl.pause(0.001)
                    for x in range(0, len(patch)):
                        patch[x].center = patch_prev[x]+i*path_diff[x]
        if animate:
            pl.show()

        plt.figure()
        plt.plot(timeplot, pressureplot)
        plt.show()

    def kinetic_energy(self):
        totalke = 0
        for i in self.p:
            totalke = totalke+0.5*i.mas()*np.linalg.norm(i.vel())*np.linalg.\
                norm(i.vel())
        return totalke

    def time_to_next_collision(self):
        '''
        

        Returns
        -------
        shortest_time : float
            The smallest time to next collision in the current frame.

        '''
        shortest_time = 100000
        for i in range(0, len(self.p)):
            for j in range(i+1, len(self.p)):
                if Ball.time_to_collision(self.p[i], self.p[j]) <\
                        shortest_time:
                    if Ball.time_to_collision(self.p[i], self.p[j]) == 0:
                        break
                    else:
                        shortest_time = Ball.time_to_collision(self.p[i],\
                                                               self.p[j])
        return shortest_time

    def next_collision(self):
        '''
        This method finds the shortest time until next collision, calculates
        the change in momentum for the container, moves the balls by the
        shortest time until collision, and collides the balls to update their
        velocities after the collision

        Returns
        -------
        Comp_Proj.Simulation
            Returns the updated balls in simulation class.

        '''
        shortest_time = 100002
        collisioni = []
        collisionj = []
        for i in range(0, len(self.p)):
            for j in range(i+1, len(self.p)):
                if Ball.time_to_collision(self.p[i], self.p[j]) <\
                        shortest_time:
                    if Ball.time_to_collision(self.p[i], self.p[j]) == 0:
                        break
                    else:
                        shortest_time = Ball.time_to_collision(self.p[i],\
                                                               self.p[j])
                        collisioni = []
                        collisionj = []
                        collisioni.append(i)
                        collisionj.append(j)
                elif Ball.time_to_collision(self.p[i], self.p[j]) ==\
                        shortest_time:
                    collisioni.append(i)
                    collisionj.append(j)
        for x in range(0, len(collisioni)):
            if collisioni[x] == 0:
                v_before = self.p[collisionj[x]].vel()
                v_after = Ball.pressure_collide(self.p[collisioni[x]],\
                                                self.p[collisionj[x]])[1]
                changemomentum = np.linalg.norm(self.p[collisionj[x]].mas() *\
                                                (v_before-v_after))
                self.dP = self.dP+changemomentum
            if collisionj[x] == 0:
                v_before = self.p[collisioni[x]].vel()
                v_after = Ball.pressure_collide(self.p[collisioni[x]],\
                                                self.p[collisionj[x]])[0]
                changemomentum = np.linalg.norm(self.p[collisioni[x]].mas() *\
                                                (v_before-v_after))
                self.dP = self.dP+changemomentum

        for i in self.p:
            i.move(shortest_time)

        for x in range(0, len(collisioni)):
            Ball.collide(self.p[collisioni[x]], self.p[collisionj[x]])

        return Simulation(self.p)
