import numpy as np
import pylab as pl


m_container = 1000


class Ball:

    def __init__(self, mass=1, radius=1, position=np.array([0, 0]), velocity=np
                 .array([0, 0])):
        '''


        Parameters
        ----------
        mass : float, optional
            The mass of the ball. The default is 1.
        radius : float, optional
            The radius of the ball. The default is 1.
        position : np array, optional
            The position of the ball. The default is np.array([0, 0]).
        velocity : np array, optional
            The vector velocity of the ball. The default is np.array([0, 0]).

        Returns
        -------
        None.

        '''
        if isinstance(position, list):
            position = np.asarray(position)
        if isinstance(velocity, list):
            velocity = np.asarray(velocity)
        self.__m = mass
        self.__R = radius
        self.__r = position
        self.__v = velocity

    def __repr__(self):
        '''


        Returns
        -------
        str
            Returns the mass, radius, position, and velocity of the ball.

        '''
        return "mass = %s, radius = %s, position = %r. velocity = %r" %\
            (self.__m, self.__R, self.__r, self.__v)

    def pos(self):
        return self.__r

    def rad(self):
        return self.__R

    def vel(self):
        return self.__v

    def mas(self):
        return self.__m

    def move(self, dt):
        '''


        Parameters
        ----------
        dt : float
            Time the ball will move for.

        Returns
        -------
        np array
            The new position of the ball.

        '''
        self.__r = self.__r+self.__v*dt
        return self.__r

    def time_to_collision(self, other):
        '''
        

        Parameters
        ----------
        other : Ball
            The ball that will collide with self.

        Returns
        -------
        float
            Time to collision acording to different conditions.

        '''
        r = self.__r-other.__r
        v = self.__v-other.__v
        if other.__m > m_container or self.__m > m_container:
            R = self.__R-other.__R
        else:
            R = self.__R+other.__R
        t1, t2 = (- (2 * np.dot(r, v)) + np.sqrt((2 * np.dot(r, v)) ** 2 - 4 *\
                                                 (np.dot(r, r)-R**2)*np.dot\
                                                 (v, v)))/(2*np.dot(v, v)),\
        (- (2 * np.dot(r, v)) - np.sqrt((2 * np.dot(r, v)) ** 2 - 4 *\
                                        (np.dot(r, r) - R ** 2) * np.dot(v, v)\
         ))/(2*np.dot(v, v))
        if t1 < 1e-8 and t2 < 1e-8:
            return 10000
        elif np.abs(t1) < 1e-8 or np.abs(t2) < 1e-8:
            return max(np.abs(t1), np.abs(t2))
        elif t1 > 1e-14 and t2 > 1e-14:
            return min(t1, t2)
        elif t1 < 1e-8 and t2 > 1e-8:
            return t2
        elif t1 > 1e-8 and t2 < 1e-8:
            return t1
        elif np.isnan(t1) or np.isnan(t2):
            return 10000

    def collide(self, other):
        '''
        This method calculates the velocities of the balls after collision and
        updates them.

        Parameters
        ----------
        other : Ball
            The ball that will collide with self.

        Returns
        -------
        None.

        '''
        v1 = self.__v - (2 * other.__m) / (self.__m + other.__m) *\
            (np.inner(self.__v - other.__v, self.__r - other.__r)) /\
            (np.linalg.norm(self.__r - other.__r)) ** 2 *\
            (self.__r - other.__r)
        v2 = other.__v - (2 * self.__m) / (self.__m + other.__m) *\
            (np.inner(other.__v - self.__v, other.__r - self.__r)) / \
            (np.linalg.norm(other.__r - self.__r)) ** 2 *\
            (other.__r - self.__r)
        self.__v = v1
        other.__v = v2

    def pressure_collide(self, other):
        '''
        This methods calculates the velocities of the balls after collision
        without updating them to self.

         Parameters
         ----------
         other : Ball
             The ball that will collide with self.

        Returns
        -------
        v1 : float
            Velocity after collision.
        v2 : float
            Velocity after collision.

        '''
        v1 = self.__v - (2 * other.__m) / (self.__m + other.__m) *\
            (np.inner(self.__v - other.__v, self.__r - other.__r)) /\
            (np.linalg.norm(self.__r - other.__r)) ** 2 *\
            (self.__r - other.__r)
        v2 = other.__v - (2 * self.__m) / (self.__m + other.__m) * \
            (np.inner(other.__v - self.__v, other.__r - self.__r)) / \
            (np.linalg.norm(other.__r - self.__r)) ** 2 *\
            (other.__r - self.__r)
        return v1, v2

    def get_patch(self):
        return pl.Circle(self.__r, self.__R, fc='white', ec='b')
