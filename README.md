Thermodynamics Snookered

Project description: 
This Python-based simulation investigates the effects of different parameters on temperature and pressure, allowing for exploration of underlying thermodynamics laws
In this project I created a 2-D container with hard sphere balls inside it that collide elastically using object-oriented programming. There are different methods in the “Comp_Proj_Simulation.py” file that make the plots listed below: 
      - Histogram of ball distance from container centre
      - Histogram of inter-ball separation
      - System kinetic energy versus time
      - Momentum versus time
      - Pressure versus temperature
      - Plots to illustrate how the changing ball radius affects the equation of state
      - Histogram of ball speeds & comparison to theoretical prediction
      - Plot of data used to fit to van der Waal’s law and plot of b against different ball radii
Matplotlib Artist animation is used in this code. Instead of only showing frames where two balls touch, a uniform time between each collision is added for better visualisation to smooth the animation, although it doesn’t reflect the “true” velocity of the balls, it is useful for intuitive understanding and checking of the code. 

How to run the project:
Open the “Investigation.py” file and select the cell that corresponds to the task that you would like to run, and run it, the plots will be save automatically. Note that the first element in the list of balls will be identified as the container. If you have any questions about a method, run “help()” in a cell or console. 
For example:
help(Simulation.van_der_waal)
