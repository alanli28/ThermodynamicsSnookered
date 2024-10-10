# Thermodynamics Snookered

A Python-based simulation investigating the effects of different parameters on temperature and pressure, allowing for exploration of underlying thermodynamic laws.

## Overview

In this project, a 2-D container with hard sphere balls is simulated, where the balls collide elastically using object-oriented programming. Various methods in the `Comp_Proj_Simulation.py` file generate the following plots:

- **Histogram of Ball Distance from Container Centre**
- **Histogram of Inter-Ball Separation**
- **System Kinetic Energy vs. Time**
- **Momentum vs. Time**
- **Pressure vs. Temperature**
- **Effect of Changing Ball Radius on the Equation of State**
- **Histogram of Ball Speeds & Comparison to Theoretical Prediction**
- **Van der Waals Law Fitting and Analysis**

Matplotlib's `ArtistAnimation` is utilized to create smooth animations by adding uniform time intervals between collisions. While this doesn't reflect the "true" velocities of the balls, it enhances visualization and aids intuitive understanding and code verification.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries:
  - `matplotlib`
  - `numpy`

### Project Report

For a detailed explanation of the methodology, results, and analysis, please refer to the Project Report.

### How to run the project:
Open the Investigation.py file and select the cell that corresponds to the task that you would like to run, and run it. The plots will be saved automatically.

**Note**: The first element in the list of balls will be identified as the container. If you have any questions about a method, run help() in a cell or console.

For example:

```bash
help(Simulation.van_der_waal)```
