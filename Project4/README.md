# Reinforcment Learning

## Overview

Run *project4.py* with lines 118-121 commented/uncommented depending on desired functionality. Each parts main code is in a function in *project4.py* which then calls a funcion in *solutions.py* to do the training.

## Policy Iteration

Run problem_1 function with variables specifying the size of grid and whether it is slippery or not. Training occurs using policy iteration function in solutions file, which iteratively calls policy evaluation and updates q values accordingly before returning values after convergence. Can test the current optima by setting test=True.

## Q Learning

Run q learning function in *project4.py* and pass in respective hyperparameters to function call in line 60, otherwise specified values will be used according to line 80 of *solutions.py*. This will train Q values according to specified hyperparameters and plot the results.

## Acrobot

Train or test acrobot environment by calling problem_3b in *project4.py*. See the code in *solutions.py* for implementation details. Maps to single state index based on environment state then uses Q learning process similar to part 2 but with some nuances described in report. Included *acrobot.mp4* file shows performance after training.

## Mountain Car

Very similar to Acrobot in approach. See included *mountain_car.mp4* video showing performance on learned policy.