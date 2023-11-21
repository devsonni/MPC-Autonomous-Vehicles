import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

T = 0.2
L = 4.45
l_r = 1.7

def shift_fun(T, cont, x0):

    x0[0] = x0[0] + cont[0] * ca.cos(x0[2] + ca.atan2(l_r*ca.tan(cont[1]), L)) * T
    x0[1] = x0[1] + cont[0] * ca.sin(x0[2] + ca.atan2(l_r*ca.tan(cont[1]), L)) * T
    x0[2] = x0[2] + cont[0] * ca.tan(cont[1]) * ca.cos(ca.atan2(l_r*ca.tan(cont[1]), L)) * T / L
    # x0[3] = x0[3] + cont[1] * T

    return x0


x0 = [100, 100, 0]
cont = [14, 0.1]
loop_run = 100
xx = np.zeros((3, loop_run))
xx[:,0] = x0

for i in range(loop_run):
    
    print(i)
    x0 = shift_fun(T, cont, x0)
    print(x0)
    xx[:, i] = x0

# print(xx)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(xx[0, 0:loop_run-1], xx[1, 0:loop_run-1], linewidth = "2", color = "green")
plt.show()