import casadi as ca
from casadi import sin, cos, pi, tan, atan2, atan
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## some constants
l_r = 1.016
l_f = 2.54 - 1.016
l_w = 1.73
L = 2.54
h_r = 0.6 
G = 9.8

# Shift function 
def shift_timestep(T, t0, x0, u, f_u, xs):
    print("controls " + str(u[:, 0]))
    f_value = f_u(x0, u[:, 0])
    #print(f_value)
    x0 = ca.DM.full(x0 + (T * f_value))

    t0 = t0 + T
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )
    v = 30
    con_t = [v, 0.15]   # Linear and angular velocity of target

    f_t_value = ca.vertcat(con_t[0] * cos(xs[2]),
                           con_t[0] * sin(xs[2]),
                           con_t[1])
    #print(xs)
    xs = ca.DM.full(xs + (T * f_t_value))
    return t0, x0, u0, xs

def DM2Arr(dm):
    return np.array(dm.full())

def SX2Arr(sx):
    return np.array(sx.full())


# mpc parameters
T = 0.2             # discretization steps
N = 15              # prediction horizon


# system model (kinematic bicycle model)
## constraints of the system
# input constraints
acc_min = -100
acc_max = 100
delta_min = -pi/3
delta_max = pi/3

# state constraints
vel_min = 0
vel_max = 33.3
omega_min = -pi/3
omega_max = pi/3


## symbolic states of the system
# states of the AV
x = ca.SX.sym('x')           # x coordinate of the AV
y = ca.SX.sym('y')           # y coordinate of the AV
psi = ca.SX.sym('psi')       # psi (orientation of the AV)
beta = ca.SX.sym('beta')     # slip angle of the AV
v = ca.SX.sym('v')           # liner velocity of the AV
a_y = ca.SX.sym('a_y')       # lateral acceleration
delta = ca.SX.sym('delta')
omega = ca.SX.sym('omega')

# state vector of the sys
states_u = ca.vertcat(
    x,
    y,
    psi,
    v,
    delta
)
n_states_u = states_u.numel()


# control input
a = ca.SX.sym('a')           # acceleration of the liner velocity

# control vector of the sys
controls_u = ca.vertcat(
    a,
    omega
)
n_controls = controls_u.numel()


# RHS of the euler discretization 
rhs_u = ca.vertcat(
    v*cos(psi+atan2(l_r*ca.tan(delta), L)),
    v*sin(psi+atan2(l_r*ca.tan(delta), L)),
    v*tan(delta)*cos(atan2(l_r*ca.tan(delta), L))/L,
    a,
    omega
)  

# Non-linear mapping function which is f(x,y)
f_u = ca.Function('f', [states_u, controls_u], [rhs_u])

U = ca.SX.sym('U', n_controls, N)       # Decision Variables
P = ca.SX.sym('P', n_states_u + 3)      # This consists of initial states of AV and last two states 
# are for reference trajectory

X = ca.SX.sym('X', n_states_u, (N+1))    # This will consists prediction of states over N

# filling sybolic variavles in the prediction
X[:, 0] = P[0:n_states_u]       # initial state

# prediction states
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    f_value = f_u(st, con)
    st_next = st + T*f_value
    X[:, k+1] = st_next

ff = ca.Function('ff', [U, P], [X])

# Objective function
obj = 0  # objective function that need to minimize over control variables
w = 1
for k in range(N):
    obj = obj + w*(ca.sqrt((X[0, k] - P[n_states_u]) ** 2 + (X[1, k] - P[n_states_u+1]) ** 2))


g = []
# compute the states or inequality constrains
for k in range(N+1):
    g = ca.vertcat(g,
    X[3, k],
    X[4, k], 
    (180*(atan2(((X[3, k]**2)*ca.sin(atan2(l_r*ca.tan(X[4, k]), L))*ca.tan(X[4, k])), (l_r*G))))/ca.pi,
    (2*0.3*(X[3, k]**2)*ca.sin(atan2(l_r*ca.tan(X[4, k]), L)))/(l_r*l_w*G),
)

# make the decision variables one column vector
OPT_variables = \
    U.reshape((-1, 1))    # Example: 6x15 ---> 90x1 where 6=controls, 16=n+1

nlp_prob = {
    'f': obj,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

D = 4

lbx = ca.DM.zeros((n_controls*N, 1))   # constraints on the controls 
ubx = ca.DM.zeros((n_controls*N, 1))
lbg = ca.DM.zeros((D*(N+1)))  # constraints on the states
ubg = ca.DM.zeros((D*(N+1)))

# constraints on the states (Inequality constrains)       
lbg[0:D*(N+1):D] = vel_min  
lbg[1:D*(N+1):D] = delta_min 
lbg[2:D*(N+1):D] = -180*atan2(h_r, L)/ca.pi
lbg[3:D*(N+1):D] = -0.5

ubg[0:D*(N+1):D] = vel_max    
ubg[1:D*(N+1):D] = delta_max 
ubg[2:D*(N+1):D] = 180*atan2(h_r, L)/ca.pi
ubg[3:D*(N+1):D] = 0.5

# constraint on controls, constraint on optimization variable
lbx[0: n_controls*N: n_controls, 0] = acc_min       # acc lower bound
lbx[1: n_controls*N: n_controls, 0] = omega_min     # theta lower bound

ubx[0: n_controls*N: n_controls, 0] = acc_max       # acc upper bound
ubx[1: n_controls*N: n_controls, 0] = omega_max     # theta upper bound

args = {
    'lbg': lbg,  # lower bound for state
    'ubg': ubg,  # upper bound for state
    'lbx': lbx,  # lower bound for controls
    'ubx': ubx   # upper bound for controls
}

# target
x_target = 100
y_target = 100
theta_target = 0

t0 = 0
x0 = [100, 100, 0, 0, 0]
xs = ca.DM(ca.vertcat(x_target,
                      y_target,
                      theta_target))  # initial target state

# xx = DM(state_init)
t = ca.DM(t0)
loop_run = 100

u0 = ca.DM.zeros((n_controls, N))          # initial control
xx = ca.DM.zeros((n_states_u,loop_run+1))                  # change size according to main loop run, works as tracker for target and UAV
ss = ca.DM.zeros((3,loop_run+1))
beta_arr = np.zeros((loop_run+1)) 
theta_thresh_arr = np.zeros((loop_run+1)) 
theta_roll = np.zeros((loop_run+1)) 
R_index = np.zeros((loop_run+1)) 

xx[:,0] = x0
ss[:,0] = xs

mpc_iter = 0
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])

###############################################################################
##############################   Main Loop    #################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while mpc_iter < loop_run:
        t1 = time()
        args['p'] = ca.vertcat(
            x0,  # current state
            xs   # target state
        )

        print(args['p'])
        # optimization variable current state
        args['x0'] = \
            ca.reshape(u0, n_controls*N, 1)

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'], n_controls, N)
        ff_value = ff(u, args['p'])

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))

        t = np.vstack((
            t,
            t0
        ))

        t0, x0, u0, xs = shift_timestep(T, t0, x0, u, f_u, xs)
        print(x0)

        # tracking states of target and UAV for plotting
        xx[:, mpc_iter+1] = x0
        ss[:, mpc_iter+1] = xs

        # xx ...
        t2 = time()
        print(mpc_iter)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

        # xx = DM2Arr(xx)

        beta_arr[mpc_iter] =  180*atan2(l_r*ca.tan(xx[4, mpc_iter]), L)/ca.pi
        theta_thresh_arr[mpc_iter] = 180*atan2(h_r, L)/ca.pi
        # print(atan2(((xx[3, mpc_iter]**2)*ca.sin(atan2(l_r*ca.tan(xx[4, mpc_iter]), L))*ca.tan(xx[4, mpc_iter])), (l_r*G)))
        theta_roll[mpc_iter] = 180*atan2(((xx[3, mpc_iter]**2)*ca.sin(atan2(l_r*ca.tan(xx[4, mpc_iter]), L))*ca.tan(xx[4, mpc_iter])), (l_r*G))/ca.pi
        R_index[mpc_iter] = (2*0.3*(xx[3, mpc_iter]**2)*ca.sin(atan2(l_r*ca.tan(xx[4, mpc_iter]), L)))/(l_r*l_w*G)

################################################################################
################################# Plotting Stuff ###############################

# print(cat_controls)
# Plotting starts from here
xx1 = np.array(xx)
xs1 = np.array(ss)
ss1 = np.zeros((loop_run+1))

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
ax = plt.axes(projection = '3d')
ax.plot3D(xx1[0,0:loop_run-1], xx1[1, 0:loop_run-1], ss1[0:loop_run-1], linewidth = "2", color = "green")
ax.plot3D(xs1[0,0:loop_run-1], xs1[1,0:loop_run-1], ss1[0:loop_run-1], ls="--", linewidth = "2", color = "red")


ax.set_title('AV')

# Calculating error between UAV and FOV
error = ca.DM.zeros(loop_run+1)
for i in range(loop_run):
    error[i] = ca.sqrt((xx1[0,i+1]-xs1[0,i])**2 + (xx1[1,i+1]-xs1[1,i])**2)

itr = loop_run
error1 = np.array(error)
sum_err = sum(error1[0:itr])
print(sum(error1[0:itr]))

fig = plt.figure()
plt.rcParams.update({'font.size': 20})
plt.plot(cat_controls[0:n_controls*loop_run:n_controls], linewidth= "2", color = "red")
plt.title("Acceleration")
plt.xlabel("Iterations")
plt.ylabel("m/s2")

fig = plt.figure()
plt.rcParams.update({'font.size': 20})
plt.plot(cat_controls[1:n_controls*loop_run:n_controls], linewidth= "2", color = "blue")
plt.title("Steering-Speed")
plt.xlabel("Iterations")
plt.ylabel("m/s")

fig = plt.figure()
plt.rcParams.update({'font.size': 20})
plt.plot(xx1[3,0:loop_run], linewidth= "2", color = "green")
plt.title("Velocity")
plt.xlabel("Iterations")
plt.ylabel("m/s")

fig = plt.figure()
plt.rcParams.update({'font.size': 20})
plt.plot(xx1[4,0:loop_run], linewidth= "2", color = "purple")
plt.title("Steering Angle")
# plt.legend(['acceleration', 'steering speed', 'velocity', 'steering angle'])
plt.xlabel("Iterations")
plt.ylabel("rad")

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
ax = fig.add_subplot(1, 1, 1)
plt.plot(xx1[0, 0:loop_run], xx1[1, 0:loop_run], linewidth = "4", color = "green")
plt.plot(xs1[0,0:loop_run], xs1[1,0:loop_run], ls="--", linewidth = "4", color = "red")
plt.legend(['AV Trajectory', 'Reference'])
plt.xlabel("x-axis")
plt.ylabel("y-axis")

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(beta_arr[0:loop_run], linewidth="2", color="blue")
plt.title("Slip Angle")
plt.xlabel("Iterations")
plt.ylabel("degree")

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(theta_thresh_arr[0:loop_run], linewidth="2", color="black")
plt.plot(theta_roll[0:loop_run], linewidth="2", color="blue")
plt.title("Rollover Angle")
plt.xlabel("Iterations")
plt.ylabel("degree")

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.plot(R_index[0:loop_run], linewidth="2", color="red")
plt.title("Rollover Index")
plt.xlabel("Iterations")
plt.ylabel("Index")

fig = plt.figure()
plt.rcParams.update({'font.size': 30})
plt.subplot(121)
plt.plot(error[0:itr], linewidth= "2", color = "red")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.subplot(122)
plt.bar('Error' , sum_err, color="red")

plt.show()