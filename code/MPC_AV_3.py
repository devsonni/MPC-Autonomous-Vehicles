import casadi as ca
from casadi import sin, cos, pi, tan, atan2, atan
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## some constants
l_r = 17 
l_f = 13
L = 45
h_r = 0.3

global con_t
con_t = [14, 0.02]   # Linear and angular velocity of target

# Shift function 
def shift_timestep(T, t0, x0, u, f_u, xs):
    print(u[:, 0])
    # u[:, 0] = [14, 0.0007]
    f_value = f_u(x0, u[:, 0])
    #print(f_value)
    x0 = ca.DM.full(x0 + (T * f_value))

    t0 = t0 + T
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

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
acc_min = -3
acc_max = 3
delta_min = -pi/6
delta_max = pi/6

# state constraints
vel_min = 0
vel_max = 33.3
omega_min = -3
omega_max = 3
yaw_min = -pi/3
yaw_max = pi/3
lat_acc_min = -ca.inf
lat_acc_max = (0.8*L*9.8)/(2*h_r)

## symbolic states of the system
# states of the AV
x = ca.SX.sym('x')           # x coordinate of the AV
y = ca.SX.sym('y')           # y coordinate of the AV
psi = ca.SX.sym('psi')       # psi (orientation of the AV)
beta = ca.SX.sym('beta')     # slip angle of the AV
v = ca.SX.sym('v')           # liner velocity of the AV
a_y = ca.SX.sym('a_y')       # lateral acceleration

omega = ca.SX.sym('omega')  
delta = ca.SX.sym('delta')   # heading angle of the vehicle

# state vector of the sys
states_u = ca.vertcat(
    x,
    y,
    psi,
    delta,
)
n_states_u = states_u.numel()


# control input
a = ca.SX.sym('a')           # acceleration of the liner velocity


# control vector of the sys
controls_u = ca.vertcat(
    v,
    omega
)
n_controls = controls_u.numel()


# RHS of the euler discretization 
rhs_u = ca.vertcat(
    v*cos(psi),
    v*sin(psi),
    v*tan(delta)/L,
    omega,
)

# Non-linear mapping function which is f(x,y)
f_u = ca.Function('f', [states_u, controls_u], [rhs_u])

U = ca.SX.sym('U', n_controls, N)           # Decision Variables
P = ca.SX.sym('P', n_states_u + 3 + 2)      # This consists of initial states of AV and last two states 
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

for k in range(N):
    stt = X[0:n_states_u, 0:N]
    w = 10
    obj = obj + (P[4] - X[0, k]) ** 2 + (P[5] - X[1, k]) ** 2 + (P[6] - X[3, k]) ** 2 


g = []
# compute the states or inequality constrains
for k in range(N+1):
    g = ca.vertcat(g,
    X[2, k],  # limit on yaw
    X[3, k],  # lim on delta
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

K = 2

lbx = ca.DM.zeros((n_controls*N, 1))   # constraints on the controls 
ubx = ca.DM.zeros((n_controls*N, 1))
lbg = ca.DM.zeros((K*(N+1)))  # constraints on the states
ubg = ca.DM.zeros((K*(N+1)))

# constraints on the states (Inequality constrains)       
lbg[0:K*(N+1):K] = yaw_min  
lbg[1:K*(N+1):K] = delta_min

ubg[0:K*(N+1):k] = yaw_max    
ubg[1:K*(N+1):k] = delta_max

# constraint on controls, constraint on optimization variable
lbx[0: n_controls*N: n_controls, 0] = vel_min       # acc lower bound
lbx[1: n_controls*N: n_controls, 0] = omega_min     # theta lower bound

ubx[0: n_controls*N: n_controls, 0] = vel_max       # acc upper bound
ubx[1: n_controls*N: n_controls, 0] = omega_max     # theta upper bound

args = {
    'lbg': lbg,  # lower bound for state
    'ubg': ubg,  # upper bound for state
    'lbx': lbx,  # lower bound for controls
    'ubx': ubx   # upper bound for controls
}

# target
x_target = 0
y_target = 0
theta_target = 0

t0 = 0
x0 = [0, 0, 0, 0]
# x0 = [100, 150, 0]
xs = ca.DM(ca.vertcat(x_target,
                      y_target,
                      theta_target))  # initial target state

# xx = DM(state_init)
t = ca.DM(t0)
loop_run = 500

u0 = ca.DM.zeros((n_controls, N))          # initial control
xx = ca.DM.zeros((n_states_u,loop_run+1))  # change size according to main loop run,
# works as tracker for target and UAV
ss = ca.DM.zeros((3,loop_run+1))

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
            xs,   # target state
            con_t
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
        # ff_value = ff(u, args['p'])

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

        # for plotting
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


################################################################################
################################# Plotting Stuff ###############################

# print(cat_controls)
# Plotting starts from here
xx1 = np.array(xx)
xs1 = np.array(ss)
ss1 = np.zeros((loop_run+1))

# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot3D(xx1[0,0:loop_run-1], xx1[1, 0:loop_run-1], ss1[0:loop_run-1], linewidth = "2", color = "green")
# ax.plot3D(xs1[0,0:loop_run-1], xs1[1,0:loop_run-1], ss1[0:loop_run-1], ls="--", linewidth = "2", color = "red")
# ax.set_xlim(0, 300)
# ax.set_ylim(-100, 100)
# ax.set_zlim(0, 0)

# ax.set_title('AV')

# Calculating error between UAV and FOV
error = ca.DM.zeros(loop_run+1)
for i in range(loop_run):
    error[i] = ca.sqrt((xx1[0,i+1]-xs1[0,i])**2 + (xx1[1,i+1]-xs1[1,i])**2)

itr = loop_run
error1 = np.array(error)
sum_err = sum(error1[0:itr])
print(sum(error1[0:itr]))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(xx1[0, 0:loop_run-1], xx1[1, 0:loop_run-1], linewidth = "2", color = "green")
plt.plot(xs1[0,0:loop_run-1], xs1[1,0:loop_run-1], ls="--", linewidth = "2", color = "red")
# ax.set_xlim(-30, 300)
# ax.set_ylim(-100, 100)    
plt.title('AV')


fig = plt.figure()
plt.plot(cat_controls[0:n_controls*loop_run:n_controls], linewidth= "2", color = "red")
plt.plot(cat_controls[1:n_controls*loop_run:n_controls], linewidth= "2", color = "blue")
# plt.plot(xx1[3,0:loop_run-1], linewidth= "2", color = "green")
plt.legend(['velocity', 'steering-speed'])
plt.xlabel("Iteration")
plt.ylabel("controls")

fig = plt.figure()
plt.subplot(121)
plt.plot(error[0:itr], linewidth= "2", color = "red")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.subplot(122)
plt.bar('Error' , sum_err, color="red")

plt.show()