# Imports
import pyomo.environ as pyo
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import art3d

# A class for the obstacles
class Obstacle:
    def __init__(self, x, y, z, wx, wy, wz, steps):
        self.x = x
        self.y = y
        self.z = z
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.steps = steps
    # __call__ function for the animation
    def __call__(self, t):
        q, r = np.divmod(t, self.steps)
        if q % 2 == 1:
            r += 1
            r *= -1
        return self.x[r], self.y[r], self.z[r]

# A class for the drone  
class Drone:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    # __call__ function for the animation
    def __call__(self, t):
        if t < len(self.x):
            return self.x[t], self.y[t], self.z[t]
        else:
            return self.x[-1], self.y[-1], self.z[-1]

# A class for the danger zones 
class Zone:
    def __init__(self, x, y, z, wx, wy, wz):
        self.x = x
        self.y = y
        self.z = z
        self.wx = wx
        self.wy = wy
        self.wz = wz

# The create_spline function returns three arrays with the x, y and z coordinates of the points of the spline
def create_spline(nodes, steps):
    x = nodes[:,0]
    y = nodes[:,1]
    z = nodes[:,2]
    tck, _ = interpolate.splprep([x,y,z],s=0)
    x1, y1, z1 = interpolate.splev(np.linspace(0, 1, steps), tck)
    inter_point_differences_x = np.diff(x1)
    inter_point_differences_y = np.diff(y1)
    inter_point_differences_z = np.diff(z1)
    inter_point_distances = np.sqrt(inter_point_differences_x ** 2 + inter_point_differences_y ** 2 + inter_point_differences_z ** 2)
    cumulative_distance = np.cumsum(inter_point_distances)
    cumulative_distance /= cumulative_distance[-1]
    cumulative_distance = np.insert(cumulative_distance, 0, 0)
    tck_prime, _ = interpolate.splprep([np.linspace(0, 1, num=len(cumulative_distance))], u=cumulative_distance, s=0)
    equidistant = interpolate.splev(np.linspace(0, 1, steps), tck_prime)
    x2, y2, z2 = interpolate.splev(equidistant, tck)
    return x2[0], y2[0], z2[0]

# A class for the 3D animation of cubes
class Cube:
    def __init__(self, x, y, z, wx, wy, wz):
        self.x = x
        self.y = y
        self.z = z
        self.wx = wx
        self.wy = wy
        self.wz = wz
        self.faces = [np.array([[x-wx/2,y-wy/2,z+wz/2],[x-wx/2,y+wy/2,z+wz/2],[x+wx/2,y+wy/2,z+wz/2],[x+wx/2,y-wy/2,z+wz/2]]),
                      np.array([[x+wx/2,y-wy/2,z+wz/2],[x+wx/2,y+wy/2,z+wz/2],[x+wx/2,y+wy/2,z-wz/2],[x+wx/2,y-wy/2,z-wz/2]]),
                      np.array([[x+wx/2,y+wy/2,z+wz/2],[x-wx/2,y+wy/2,z+wz/2],[x-wx/2,y+wy/2,z-wz/2],[x+wx/2,y+wy/2,z-wz/2]]),
                      np.array([[x-wx/2,y+wy/2,z+wz/2],[x-wx/2,y-wy/2,z+wz/2],[x-wx/2,y-wy/2,z-wz/2],[x-wx/2,y+wy/2,z-wz/2]]),
                      np.array([[x-wx/2,y-wy/2,z+wz/2],[x+wx/2,y-wy/2,z+wz/2],[x+wx/2,y-wy/2,z-wz/2],[x-wx/2,y-wy/2,z-wz/2]]),
                      np.array([[x+wx/2,y-wy/2,z-wz/2],[x+wx/2,y+wy/2,z-wz/2],[x-wx/2,y+wy/2,z-wz/2],[x-wx/2,y-wy/2,z-wz/2]]),]
    # The change_center function returns the faces according to the new centre point
    def change_center(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.faces = [np.array([[x-self.wx/2,y-self.wy/2,z+self.wz/2],[x-self.wx/2,y+self.wy/2,z+self.wz/2],[x+self.wx/2,y+self.wy/2,z+self.wz/2],[x+self.wx/2,y-self.wy/2,z+self.wz/2]]),
                      np.array([[x+self.wx/2,y-self.wy/2,z+self.wz/2],[x+self.wx/2,y+self.wy/2,z+self.wz/2],[x+self.wx/2,y+self.wy/2,z-self.wz/2],[x+self.wx/2,y-self.wy/2,z-self.wz/2]]),
                      np.array([[x+self.wx/2,y+self.wy/2,z+self.wz/2],[x-self.wx/2,y+self.wy/2,z+self.wz/2],[x-self.wx/2,y+self.wy/2,z-self.wz/2],[x+self.wx/2,y+self.wy/2,z-self.wz/2]]),
                      np.array([[x-self.wx/2,y+self.wy/2,z+self.wz/2],[x-self.wx/2,y-self.wy/2,z+self.wz/2],[x-self.wx/2,y-self.wy/2,z-self.wz/2],[x-self.wx/2,y+self.wy/2,z-self.wz/2]]),
                      np.array([[x-self.wx/2,y-self.wy/2,z+self.wz/2],[x+self.wx/2,y-self.wy/2,z+self.wz/2],[x+self.wx/2,y-self.wy/2,z-self.wz/2],[x-self.wx/2,y-self.wy/2,z-self.wz/2]]),
                      np.array([[x+self.wx/2,y-self.wy/2,z-self.wz/2],[x+self.wx/2,y+self.wy/2,z-self.wz/2],[x-self.wx/2,y+self.wy/2,z-self.wz/2],[x-self.wx/2,y-self.wy/2,z-self.wz/2]]),]
        return self.faces

# The animate_scene function creates an animation of the drones, obstacles and zones
def animate_scene(obstacles, zones, drones, N, x_min, x_max, y_min, y_max, z_min, z_max):
    fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_aspect("equal")
    # Creating the obstacles as blue Cubes
    obs_art = []
    cubes_obs = []
    for i in range(0, len(obstacles)):
        x = obstacles[i].x[0]
        y = obstacles[i].y[0]
        z = obstacles[i].z[0]
        wx = obstacles[i].wx
        wy = obstacles[i].wy
        wz = obstacles[i].wz
        cubes_obs.append(Cube(x,y,z,wx,wy,wz))
        obs_art.append(art3d.Poly3DCollection(cubes_obs[i].faces, edgecolor="black", facecolor=None))
        ax.add_collection3d(obs_art[i])
    # Creating the danger zones as green Cubes
    zone_art = []
    cubes_zone = []
    for i in range(0,len(zones)):
        x = zones[i].x
        y = zones[i].y
        z = zones[i].z
        wx = zones[i].wx
        wy = zones[i].wy
        wz = zones[i].wz
        cubes_zone.append(Cube(x,y,z,wx,wy,wz))
        zone_art.append(art3d.Poly3DCollection(cubes_zone[i].faces, edgecolor="black", facecolor="green"))
        ax.add_collection3d(zone_art[i])
    # Creating the drones as a red Cubes
    drone_cube = []
    drone_art = []
    for i in range(0,len(drones)):
        drone_cube.append(Cube(drones[i].x[0], drones[i].y[0], drones[i].z[0], drones[i].w, drones[i].w, drones[i].w))
        drone_art.append(art3d.Poly3DCollection(drone_cube[i].faces, edgecolor="red", facecolor="red"))
        ax.add_collection3d(drone_art[i])
    # Plotting the trajectory of the drone
    for i in range(0,len(drones)):
        ax.plot(drones[i].x, drones[i].y, drones[i].z, color="red")
    tickrate = 30 # ms
    # The animator subfunction
    def animator(k):
        # Moving the obstacles to the accurate positions at each time step
        for i in range(0, len(obstacles)):
            x, y, z = obstacles[i](k)
            obs_art[i].set_verts(cubes_obs[i].change_center(x,y,z))
        # Moving the drone to the accurate position at each time step
        for i in range(0,len(drones)):
            x_drone, y_drone, z_drone = drones[i](k)
            drone_art[i].set_verts(drone_cube[i].change_center(x_drone, y_drone, z_drone))
    anim = animation.FuncAnimation(
        fig, animator, frames=2*(N+1), interval=tickrate, blit=False)
    plt.show()
    return anim

# Creating and solving the Pyomo model
def solve_model(N, x_min, x_max, y_min, y_max, z_min, z_max, v_max, a_max, obstacles, zones, drone_w, drones, x_start, y_start, z_start, x_goal, y_goal, z_goal):
    model = pyo.ConcreteModel()
    # Creating a set for the time steps
    model.steps = pyo.RangeSet(0, N)
    # The sampling time (dt) and its square (dt2)
    model.dt = pyo.Var(domain=pyo.PositiveReals, bounds=(0.01,1))
    model.dt2 = pyo.Var(domain=pyo.PositiveReals, bounds=(0.0001,1))
    # The square is approximated with a piecewise linear function
    time_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    time_y = [0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1]
    model.time = pyo.Piecewise(model.dt2, model.dt,
                               pw_pts=time_x,
                               pw_constr_type="EQ",
                               f_rule=time_y,
                               pw_repn="SOS2")
    # Setting up the bounds for the position coordinates, velocity and acceleration components
    v_lim = np.sqrt(v_max ** 2 / 3)
    a_lim = np.sqrt(a_max ** 2 / 3)
    def x_bounds(model, i, j, k):
        if i == 0:
            return (x_min,x_max)
        elif i == 1:
            return (y_min,y_max)
        elif i == 2:
            return (z_min,z_max)
        else:
            return (-v_lim,v_lim)
    # The variable x includes the (x,y,z) position coordinates and the (v_x,v_y,v_z) velocity components at each time step
    model.x = pyo.Var(range(0,6), model.steps, range(0,drones), domain=pyo.Reals, bounds=x_bounds)
    # The variable u includes the (a_x,a_y,a_z) acceleration components at each time step
    model.u = pyo.Var(range(0,3), model.steps, range(0,drones), domain=pyo.Reals, bounds=(-a_lim,a_lim))
    # Binary variables for avoiding obstacles
    model.x_less = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.x_more = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.y_less = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.y_more = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.z_less = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.z_more = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    # Binary variables for avoiding danger zones
    model.zone_x_less = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_x_more = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_y_less = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_y_more = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_z_less = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_z_more = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    #Binary variables for avoiding other drones
    model.drone_x_less = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_x_more = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_y_less = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_y_more = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_z_less = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_z_more = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    # Constraints that describe the dynamics of the drones
    @model.Constraint(range(0,N), range(0,drones))
    def SSMx(model, i, j):
        return model.x[0,i+1,j] == model.x[0,i,j] + model.dt * model.x[3,i,j] + 0.5 * model.dt2 * model.u[0,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMy(model, i, j):
        return model.x[1,i+1,j] == model.x[1,i,j] + model.dt * model.x[4,i,j] + 0.5 * model.dt2 * model.u[1,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMz(model, i, j):
        return model.x[2,i+1,j] == model.x[2,i,j] + model.dt * model.x[5,i,j] + 0.5 * model.dt2 * model.u[2,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMvx(model, i, j):
        return model.x[3,i+1,j] == model.x[3,i,j] + model.dt * model.u[0,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMvy(model, i, j):
        return model.x[4,i+1,j] == model.x[4,i,j] + model.dt * model.u[1,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMvz(model, i, j):
        return model.x[5,i+1,j] == model.x[5,i,j] + model.dt * model.u[2,i,j]
    # Constraint for avoiding obstacles and danger zones created with the big-M method
    M = np.abs(x_max - x_min + y_max - y_min + z_max - z_min)
    # These constraints guarantee that the drones avoid each obstacle at every time step
    model.x_less_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] <= obstacles[i].x[j]-(obstacles[i].wx/2)-drone_w/2 + M * model.x_less[i,j,k])
    model.x_more_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] >= obstacles[i].x[j]+(obstacles[i].wx/2)+drone_w/2 - M * model.x_more[i,j,k])
    model.y_less_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] <= obstacles[i].y[j]-(obstacles[i].wy/2)-drone_w/2 + M * model.y_less[i,j,k])
    model.y_more_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] >= obstacles[i].y[j]+(obstacles[i].wy/2)+drone_w/2 - M * model.y_more[i,j,k])
    model.z_less_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[2,j,k] <= obstacles[i].z[j]-(obstacles[i].wz/2)-drone_w/2 + M * model.z_less[i,j,k])
    model.z_more_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[2,j,k] >= obstacles[i].z[j]+(obstacles[i].wz/2)+drone_w/2 - M * model.z_more[i,j,k])
    model.avoid = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x_less[i,j,k] + model.x_more[i,j,k] + model.y_less[i,j,k] + model.y_more[i,j,k] + model.z_less[i,j,k] + model.z_more[i,j,k] <= 5)
    # These constraints inform us if any danger zone contain the drones at any time step
    model.zone_x_less_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] <= zones[i].x-(zones[i].wx/2)-drone_w/2 + M * model.zone_x_less[i,j,k])
    model.zone_x_more_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] >= zones[i].x+(zones[i].wx/2)+drone_w/2 - M * model.zone_x_more[i,j,k])
    model.zone_y_less_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] <= zones[i].y-(zones[i].wy/2)-drone_w/2 + M * model.zone_y_less[i,j,k])
    model.zone_y_more_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] >= zones[i].y+(zones[i].wy/2)+drone_w/2 - M * model.zone_y_more[i,j,k])
    model.zone_z_less_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[2,j,k] <= zones[i].z-(zones[i].wz/2)-drone_w/2 + M * model.zone_z_less[i,j,k])
    model.zone_z_more_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[2,j,k] >= zones[i].z+(zones[i].wz/2)+drone_w/2 - M * model.zone_z_more[i,j,k])
    model.zone_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.zone_x_less[i,j,k]+model.zone_x_more[i,j,k]+model.zone_y_less[i,j,k]+model.zone_y_more[i,j,k]+model.zone_z_less[i,j,k]+model.zone_z_more[i,j,k]-model.zone[i,j,k] <= 5)
    # Avoiding collisions between drones
    def x_less(model, i,j,k):
        if i<j:
            return model.x[0,k,i] <= model.x[0,k,j]-drone_w + M * model.drone_x_less[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_x_less_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=x_less)
    def x_more(model, i,j,k):
        if i<j:
            return model.x[0,k,i] >= model.x[0,k,j]+drone_w - M * model.drone_x_more[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_x_more_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=x_more)
    def y_less(model, i,j,k):
        if i<j:
            return model.x[1,k,i] <= model.x[1,k,j]-drone_w + M * model.drone_y_less[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_y_less_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=y_less)
    def y_more(model, i,j,k):
        if i<j:
            return model.x[1,k,i] >= model.x[1,k,j]+drone_w - M * model.drone_y_more[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_y_more_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=y_more)
    def z_less(model, i,j,k):
        if i<j:
            return model.x[2,k,i] <= model.x[2,k,j]-drone_w + M * model.drone_z_less[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_z_less_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=z_less)
    def z_more(model, i,j,k):
        if i<j:
            return model.x[2,k,i] >= model.x[2,k,j]+drone_w - M * model.drone_z_more[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_z_more_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=z_more)
    def con(model,i,j,k):
        if i<j:
            return model.drone_x_less[i,j,k] + model.drone_x_more[i,j,k] + model.drone_y_less[i,j,k] + model.drone_y_more[i,j,k] + model.drone_z_less[i,j,k] + model.drone_z_more[i,j,k] <= 5
        else:
            return pyo.Constraint.Skip
    model.drone_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=con)
    # A constraint list that includes the initial and final conditions
    model.conditions = pyo.ConstraintList()
    for i in range(0, drones):
        model.conditions.add(model.x[0,0,i] == x_start[i])
        model.conditions.add(model.x[1,0,i] == y_start[i])
        model.conditions.add(model.x[2,0,i] == z_start[i])
        model.conditions.add(model.x[3,0,i] == 0)
        model.conditions.add(model.x[4,0,i] == 0)
        model.conditions.add(model.x[5,0,i] == 0)
        model.conditions.add(model.x[0,N,i] == x_goal[i])
        model.conditions.add(model.x[1,N,i] == y_goal[i])
        model.conditions.add(model.x[2,N,i] == z_goal[i])
        model.conditions.add(model.x[3,N,i] == 0)
        model.conditions.add(model.x[4,N,i] == 0)
        model.conditions.add(model.x[5,N,i] == 0)

    # The objective is to minimize the sampling time
    # The objective includes and extra term, that serves as a penalty for spending time in danger zones
    alpha = 10
    obj_expr = model.dt + sum(model.zone[i,j,k]*alpha for i in range(len(zones)) for j in model.steps for k in range(0,drones))
    model.obj = pyo.Objective(sense=pyo.minimize, expr=obj_expr)
    solver = pyo.SolverFactory("gurobi_direct")
    results = solver.solve(model, options={"NonConvex": 2}, tee=True)
    # If the model is solved successfully, the function returns three lists containing the x, y and z coordinates of the trajectories at each time step and the calculated sampling time
    # If the model is not solved (e.g. the model is infeasible), the function returns the solver status and the termination condition
    if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        for i in range(0,drones):
            x_result = []
            y_result = []
            z_result = []
            for j in model.steps:
                x_result.append(pyo.value(model.x[0,j,i]))
                y_result.append(pyo.value(model.x[1,j,i]))
                z_result.append(pyo.value(model.x[2,j,i]))
            trajectory_x.append(x_result)
            trajectory_y.append(y_result)
            trajectory_z.append(z_result)
        return trajectory_x, trajectory_y, trajectory_z, pyo.value(model.dt)
    else:
        return -1, -1, -1, -1

# Main
def main():
    # The parameters can be specified here
    # The number of time steps:
    N = 100
    # The available space for the movement of the drones:
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    z_min, z_max = 0, 10
    # The maximum velocity of the drones:
    v_max = 4
    # The maximum acceleration of the drones:
    a_max = 4
    # The list of obstacles
    # Obstacle can be created using the Obstacle class
    obstacles = []
    nodes = np.array([[1, 9, 1.5],
                      [4, 7, 3],
                      [2.5, 5, 4],
                      [5, 2, 5.5],
                      [8, 4, 7],
                      [6, 6, 8.5]])
    x_spline, y_spline, z_spline = create_spline(nodes, N+1)
    obstacles.append(Obstacle(x_spline, y_spline, z_spline, 2, 2, 2, N+1))
    obstacles.append(Obstacle(np.full(N+1,3.5), np.linspace(5,1,N+1), np.linspace(5,1,N+1), 2, 2, 2, N+1))
    obstacles.append(Obstacle(np.full(N+1,1.5), np.full(N+1,1), np.full(N+1,1), 1, 2, 2, N+1))
    obstacles.append(Obstacle(np.full(N+1,9), np.full(N+1,8), np.full(N+1,8), 1, 4, 4, N+1))
    # The list of danger zones
    # Danger zones can be created using the Zone class
    zones = []
    zones.append(Zone(7, 5, 5, 2, 2, 2))
    # The width of the drones
    drone_radius = 0.2
    # The number of drones
    drones_number = 2
    # The coordinates of the starting position of the drones
    x_start, y_start, z_start = [0,10], [0,0], [0,0]
    # The coordinates of the destination of the drones
    x_goal, y_goal, z_goal = [10,5], [10,10], [10,10]
    # Solving the model
    x, y, z, dt = solve_model(N, x_min, x_max, y_min, y_max, z_min, z_max, v_max, a_max, obstacles, zones, drone_radius, drones_number, x_start, y_start, z_start, x_goal, y_goal, z_goal)
    # If the model was solved successfully:
    #   - The total travel time is printed
    #   - An animation shows the drones reaching their destinations
    if x != -1 and y != -1 and z != -1:
        time = N * dt
        print(f"Total travel time: {round(time,3)} seconds")
        drones = []
        for i in range(0,drones_number):
            drones.append(Drone(x[i], y[i], z[i], drone_radius))
        animate_scene(obstacles, zones, drones, N, x_min, x_max, y_min, y_max, z_min, z_max)
    else:
        print("The problem was infeasible.")

if __name__ == "__main__":
    main()