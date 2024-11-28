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

# The animate_scene function creates an animation of the drone, obstacles and zones
def animate_scene(obstacles, zones, drone, N, x_min, x_max, y_min, y_max, z_min, z_max):
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
    # Creating the drone as a red Cube
    drone_cube = Cube(drone.x[0], drone.y[0], drone.z[0], drone.w, drone.w, drone.w)
    drone_art = art3d.Poly3DCollection(drone_cube.faces, edgecolor="red", facecolor="red")
    ax.add_collection3d(drone_art)
    # Plotting the trajectory of the drone
    ax.plot(drone.x, drone.y, drone.z, color="red")
    tickrate = 30 # ms
    # The animator subfunction
    def animator(k):
        # Moving the obstacles to the accurate positions at each time step
        for i in range(0, len(obstacles)):
            x, y, z = obstacles[i](k)
            obs_art[i].set_verts(cubes_obs[i].change_center(x,y,z))
        # Moving the drone to the accurate position at each time step
        x_drone, y_drone, z_drone = drone(k)
        drone_art.set_verts(drone_cube.change_center(x_drone, y_drone, z_drone))
    anim = animation.FuncAnimation(
        fig, animator, frames=2*(N+1), interval=tickrate, blit=False)
    plt.show()
    return anim

# Creating and solving the Pyomo model
def solve_model(N, x_min, x_max, y_min, y_max, z_min, z_max, v_max, a_max, obstacles, zones, drone_w, x_start, y_start, z_start, x_goal, y_goal, z_goal):
    model = pyo.ConcreteModel()
    # Creating a set for the time steps
    model.steps = pyo.RangeSet(0, N)
    # The sampling time (dt) and its square (dt2)
    model.dt = pyo.Var(domain=pyo.PositiveReals, bounds=(0.01,1))
    # Setting up the bounds for the position coordinates, velocity and acceleration components
    v_lim = np.sqrt(v_max ** 2 / 3)
    a_lim = np.sqrt(a_max ** 2 / 3)
    def x_bounds(model, i, j):
        if i == 0:
            return (x_min,x_max)
        elif i == 1:
            return (y_min,y_max)
        elif i == 2:
            return (z_min,z_max)
        else:
            return (-v_lim,v_lim)
    # The variable x includes the (x,y,z) position coordinates and the (v_x,v_y,v_z) velocity components at each time step
    model.x = pyo.Var(range(0,6), model.steps, domain=pyo.Reals, bounds=x_bounds)
    # The variable u includes the (a_x,a_y,a_z) acceleration components at each time step
    model.u = pyo.Var(range(0,3), model.steps, domain=pyo.Reals, bounds=(-a_lim,a_lim))
    # Binary variables for avoiding obstacles
    model.x_less = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.x_more = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.y_less = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.y_more = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.z_less = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.z_more = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    # Binary variables for avoiding danger zones
    model.zone_x_less = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_x_more = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_y_less = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_y_more = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_z_less = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_z_more = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    # Constraints that describe the dynamics of the drone
    @model.Constraint(range(0,N))
    def SSMx(model, i):
        return model.x[0,i+1] == model.x[0,i] + model.dt * model.x[3,i] + 0.5 * model.dt ** 2 * model.u[0,i]
    @model.Constraint(range(0,N))
    def SSMy(model, i):
        return model.x[1,i+1] == model.x[1,i] + model.dt * model.x[4,i] + 0.5 * model.dt ** 2 * model.u[1,i]
    @model.Constraint(range(0,N))
    def SSMz(model, i):
        return model.x[2,i+1] == model.x[2,i] + model.dt * model.x[5,i] + 0.5 * model.dt ** 2 * model.u[2,i]
    @model.Constraint(range(0,N))
    def SSMvx(model, i):
        return model.x[3,i+1] == model.x[3,i] + model.dt * model.u[0,i]
    @model.Constraint(range(0,N))
    def SSMvy(model, i):
        return model.x[4,i+1] == model.x[4,i] + model.dt * model.u[1,i]
    @model.Constraint(range(0,N))
    def SSMvz(model, i):
        return model.x[5,i+1] == model.x[5,i] + model.dt * model.u[2,i]
    # Constraint for avoiding obstacles and danger zones created with the big-M method
    M = np.abs(x_max - x_min + y_max - y_min + z_max - z_min)
    # These constraints guarantee that the drone avoids each obstacle at every time step
    model.x_less_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[0,j] <= obstacles[i].x[j]-(obstacles[i].wx/2)-drone_w/2 + M * model.x_less[i,j])
    model.x_more_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[0,j] >= obstacles[i].x[j]+(obstacles[i].wx/2)+drone_w/2 - M * model.x_more[i,j])
    model.y_less_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[1,j] <= obstacles[i].y[j]-(obstacles[i].wy/2)-drone_w/2 + M * model.y_less[i,j])
    model.y_more_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[1,j] >= obstacles[i].y[j]+(obstacles[i].wy/2)+drone_w/2 - M * model.y_more[i,j])
    model.z_less_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[2,j] <= obstacles[i].z[j]-(obstacles[i].wz/2)-drone_w/2 + M * model.z_less[i,j])
    model.z_more_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[2,j] >= obstacles[i].z[j]+(obstacles[i].wz/2)+drone_w/2 - M * model.z_more[i,j])
    model.avoid = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x_less[i,j] + model.x_more[i,j] + model.y_less[i,j] + model.y_more[i,j] + model.z_less[i,j] + model.z_more[i,j] <= 5)
    # These constraints inform us if any danger zone contain the drone at any time step
    model.zone_x_less_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[0,j] <= zones[i].x-(zones[i].wx/2)-drone_w/2 + M * model.zone_x_less[i,j])
    model.zone_x_more_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[0,j] >= zones[i].x+(zones[i].wx/2)+drone_w/2 - M * model.zone_x_more[i,j])
    model.zone_y_less_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[1,j] <= zones[i].y-(zones[i].wy/2)-drone_w/2 + M * model.zone_y_less[i,j])
    model.zone_y_more_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[1,j] >= zones[i].y+(zones[i].wy/2)+drone_w/2 - M * model.zone_y_more[i,j])
    model.zone_z_less_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[2,j] <= zones[i].z-(zones[i].wz/2)-drone_w/2 + M * model.zone_z_less[i,j])
    model.zone_z_more_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[2,j] >= zones[i].z+(zones[i].wz/2)+drone_w/2 - M * model.zone_z_more[i,j])
    model.zone_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.zone_x_less[i,j]+model.zone_x_more[i,j]+model.zone_y_less[i,j]+model.zone_y_more[i,j]+model.zone_z_less[i,j]+model.zone_z_more[i,j]-model.zone[i,j] <= 5)
    # A constraint list that includes the initial and final conditions
    model.conditions = pyo.ConstraintList()
    model.conditions.add(model.x[0,0] == x_start)
    model.conditions.add(model.x[1,0] == y_start)
    model.conditions.add(model.x[2,0] == z_start)
    model.conditions.add(model.x[3,0] == 0)
    model.conditions.add(model.x[4,0] == 0)
    model.conditions.add(model.x[5,0] == 0)
    model.conditions.add(model.x[0,N] == x_goal)
    model.conditions.add(model.x[1,N] == y_goal)
    model.conditions.add(model.x[2,N] == z_goal)
    model.conditions.add(model.x[3,N] == 0)
    model.conditions.add(model.x[4,N] == 0)
    model.conditions.add(model.x[5,N] == 0)
    # The objective is to minimize the sampling time
    # # The objective includes and extra term, that serves as a penalty for spending time in danger zones
    alpha = 10
    obj_expr = model.dt + sum(model.zone[i,j]*alpha for i in range(len(zones)) for j in model.steps)
    model.obj = pyo.Objective(sense=pyo.minimize, expr=obj_expr)
    solver = pyo.SolverFactory("baron")
    results = solver.solve(model, options={"threads": 8}, tee=True)
    # If the model is solved successfully, the function returns three lists containing the x, y and z coordinates of the trajectory at each time step and the calculated sampling time
    # If the model is not solved (e.g. the model is infeasible), the function returns the solver status and the termination condition
    if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        for i in model.steps:
            trajectory_x.append(pyo.value(model.x[0,i]))
            trajectory_y.append(pyo.value(model.x[1,i]))
            trajectory_z.append(pyo.value(model.x[2,i]))
        return trajectory_x, trajectory_y, trajectory_z, pyo.value(model.dt)
    else:
        return -1, -1, -1, -1

# Main
def main():
    # The parameters can be specified here
    # The number of time steps:
    N = 100
    # The available space for the movement of the drone:
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    z_min, z_max = 0, 10
    # The maximum velocity of the drone:
    v_max = 4
    # The maximum acceleration of the drone:
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
    # The width of the drone
    drone_w = 0.2
    # The coordinates of the starting position of the drone
    x_start, y_start, z_start = 0, 0, 0
    # The coordinates of the destination of the drone
    x_goal, y_goal, z_goal = 10, 10, 10
    # Solving the model
    x, y, z, dt = solve_model(N, x_min, x_max, y_min, y_max, z_min, z_max, v_max, a_max, obstacles, zones, drone_w, x_start, y_start, z_start, x_goal, y_goal, z_goal)
    # If the model was solved successfully:
    #   - The coordinates of the trajectory is printed at each time step
    #   - The total travel time is printed
    #   - An animation shows the drone reaching its destination
    if x != -1 and y != -1 and z != -1:
        for i in range(N+1):
            print(f"x[{i}]:\t{round(x[i],4)}\ty[{i}]:\t{round(y[i],4)}\tz[{i}]:\t{round(z[i],4)}")
        time = N * dt
        print(f"Total travel time: {round(time,3)} seconds")
        drone = Drone(x, y, z, drone_w)
        animate_scene(obstacles, zones, drone, N, x_min, x_max, y_min, y_max, z_min, z_max)
    else:
        print("The problem was infeasible.")

if __name__ == "__main__":
    main()