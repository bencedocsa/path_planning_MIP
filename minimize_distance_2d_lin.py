# Imports
import pyomo.environ as pyo
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# A class for the obstacles
class Obstacle:
    def __init__(self, x, y, wx, wy, steps):
        self.x = x
        self.y = y
        self.wx = wx
        self.wy = wy
        self.steps = steps
    # __call__ function for the animation
    def __call__(self, t):
        q, r = np.divmod(t, self.steps)
        if q % 2 == 1:
            r += 1
            r *= -1
        return self.x[r] - (self.wx / 2), self.y[r] - (self.wy / 2)

# A class for the drone
class Drone:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w
    # __call__ function for the animation
    def __call__(self, t):
        if t < len(self.x):
            return self.x[t] - (self.w / 2), self.y[t] - (self.w / 2)
        else:
            return self.x[-1] - (self.w / 2), self.y[-1] - (self.w / 2)

# A class for the danger zones 
class Zone:
    def __init__(self, x, y, wx, wy):
        self.x = x
        self.y = y
        self.wx = wx
        self.wy = wy

# The create_spline function returns two arrays with the x and y coordinates of the points of the spline
def create_spline(nodes, steps):
    x = nodes[:,0]
    y = nodes[:,1]
    tck, _ = interpolate.splprep([x,y],s=0)
    x1, y1 = interpolate.splev(np.linspace(0, 1, steps), tck)
    inter_point_differences_x = np.diff(x1)
    inter_point_differences_y = np.diff(y1)
    inter_point_distances = np.sqrt(inter_point_differences_x ** 2 + inter_point_differences_y ** 2)
    cumulative_distance = np.cumsum(inter_point_distances)
    cumulative_distance /= cumulative_distance[-1]
    cumulative_distance = np.insert(cumulative_distance, 0, 0)
    tck_prime, _ = interpolate.splprep([np.linspace(0, 1, num=len(cumulative_distance))], u=cumulative_distance, s=0)
    equidistant = interpolate.splev(np.linspace(0, 1, steps), tck_prime)
    x2, y2 = interpolate.splev(equidistant, tck)
    return x2[0], y2[0]

# The animate_scene function creates an animation of the drone, obstacles and zones
def animate_scene(obstacles, zones, drone, N, x_min, x_max, y_min, y_max):
    fig,ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    # Creating the obstacles as blue Rectangle patches
    obs_art = []
    for i in range(0, len(obstacles)):
        x = obstacles[i].x[0] - (obstacles[i].wx / 2)
        y = obstacles[i].y[0] - (obstacles[i].wy / 2)
        wx = obstacles[i].wx
        wy = obstacles[i].wy
        obs_art.append(plt.Rectangle((x, y), wx, wy, color="blue"))
        ax.add_patch(obs_art[i])
    # Creating the danger zones as green Rectangle patches
    zone_art = []
    for i in range(0,len(zones)):
        x = zones[i].x - (zones[i].wx / 2)
        y = zones[i].y - (zones[i].wy / 2)
        wx = zones[i].wx
        wy = zones[i].wy
        zone_art.append(plt.Rectangle((x, y), wx, wy, color="green"))
        ax.add_patch(zone_art[i])
    # Creating the drone as a red Rectangle patch
    drone_art = plt.Rectangle((drone.x[0] - drone.w / 2, drone.y[0] - drone.w / 2), drone.w, drone.w, color="red")
    ax.add_patch(drone_art)
    # Plotting the trajectory of the drone
    ax.plot(drone.x, drone.y, color="red")
    tickrate = 30 # ms
    # The animator subfunction
    def animator(k):
        # Moving the obstacles to the accurate positions at each time step
        for i in range(0, len(obstacles)):
            x, y = obstacles[i](k)
            obs_art[i].set_xy((x, y))
        # Moving the drone to the accurate position at each time step
        x_drone, y_drone = drone(k)
        drone_art.set_xy((x_drone, y_drone))
    anim = animation.FuncAnimation(
        fig, animator, frames=2*(N+1), interval=tickrate, blit=False)
    plt.show()
    return anim

# Creating and solving the Pyomo model
def solve_model(N, dt, x_min, x_max, y_min, y_max, v_max, a_max, obstacles, zones, drone_w, x_start, y_start, x_goal, y_goal):
    model = pyo.ConcreteModel()
    # Creating a set for the time steps
    model.steps = pyo.RangeSet(0, N)
    # Setting up the bounds for the position coordinates, velocity and acceleration components
    v_lim = np.sqrt(v_max ** 2 / 2)
    a_lim = np.sqrt(a_max ** 2 / 2)
    def x_bounds(model, i, j):
        if i == 0:
            return (x_min,x_max)
        elif i == 1:
            return (y_min,y_max)
        else:
            return (-v_lim,v_lim)
    # The variable x includes the (x,y) position coordinates and the (v_x,v_y) velocity components at each time step
    model.x = pyo.Var(range(0,4), model.steps, domain=pyo.Reals, bounds=x_bounds)
    # The variable u includes the (a_x,a_y) acceleration components at each time step
    model.u = pyo.Var(range(0,2), model.steps, domain=pyo.Reals, bounds=(-a_lim,a_lim))
    # Slack variables
    model.s_x = pyo.Var(model.steps, domain=pyo.PositiveReals)
    model.s_y = pyo.Var(model.steps, domain=pyo.PositiveReals)
    # Binary variables for avoiding obstacles
    model.left = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.right = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.below = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    model.above = pyo.Var(range(0,len(obstacles)), model.steps, domain=pyo.Binary)
    # Binary variables for avoiding danger zones
    model.zone_left = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_right = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_below = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone_above = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    model.zone = pyo.Var(range(0,len(zones)), model.steps, domain=pyo.Binary)
    # Constraints that describe the dynamics of the drone
    @model.Constraint(range(0,N))
    def SSMx(model, i):
        return model.x[0,i+1] == model.x[0,i] + dt * model.x[2,i] + 0.5 * dt ** 2 * model.u[0,i]
    @model.Constraint(range(0,N))
    def SSMy(model, i):
        return model.x[1,i+1] == model.x[1,i] + dt * model.x[3,i] + 0.5 * dt ** 2 * model.u[1,i]
    @model.Constraint(range(0,N))
    def SSMvx(model, i):
        return model.x[2,i+1] == model.x[2,i] + dt * model.u[0,i]
    @model.Constraint(range(0,N))
    def SSMvy(model, i):
        return model.x[3,i+1] == model.x[3,i] + dt * model.u[1,i]
    # Constraints to use the slack variables
    model.s_x_pos_con = pyo.Constraint(model.steps, rule=lambda model, i: model.s_x[i] >= model.x[0,i]-x_goal)
    model.s_x_neg_con = pyo.Constraint(model.steps, rule=lambda model, i: model.s_x[i] >= -model.x[0,i]+x_goal)
    model.s_y_pos_con = pyo.Constraint(model.steps, rule=lambda model, i: model.s_y[i] >= model.x[1,i]-y_goal)
    model.s_y_neg_con = pyo.Constraint(model.steps, rule=lambda model, i: model.s_y[i] >= -model.x[1,i]+y_goal)
    # Constraint for avoiding obstacles and danger zones created with the big-M method
    M = np.abs(x_max - x_min + y_max - y_min)
    # These constraints guarantee that the drone avoids each obstacle at every time step
    model.left_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[0,j] <= obstacles[i].x[j]-(obstacles[i].wx/2)-drone_w/2 + M * model.left[i,j])
    model.right_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[0,j] >= obstacles[i].x[j]+(obstacles[i].wx/2)+drone_w/2 - M * model.right[i,j])
    model.below_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[1,j] <= obstacles[i].y[j]-(obstacles[i].wy/2)-drone_w/2 + M * model.below[i,j])
    model.above_con = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.x[1,j] >= obstacles[i].y[j]+(obstacles[i].wy/2)+drone_w/2 - M * model.above[i,j])
    model.avoid = pyo.Constraint(range(0,len(obstacles)), model.steps, rule=lambda model, i, j: model.left[i,j] + model.right[i,j] + model.below[i,j] + model.above[i,j] <= 3)
    # These constraints inform us if any danger zone contain the drone at any time step
    model.zone_left_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[0,j] <= zones[i].x-(zones[i].wx/2)-drone_w/2 + M * model.zone_left[i,j])
    model.zone_right_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[0,j] >= zones[i].x+(zones[i].wx/2)+drone_w/2 - M * model.zone_right[i,j])
    model.zone_below_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[1,j] <= zones[i].y-(zones[i].wy/2)-drone_w/2 + M * model.zone_below[i,j])
    model.zone_above_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.x[1,j] >= zones[i].y+(zones[i].wy/2)+drone_w/2 - M * model.zone_above[i,j])
    model.zone_con = pyo.Constraint(range(0,len(zones)), model.steps, rule=lambda model, i, j: model.zone_left[i,j]+model.zone_right[i,j]+model.zone_below[i,j]+model.zone_above[i,j]-model.zone[i,j] <= 3)
    # A constraint list that includes the initial and final conditions
    model.conditions = pyo.ConstraintList()
    model.conditions.add(model.x[0,0] == x_start)
    model.conditions.add(model.x[1,0] == y_start)
    model.conditions.add(model.x[2,0] == 0)
    model.conditions.add(model.x[3,0] == 0)
    model.conditions.add(model.x[0,N] == x_goal)
    model.conditions.add(model.x[1,N] == y_goal)
    model.conditions.add(model.x[2,N] == 0)
    model.conditions.add(model.x[3,N] == 0)
    # The objective is to minimize the sum of distances from the destination
    # The objective includes an extra term, that serves as a penalty for spending time in danger zones
    alpha = 1000
    obj_expr = sum(model.s_x[i] + model.s_y[i] for i in model.steps) + sum(model.zone[i,j]*alpha for i in range(len(zones)) for j in model.steps)
    model.obj = pyo.Objective(sense=pyo.minimize, expr=obj_expr)
    solver = pyo.SolverFactory("gurobi_direct")
    results = solver.solve(model, options={"NonConvex": 2}, tee=True)
    # If the model is solved successfully, the function returns two lists containing the x and y coordinates of the trajectory at each time step and the calculated sampling time
    # If the model is not solved (e.g. the model is infeasible), the function returns the solver status and the termination condition
    if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
        trajectory_x = []
        trajectory_y = []
        for i in model.steps:
            trajectory_x.append(pyo.value(model.x[0,i]))
            trajectory_y.append(pyo.value(model.x[1,i]))
        return trajectory_x, trajectory_y, dt
    else:
        return -1, -1, -1
    
# Main
def main():
    # The parameters can be specified here
    # The number of time steps:
    N = 100
    # The sampling time:
    dt = 0.1
    # The available space for the movement of the drone:
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    # The maximum velocity of the drone:
    v_max = 4
    # The maximum acceleration of the drone:
    a_max = 4
    # The list of obstacles
    # Obstacle can be created using the Obstacle class
    obstacles = []
    nodes = np.array([[1, 9],
                      [4, 7],
                      [2.5, 5],
                      [5, 2],
                      [8, 4],
                      [6, 6]])
    x_spline, y_spline = create_spline(nodes, N+1)
    obstacles.append(Obstacle(x_spline, y_spline, 2, 2, N+1))
    obstacles.append(Obstacle(np.full(N+1,3.5), np.linspace(5,1,N+1), 2, 2, N+1))
    obstacles.append(Obstacle(np.full(N+1,1.5), np.full(N+1,1), 1, 2, N+1))
    obstacles.append(Obstacle(np.full(N+1,9), np.full(N+1,8), 1, 4, N+1))
    # The list of danger zones
    # Danger zones can be created using the Zone class
    zones = []
    zones.append(Zone(7, 5, 2, 2))
    # The width of the drone
    drone_w = 0.2
    # The coordinates of the starting position of the drone
    x_start, y_start = 0, 0
    # The coordinates of the destination of the drone
    x_goal, y_goal = 10, 10
    # Solving the model
    x, y, dt = solve_model(N, dt, x_min, x_max, y_min, y_max, v_max, a_max, obstacles, zones, drone_w, x_start, y_start, x_goal, y_goal)
    # If the model was solved successfully:
    #   - The coordinates of the trajectory is printed at each time step
    #   - The total travel time is printed
    #   - An animation shows the drone reaching its destination
    if x != -1 and y != -1:
        for i in range(N+1):
            print(f"x[{i}]:\t{round(x[i],4)}\ty[{i}]:\t{round(y[i],4)}")
        time = N * dt
        print(f"Total travel time: {round(time,3)} seconds")
        drone = Drone(x, y, drone_w)
        animate_scene(obstacles, zones, drone, N, x_min, x_max, y_min, y_max)
    else:
        print("The problem was infeasible.")

if __name__ == "__main__":
    main()