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
    
# A class for the drones
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

# The animate_scene function creates an animation of the drones, obstacles and zones
def animate_scene(obstacles, zones, drones, N, x_min, x_max, y_min, y_max):
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
    # Creating the drones as red Rectangle patches
    drone_art = []
    for i in range(0,len(drones)):
        drone_art.append(plt.Rectangle((drones[i].x[0] - drones[i].w / 2, drones[i].y[0] - drones[i].w / 2), drones[i].w, drones[i].w, color="red"))
        ax.add_patch(drone_art[i])
    # Plotting the trajectory of the drones
    for i in range(0,len(drones)):
        ax.plot(drones[i].x, drones[i].y, color="red")
    tickrate = 30 # ms
    # The animator subfunction
    def animator(k):
        # Moving the obstacles to the accurate positions at each time step
        for i in range(0, len(obstacles)):
            x, y = obstacles[i](k)
            obs_art[i].set_xy((x, y))
        # Moving the drones to the accurate position at each time step
        for i in range(0,len(drones)):
            x_drone, y_drone = drones[i](k)
            drone_art[i].set_xy((x_drone, y_drone))
    anim = animation.FuncAnimation(
        fig, animator, frames=2*(N+1), interval=tickrate, blit=False)
    plt.show()
    return anim

# Creating and solving the Pyomo model
def solve_model(N, x_min, x_max, y_min, y_max, v_max, a_max, obstacles, zones, drone_w, drones, x_start, y_start, x_goal, y_goal):
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
    v_lim = np.sqrt(v_max ** 2 / 2)
    a_lim = np.sqrt(a_max ** 2 / 2)
    def x_bounds(model, i, j, k):
        if i == 0:
            return (x_min,x_max)
        elif i == 1:
            return (y_min,y_max)
        else:
            return (-v_lim,v_lim)
    # The variable x includes the (x,y) position coordinates and the (v_x,v_y) velocity components at each time step
    model.x = pyo.Var(range(0,4), model.steps, range(0,drones), domain=pyo.Reals, bounds=x_bounds)
    # The variable u includes the (a_x,a_y) acceleration components at each time step
    model.u = pyo.Var(range(0,2), model.steps, range(0,drones), domain=pyo.Reals, bounds=(-a_lim,a_lim))
    # Binary variables for avoiding obstacles
    model.left = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.right = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.below = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    model.above = pyo.Var(range(0,len(obstacles)), model.steps, range(0,drones), domain=pyo.Binary)
    # Binary variables for avoiding danger zones
    model.zone_left = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_right = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_below = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone_above = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    model.zone = pyo.Var(range(0,len(zones)), model.steps, range(0,drones), domain=pyo.Binary)
    # Binary variables for avoiding other drones
    model.drone_left = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_right = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_below = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    model.drone_above = pyo.Var(range(0,drones), range(0,drones), model.steps, domain=pyo.Binary)
    # Constraints that describe the dynamics of the drones
    @model.Constraint(range(0,N), range(0,drones))
    def SSMx(model, i, j):
        return model.x[0,i+1,j] == model.x[0,i,j] + model.dt * model.x[2,i,j] + 0.5 * model.dt2 * model.u[0,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMy(model, i, j):
        return model.x[1,i+1,j] == model.x[1,i,j] + model.dt * model.x[3,i,j] + 0.5 * model.dt2 * model.u[1,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMvx(model, i, j):
        return model.x[2,i+1,j] == model.x[2,i,j] + model.dt * model.u[0,i,j]
    @model.Constraint(range(0,N), range(0,drones))
    def SSMvy(model, i, j):
        return model.x[3,i+1,j] == model.x[3,i,j] + model.dt * model.u[1,i,j]
    # Constraint for avoiding obstacles and danger zones created with the big-M method
    M = np.abs(x_max - x_min + y_max - y_min)
    # These constraints guarantee that the drones avoid each obstacle at every time step
    model.left_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] <= obstacles[i].x[j]-(obstacles[i].wx/2)-drone_w/2 + M * model.left[i,j,k])
    model.right_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] >= obstacles[i].x[j]+(obstacles[i].wx/2)+drone_w/2 - M * model.right[i,j,k])
    model.below_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] <= obstacles[i].y[j]-(obstacles[i].wy/2)-drone_w/2 + M * model.below[i,j,k])
    model.above_con = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] >= obstacles[i].y[j]+(obstacles[i].wy/2)+drone_w/2 - M * model.above[i,j,k])
    model.avoid = pyo.Constraint(range(0,len(obstacles)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.left[i,j,k] + model.right[i,j,k] + model.below[i,j,k] + model.above[i,j,k] <= 3)
    # These constraints inform us if any danger zone contain the drones at any time step
    model.zone_left_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] <= zones[i].x-(zones[i].wx/2)-drone_w/2 + M * model.zone_left[i,j,k])
    model.zone_right_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[0,j,k] >= zones[i].x+(zones[i].wx/2)+drone_w/2 - M * model.zone_right[i,j,k])
    model.zone_below_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] <= zones[i].y-(zones[i].wy/2)-drone_w/2 + M * model.zone_below[i,j,k])
    model.zone_above_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.x[1,j,k] >= zones[i].y+(zones[i].wy/2)+drone_w/2 - M * model.zone_above[i,j,k])
    model.zone_con = pyo.Constraint(range(0,len(zones)), model.steps, range(0,drones), rule=lambda model, i, j, k: model.zone_left[i,j,k]+model.zone_right[i,j,k]+model.zone_below[i,j,k]+model.zone_above[i,j,k]-model.zone[i,j,k] <= 3)
    # Avoiding collisions between drones
    def left(model, i,j,k):
        if i<j:
            return model.x[0,k,i] <= model.x[0,k,j]-drone_w + M * model.drone_left[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_left_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=left)
    def right(model, i,j,k):
        if i<j:
            return model.x[0,k,i] >= model.x[0,k,j]+drone_w - M * model.drone_right[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_right_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=right)
    def below(model, i,j,k):
        if i<j:
            return model.x[1,k,i] <= model.x[1,k,j]-drone_w + M * model.drone_below[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_below_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=below)
    def above(model, i,j,k):
        if i<j:
            return model.x[1,k,i] >= model.x[1,k,j]+drone_w - M * model.drone_above[i,j,k]
        else:
            return pyo.Constraint.Skip
    model.drone_above_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=above)
    def con(model,i,j,k):
        if i<j:
            return model.drone_left[i,j,k] + model.drone_right[i,j,k] + model.drone_below[i,j,k] + model.drone_above[i,j,k] <= 3
        else:
            return pyo.Constraint.Skip
    model.drone_con = pyo.Constraint(range(0,drones), range(0,drones), model.steps, rule=con)
    # A constraint list that includes the initial and final conditions
    model.conditions = pyo.ConstraintList()
    for i in range(0,drones):
        model.conditions.add(model.x[0,0,i] == x_start[i])
        model.conditions.add(model.x[1,0,i] == y_start[i])
        model.conditions.add(model.x[2,0,i] == 0)
        model.conditions.add(model.x[3,0,i] == 0)
        model.conditions.add(model.x[0,N,i] == x_goal[i])
        model.conditions.add(model.x[1,N,i] == y_goal[i])
        model.conditions.add(model.x[2,N,i] == 0)
        model.conditions.add(model.x[3,N,i] == 0)
    # The objective is to minimize the sampling time
    # The objective includes and extra term, that serves as a penalty for spending time in danger zones
    alpha = 10
    obj_expr = model.dt + sum(model.zone[i,j,k]*alpha for i in range(len(zones)) for j in model.steps for k in range(0,drones))
    model.obj = pyo.Objective(sense=pyo.minimize, expr=obj_expr)
    solver = pyo.SolverFactory("gurobi_direct")
    results = solver.solve(model, options={"NonConvex": 2}, tee=True)
    # If the model is solved successfully, the function returns two lists containing the x and y coordinates of the trajectories at each time step and the calculated sampling time
    # If the model is not solved (e.g. the model is infeasible), the function returns the solver status and the termination condition
    if results.solver.status == pyo.SolverStatus.ok and results.solver.termination_condition == pyo.TerminationCondition.optimal:
        trajectory_x = []
        trajectory_y = []
        for i in range(0,drones):
            x_result = []
            y_result = []
            for j in model.steps:
                x_result.append(pyo.value(model.x[0,j,i]))
                y_result.append(pyo.value(model.x[1,j,i]))
            trajectory_x.append(x_result)
            trajectory_y.append(y_result)
        return trajectory_x, trajectory_y, pyo.value(model.dt)
    else:
        return -1, -1, -1
    
# Main
def main():
    # The parameters can be specified here
    # The number of time steps:
    N = 100
    # The available space for the movement of the drones:
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10
    # The maximum velocity of the drones:
    v_max = 4
    # The maximum acceleration of the drones:
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
    # The width of the drones
    drone_w = 0.2
    # The number of drones
    drones_number = 3
    # The coordinates of the starting positions of the drones
    x_start, y_start = [0,10,5], [0,0,0]
    # The coordinates of the destinations of the drones
    x_goal, y_goal = [10,5,0], [10,10,10]
    # Solving the model
    x, y, dt = solve_model(N, x_min, x_max, y_min, y_max, v_max, a_max, obstacles, zones, drone_w, drones_number, x_start, y_start, x_goal, y_goal)
    # If the model was solved successfully:
    #   - The total travel time is printed
    #   - An animation shows the drones reaching their destinations
    if x != -1 and y != -1:
        time = N * dt
        print(f"Total travel time: {round(time,3)} seconds")
        drones = []
        for i in range(0,drones_number):
            drones.append(Drone(x[i], y[i], drone_w))
        animate_scene(obstacles, zones, drones, N, x_min, x_max, y_min, y_max)
    else:
        print("The problem was infeasible.")

if __name__ == "__main__":
    main()