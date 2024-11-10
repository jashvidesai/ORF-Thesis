import random
import gurobipy as gp
from gurobipy import GRB

# new model
model = gp.Model("VRPTW_MTZ")

# Define sets and parameters (simple for now)
V = range(5) # Set of all nodes: {0, 1, 2, 3, 4}
K = 5 # number of vehicles
C = 100 # capacity of each homogenous vehicle

d = {1: 5, 2: 5, 3: 5, 4: 5} # Demand at each customer node (d[j] is the demand for customer j)

# Time windows for each node (a[i] and b[i] define the earliest and latest times for service at node i)
a = {1: 360, 2: 360, 3: 360, 4: 360} # Earliest times, all 6 AM
b = {1: 900, 2: 960, 3: 1020, 4: 1080} # End times vary: 3 PM, 4 PM, 5 PM, 6 PM
s = {0: 30, 1: 30, 2: 30, 3: 30, 4: 30} # service time in minutes

SPEED = 60 # km/min
# Generate distances between nodes that respect the triangle inequality
num_nodes = 5
distances = {}
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        dist = random.randint(10, 50) # generate a base distance
        distances[i, j] = dist
        distances[j, i] = dist

# Enforce triangle inequality
for i in range(num_nodes):
    for j in range(num_nodes):
        for k in range(num_nodes):
            if i != j and j != k and i != k:
                distances[i, k] = min(distances[i, k], distances[i, j] + distances[j, k])

# Convert distances to travel times (in minutes)
t = {(i, j): round(distances[i, j] / SPEED * 60) for i in range(num_nodes) for j in range(num_nodes) if i != j}
c = t

connections = [(i, j) for i in V for j in V if i != j] # all valid connections (all pairs except self-loops)
x = model.addVars(connections, vtype=GRB.BINARY, name="x") # binary variable, 1 if the route from i to j is used, 0 otherwise
tau_i = model.addVars(V, vtype=GRB.CONTINUOUS, name="tau_i") # continuous variable representing the arrival time at each node i
u = model.addVars(V, vtype=GRB.CONTINUOUS, name="u")  # Represents the load after visiting each node

# Set Objective Function
model.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in V for j in V if i != j), GRB.MINIMIZE) # eqtn 4.1

# Constraints
# 1. eqtn 4.2
for j in V:
    if j != 0:
        model.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, name=f"visit_{j}")

# 2. eqtn 4.3
for i in V:
    if i != 0:
        model.addConstr(gp.quicksum(x[i, j] for j in V if i != j) == 1, name=f"leave_{i}")

# 3. eqtn 4.4
model.addConstr(gp.quicksum(x[i, 0] for i in V if i != 0) == K, name="return_depot")

# 4. eqtn 4.5
model.addConstr(gp.quicksum(x[0, j] for j in V if j != 0) == K, name="depart_depot")

# 5. eqtn 4.6 (subtour)
# a. MTZ load continuity constraint
for i in V:
    for j in V:
        if i != j and i != 0 and j != 0 and d[i] + d[j] <= C:
            model.addConstr(u[i] - u[j] + C * x[i, j] <= C - d[j], name=f"MTZ_continuity_{i}_{j}")

# b. Bounds on cumulative load u at each node
for i in V:
    if i != 0:
        model.addConstr(d[i] <= u[i], name=f"load_lower_bound_{i}")
        model.addConstr(u[i] <= C, name=f"load_upper_bound_{i}")

# 6. Binary Constraint for x, already typed above.

# 7. eqtn 4.8
for i in V:
    if i != 0:
        model.addConstr(gp.quicksum(d[j] * x[i, j] for j in V if j != 0 and i != j) <= C, name=f"capacity_{i}")

# 8. eqtn 4.9
for i in V:
    if i != 0:
        model.addConstr(a[i] <= tau_i[i], name=f"time_window_lower_{i}")  # Earliest arrival time
        model.addConstr(tau_i[i] <= b[i], name=f"time_window_upper_{i}")  # Latest arrival time

# 9. eqtn 4.10 (with added condition on x_{i, j} being used)
for i in V:
    for j in V:
        if i != j and i != 0 and j != 0:
            model.addGenConstrIndicator(x[i, j], True, tau_i[j] >= tau_i[i] + t[i, j] + s[i], name=f"time_seq_{i}_{j}")

# optimize model using lazy constraints
model.optimize()

# Check for infeasibility
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    model.computeIIS()
    model.write("infeasible_model.mps")
else:
    # extract routes
    used_connections = [(i, j) for i, j in x.keys() if x[i, j].X > 0.5]
    
    # Build routes starting from the depot
    routes = []
    for start in [j for i, j in used_connections if i == 0]:  # starting points for each route
        route = [0, start]  # start each route from the depot
        next_node = start
        while next_node != 0:
            next_node = next((j for i, j in used_connections if i == next_node), 0)  # move to next customer or back to depot
            route.append(next_node)
        routes.append(route)

    # print routes!
    for i, route in enumerate(routes):
        print(f"Route {i + 1}: {' -> '.join(map(str, route))}")