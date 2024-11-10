import random
import gurobipy as gp
from gurobipy import GRB
import math
from collections import defaultdict
from itertools import combinations

# function to calculate the minimum vehicles required
def calculate_r_S(S, d, C):
    demand_sum = sum(d[i] for i in S)
    return math.ceil(demand_sum / C)

# Function to find the shortest subtour in the current solution
def shortest_subtour(edges):
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)

    unvisited = set(node_neighbors)
    shortest = None
    while unvisited:
        cycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors.pop()
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in node_neighbors[current] if j in unvisited]
        if shortest is None or len(cycle) < len(shortest):
            shortest = cycle
    return shortest

# Define the callback for subtour elimination
class TSPCallback:
    def __init__(self, nodes, x, d, C):
        self.nodes = nodes
        self.x = x
        self.d = d
        self.C = C

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            self.eliminate_subtours(model)
    
    def eliminate_subtours(self, model):
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        tour = shortest_subtour(edges)
        if len(tour) < len(self.nodes):
            r_S = calculate_r_S(tour, self.d, self.C)
            subset_size = len(tour)
            # Add subtour elimination constraint for this subset
            model.cbLazy(
                gp.quicksum(self.x[i, j] for i, j in combinations(tour, 2)) <= subset_size - r_S # 1.12 in Toth
            )

# new model
model = gp.Model("VRPTW")

# Define sets and parameters (simple for now)
V = range(5) # Set of all nodes: {0, 1, 2, 3, 4}
K = 5 # number of vehicles
C = 100 # capacity of each homogenous vehicle

d = {1: 10, 2: 15, 3: 20, 4: 25} # Demand at each customer node (d[j] is the demand for customer j)

# Time windows for each node (a[i] and b[i] define the earliest and latest times for service at node i)
a = {1: 360, 2: 360, 3: 360, 4: 360} # Earliest times, all 6 AM
b = {1: 900, 2: 960, 3: 1020, 4: 1080} # End times vary: 3 PM, 4 PM, 5 PM, 6 PM
s = {0: 30, 1: 60, 2: 90, 3: 60, 4: 60} # service time in minutes

SPEED = 60 # km/min
# Generate distances between nodes that respect the triangle inequality
num_nodes = 5
distances = {}
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        dist = random.randint(10, 50)  # generate a base distance
        distances[i, j] = dist
        distances[j, i] = dist  # symmetry

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
# Done above via lazy constraints

# 6. Binary Constraint for x, already typed above.

# 7. eqtn 4.8
for i in V:
    if i != 0:  # Exclude depot
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
model.Params.LazyConstraints = 1
model._callback = TSPCallback(V, x, d, C)
model.optimize(model._callback)

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