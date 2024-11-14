import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Example parameters
V = range(4)  # nodes including depot, 0 is depot
V_star = range(1, 4)  # nodes excluding depot
K = range(5)  # two vehicles for example
Q = 15  # capacity of each vehicle
M = 10000  # large constant for constraints
M_2 = 10000 # another large constraint for constraints
vehicle_speed = 60  # vehicle speed in km/h

# DISTANCES
np.random.seed(42)
distances = np.random.randint(10, 51, size=(len(V), len(V)))
# distance matrix symmetric and set the diagonal to 0
for i in V:
    distances[i, i] = 0
    for j in range(i + 1, len(V)):
        distances[j, i] = distances[i, j]

# triangle inequality enforced
for i in V:
    for j in V:
        for k in V:
            if distances[i, j] > distances[i, k] + distances[k, j]:
                distances[i, j] = distances[i, k] + distances[k, j]

# calculate travel times (distance / speed) and travel costs (distance * $0.15/mile * 0.62 mile/km)
# $0.15/mile is the average American cost for gas as per New York Times
t = distances / vehicle_speed
c = distances * 0.093

# Sample demands (including depot which has 0 demand)
d = [0, 5, 10, 8]

# time window parameters
a = [0, 10, 15, 20]
b = [100, 100, 100, 100]
s = [30, 30, 30, 30]

# Model setup
model = gp.Model("VRPTW")

# Decision variables
x = model.addVars(V, V, vtype=GRB.BINARY, name="x")  # binary variable
q = model.addVars(V, vtype=GRB.CONTINUOUS, name="q")  # continuous variable
tau = model.addVars(V, vtype=GRB.CONTINUOUS, name="tau") # continuous variable

# objective fn.
model.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in V for j in V), GRB.MINIMIZE)

# constraints

# constraint 1, 2
for j in V_star:
    model.addConstr(gp.quicksum(x[i, j] for i in V) == 1, name=f"visit_{j}")
for i in V_star:
    model.addConstr(gp.quicksum(x[i, j] for j in V) == 1, name=f"departure_{i}")

# constraint 3, 4
model.addConstr(gp.quicksum(x[i, 0] for i in V_star) <= len(K), name="vehicle_out")
model.addConstr(gp.quicksum(x[0, j] for j in V_star) <= len(K), name="vehicle_in")

# constraint 5 (fleet size)
model.addConstr(gp.quicksum(x[0, j] for j in V_star) <= np.ceil(sum(d[i] for i in V_star) / Q), name="fleet_capacity")

# constraint 6 (set depot demand 0)
model.addConstr(q[0] == 0, name="depot_demand")

# constraint 7, 8, 9 (subtour elimination)
model.addConstrs((q[i] + d[j] <= q[j] + Q * (1 - x[i, j]) for i in V_star for j in V_star), name="load_balance1")
model.addConstrs((q[i] + d[j] >= q[j] - M * (1 - x[i, j]) for i in V_star for j in V_star), name="load_balance2")
model.addConstrs((q[i] >= d[i] for i in V_star), name="demand_lower_bound")
model.addConstrs((q[i] <= Q for i in V_star), name="demand_upper_bound")

# constraint 10 (service time subtour)
model.addConstrs((tau[i] + s[i] + t[i, j] <= tau[j] + M_2 * (1 - x[i, j]) for i in V_star for j in V_star if i != j), name="service_time")

# constraint 11 (tau bounds)
model.addConstrs((a[i] <= tau[i] for i in V_star), name="time_window_start")
model.addConstrs((tau[i] <= b[i] for i in V_star), name="time_window_end")

# set depot time to zero
model.addConstr(tau[0] == 0, name="depot_time")

# solve
model.optimize()

# output
if model.status == GRB.OPTIMAL:
    solution = model.getAttr('x', x)
    for i in V:
        for j in V:
            if solution[i, j] > 0.5:
                print(f"Edge ({i}, {j}) is in the optimal route with cost {c[i, j]}")
    satisfied_demand = model.getAttr('x', q)
    for i in V:
        print(f"Demand satisfied at node {i}: {satisfied_demand[i]}")
else:
    print("No optimal solution found.")