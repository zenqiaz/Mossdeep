import pulp
import numpy as np
import sys
import matplotlib.pyplot as plt
# Create a LP problem object


def rec(R_matrix, epsilon = 0.1, max_turns = 20):#playerA wants to maximize aRb, and player B wants to minimize it, so high eq is that B chooses a strategy and A responses, then high eq > low eq. Reward function is aRb, so size of A strategy vector is len(R), size of B strategy vector is len(R[0]).
    A_set, B_set = [],[]
    num_rows, num_columns = R_matrix.shape
    for i in range(num_rows):
        one_hot_vector = np.zeros(num_rows)
        position = i  
        one_hot_vector[position] = 1
        A_set.append(one_hot_vector)
    for j in range(num_columns):
        one_hot_vector = np.zeros(num_columns)
        position = j  
        one_hot_vector[position] = 1
        B_set.append(one_hot_vector)
    ep = float('inf')
    turns = 0
    flag_c = 0 
    while ep > epsilon and turns < max_turns:
        A_opst, B_opst, high_eq, low_eq, A_stat, B_stat, A_response, B_response = turn(A_set, B_set, R_matrix)
        A_set.append(np.array(A_opst))
        B_set.append(np.array(B_opst))
        new_ep = high_eq - low_eq
        if ep <= new_ep:
            flag_c = 1
            break
        ep = new_ep
        turns += 1
    if flag_c and ep > epsilon:
        print('no converge!',ep,flag_c)
    return A_opst, B_opst, high_eq, low_eq, A_response, B_response


def turn(R_matrix):#add a new strategy vector into current strategy set
    A_set, B_set = [],[]
    num_rows, num_columns = R_matrix.shape
    for i in range(num_rows):
        one_hot_vector = np.zeros(num_rows)
        position = i  
        one_hot_vector[position] = 1
        A_set.append(one_hot_vector)
    for j in range(num_columns):
        one_hot_vector = np.zeros(num_columns)
        position = j  
        one_hot_vector[position] = 1
        B_set.append(one_hot_vector)
    A_stra, A_res, high_eq, A_stat, A_responses = [],[],-float('inf'), [], []
    for i in A_set:
        A_stra.append(np.dot(i, R_matrix))
    B_stra, B_res, low_eq, B_stat, B_responses = [],[],float('inf'), [], []
    for i in B_set:
        B_stra.append(np.dot(R_matrix, i.transpose()))
    for i in range(len(A_stra)):
        problem = pulp.LpProblem(f"find_low_eq_{i}", pulp.LpMinimize)
        variables = [pulp.LpVariable(f'b{j}', lowBound=0, upBound=1) for j in range(len(B_set[0]))]
        problem += pulp.lpDot(A_stra[i], variables)
        problem += sum(variables) == 1
        for j in range(len(A_stra)):
            if j != i:
                problem += pulp.lpDot(A_stra[j], variables) - pulp.lpDot(A_stra[i], variables) <= 0
        problem.solve()
        Status = pulp.LpStatus[problem.status]
        A_stat.append(Status)
        if Status == 'Optimal':
            if low_eq > pulp.value(np.dot(A_stra[i], list(variables))):
                B_res = np.array([pulp.value(k) for k in variables])
                low_eq = pulp.value(np.dot(A_stra[i], list(variables)))
                A_pos = np.where(R_matrix.transpose()[0] == A_stra[i][0])[0][0]
                A_response = np.zeros_like(A_set[0])
                A_response[A_pos] = 1
                A_responses = [A_response]
            #elif low_eq <= 0.01 + pulp.value(np.dot(A_stra[i], list(variables))):
             #   A_pos = np.where(R_matrix.transpose()[0] == A_stra[i][0])[0][0]
              #  A_response = np.zeros_like(A_set[0])
               # A_response[A_pos] = 1
                #A_responses.append(A_response)
    #if not B_res:
     #   return 
    for i in range(len(B_stra)):
        problem = pulp.LpProblem(f"find_high_eq_{i}", pulp.LpMaximize)
        variables = [pulp.LpVariable(f'a{j}', lowBound=0, upBound=1) for j in range(len(A_set[0]))]
        problem += pulp.lpDot(B_stra[i], variables)
        problem += sum(variables) == 1
        for j in range(len(B_stra)):
            if j != i:
                problem += pulp.lpDot(B_stra[j], variables) - pulp.lpDot(B_stra[i], variables) >= 0
        problem.solve()
        Status = pulp.LpStatus[problem.status]
        B_stat.append(Status)
        if Status == 'Optimal':
            if high_eq < pulp.value(np.dot(B_stra[i], list(variables))):
                A_res = np.array([pulp.value(k) for k in variables])
                high_eq = pulp.value(np.dot(B_stra[i], list(variables)))
                B_pos = np.where(R_matrix[0] == B_stra[i][0])[0][0]
                B_response = np.zeros_like(B_set[0])
                B_response[B_pos] = 1
                B_response = B_response.transpose()
                B_responses = [B_response]
            #elif low_eq >= -0.01 + pulp.value(np.dot(B_stra[i], list(variables))):
             #   B_pos = np.where(R_matrix[0] == B_stra[i][0])[0][0]
              #  B_response = np.zeros_like(B_set[0])
               # B_response[B_pos] = 1
                #B_responses.append(B_response)
    #if not A_res:
     #   return 
    return A_res, B_res, high_eq, low_eq, A_stat, B_stat, A_responses, B_responses

R_matrix = np.random.randn(2,2)

A_opst, B_opst, high_eq, low_eq, A_stat, B_stat, A_responses, B_responses = turn(R_matrix)
R_matrix
A_opst
B_opst
high_eq
low_eq
A_response = A_responses[0]
B_response = B_responses[0]
A_opst @ R_matrix
R_matrix @ B_opst
#for A_response in A_responses:
 #   print(A_response @ R_matrix @ B_opst)

for i in range(A_opst.size):
    if A_opst[i] > 0.01:
        A_sp = np.zeros_like(A_opst)
        A_sp[i] = 1
        print(A_sp @ R_matrix @ B_opst)

for i in range(B_opst.size):
    if B_opst[i] > 0.01:
        B_sp = np.zeros_like(B_opst)
        B_sp[i] = 1
        print(A_opst @ R_matrix @ B_sp)

A_response @ R_matrix @ B_response
A_opst @ R_matrix @ B_opst
lmd = 0.5
A_lmd = lmd*A_opst + (1-lmd)*A_response
B_lmd = (1-lmd)*B_opst + lmd*B_response
print(A_lmd @ R_matrix @ B_lmd)

# Define the parameters for the plane
a, b = R_matrix[0]  # example coefficients
a1,b1 = R_matrix[1]
point = np.append(B_opst, high_eq)
# Create a grid of x and y values
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

# Create a meshgrid from x and y values
X, Y = np.meshgrid(x, y)

# Calculate corresponding z values for the plane
Z = a * X + b * Y
Z1 = a1 * X + b1 * Y
Y_intersection = (a1 * X[0] - a * X[0]) / (b - b1)
Z_intersection = a * X[0] + b * Y_intersection  

# Plot the plane within the specified bounds
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, color='blue', rstride=100, cstride=100)
ax.plot_surface(X, Y, Z1, alpha=0.5, color='red', rstride=100, cstride=100)
ax.scatter([point[0]], [point[1]], [point[2]], color='yellow', s=50)
# Plot the line of intersection
ax.plot(X[0], Y_intersection, Z_intersection, color='green', linewidth=3)

# Set limits for x, y, z axes
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Labels for axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
Y_intersection
(R_matrix[1][0]-R_matrix[0][0])/(R_matrix[0][1]-R_matrix[1][1])
(R_matrix[1][1]*R_matrix[0][0]-R_matrix[1][0]/R_matrix[0][1])/(R_matrix[1][1]+R_matrix[0][0]-R_matrix[1][0]-R_matrix[0][1])