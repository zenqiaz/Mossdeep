import pulp
import numpy as np
import time
import sys
import json
import nashpy as nash
import matplotlib.pyplot as plt
from winrate_matrix_str2 import winrate_matrix
from strategy_matrix import strategy_matrix
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
    return A_res, B_res, high_eq


class chan_fight:
    mach_low = 18
    mach_high = 21
    focus_low = 67
    focus_high = 79
    upper_low = 29
    upper_high = 35
    def __init__(self):
        self.winrate_matrix = [[0 for i in range(101)] for j in range(101)]
        self.strategy_matrix = [[0 for i in range(101)] for j in range(101)]
        mach_low = 18
        mach_high = 21
        focus_low = 67
        focus_high = 79
        upper_low = 29
        upper_high = 35
        start_time = time.time()
        #for i in range(1,30):
         #   for j in range(1,30):
          #      self.strategy_matrix[i][j], self.winrate_matrix[i][j] = self.DP(i,j)
        #for i in range(1,101):
         #   for j in range(1,101):
          #      self.strategy_matrix[i][j], self.winrate_matrix[i][j] = self.DP(i,j)
        end_time = time.time()
        self.runtime = end_time - start_time#3033 for 1st trail, 3504 for 2nd
    
    def winrate(self, a, b):
        if a <= 0:
            return -100
        if b <= 0:
            return 100
        else:
            return self.winrate_matrix[a][b]
    
    def average_winrate(self, a, b, highdam, lowdam, winner):
        l = highdam-lowdam+1
        res = 0
        if winner == 'a':
            for i in range(l):
                res += self.winrate(a,b-i-lowdam)
        if winner == 'b':
            for i in range(l):
                res += self.winrate(a-i-lowdam,b)
        return res/l
    
    def speed_roll(self, a, b, a_high, a_low, b_high, b_low, winner):
        a_range = a_high - a_low + 1
        b_range = b_high - b_low + 1
        res = 0
        for i in range(a-a_high, a-a_low+1):
            for j in range(b-b_high, b-b_low+1):
                if winner == 'a' and j <= 0:
                    res += 100
                elif winner == 'b' and i <= 0:
                    res -= 100
                else:
                    res += self.winrate(i,j)
        return res/(a_range*b_range)
                    
    
    def solve_P_matrix(self, R_matrix):
        A=R_matrix
        #print(R_matrix, hp_a, hp_b,strategy_matrix[hp_a][hp_b])
        B=-A
        game = nash.Game(A, B)
        equilibria = game.support_enumeration()
        equilibria_list = list(equilibria)
        return equilibria_list
    
    def DP(self,i,j):#0: mach punch; 1: focus punch; 2: upper hand
        res = [[0,0,0],[0,0,0],[0,0,0]]
        res[0][0] = (self.speed_roll(i,j, self.mach_high, self.mach_low, self.mach_high, self.mach_low, winner = 'a')+self.speed_roll(i,j, self.mach_high, self.mach_low, self.mach_high, self.mach_low, winner = 'b'))/2
        res[0][1] = self.average_winrate(i,j, self.mach_high, self.mach_low, winner = 'a')
        res[0][2] = self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'b')
        res[1][0] = self.average_winrate(i,j, self.mach_high, self.mach_low, winner = 'b')
        res[1][1] = (self.average_winrate(i,j, self.focus_high, self.focus_low, self.focus_high, self.focus_low, winner = 'a')+self.average_winrate(i,j, self.focus_high, self.focus_low, self.focus_high, self.focus_low, winner = 'b'))/2
        res[1][2] = self.average_winrate(i,j, self.focus_high, self.focus_low, winner = 'a')
        res[2][0] = self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'a')
        res[2][1] = self.average_winrate(i,j, self.focus_high, self.focus_low, winner = 'b')
        res[2][2] = (self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'b')+self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'a'))/2
        R_matrix = np.array(res)
        A_opst, B_opst = self.solve_P_matrix(R_matrix)
        high_eq = A_opst @ R_matrix @ B_opst
        #print(R_matrix)
        return (A_opst, B_opst), high_eq#, R_matrix
    
    def DP_show_matrix(self,i,j):#0: mach punch; 1: focus punch; 2: upper hand
        res = [[0,0,0],[0,0,0],[0,0,0]]
        res[0][0] = (self.speed_roll(i,j, self.mach_high, self.mach_low, self.mach_high, self.mach_low, winner = 'a')+self.speed_roll(i,j, self.mach_high, self.mach_low, self.mach_high, self.mach_low, winner = 'b'))/2
        res[0][1] = self.average_winrate(i,j, self.mach_high, self.mach_low, winner = 'a')
        res[0][2] = self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'b')
        res[1][0] = self.average_winrate(i,j, self.mach_high, self.mach_low, winner = 'b')
        res[1][1] = (self.average_winrate(i,j, self.focus_high, self.focus_low, self.focus_high, self.focus_low, winner = 'a')+self.average_winrate(i,j, self.focus_high, self.focus_low, self.focus_high, self.focus_low, winner = 'b'))/2
        res[1][2] = self.average_winrate(i,j, self.focus_high, self.focus_low, winner = 'a')
        res[2][0] = self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'a')
        res[2][1] = self.average_winrate(i,j, self.focus_high, self.focus_low, winner = 'b')
        res[2][2] = (self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'b')+self.average_winrate(i,j, self.upper_high, self.upper_low, winner = 'a'))/2
        R_matrix = np.array(res)
        A_opst, B_opst = self.solve_P_matrix(R_matrix)
        high_eq = A_opst @ R_matrix @ B_opst
        #print(R_matrix)
        return (A_opst, B_opst), high_eq, R_matrix




fight = chan_fight()
fight.strategy_matrix[50][50]
fight.winrate_matrix[28][28]



i,j = 50,50
fight.strategy_matrix[i][j], fight.winrate_matrix[i][j],R_matrix = fight.DP_show_matrix(i,j)
fight.strategy_matrix[i][j]#(array([0.53082899, 0.30156925, 0.16760177]), array([0.53082899, 0.30156924, 0.16760177])) for 100,100
fight.winrate_matrix[i][j]

(A_opst, B_opst), high_eq, R_matrix = fight.DP_show_matrix(80,61)





fight.winrate_matrix_list = fight.winrate_matrix.tolist()

# Convert the variable to a JSON string
winrate_matrix_str = json.dumps(fight2.winrate_matrix)

converted_nested_list = [
    [(arr1[0].tolist(), arr1[1].tolist()) for arr1 in sublist]
    for sublist in backup_s_matrix
]
for i in converted_nested_list:
    i.insert(0,0)

converted_nested_list.insert(0,0)
# Serialize the list of tuples of lists to JSON
json_data = json.dumps(converted_nested_list)


# Save the string to a file
with open("strategy_matrix.txt", "w") as file:
    file.write(json_data)

                                             
with open("winrate_matrix_str", "w") as file:
    file.write(winrate_matrix_str)
for row in fight.winrate_matrix:
    row.append(0)

# Add a new row of 101 zeros at the end of the nested list
for col in 
new_row = [0 for _ in range(101)]
fight.winrate_matrix.append(new_row)
A_opst, B_opst, high_eq = turn(np.array(winrate_matrix_100)/2-1)
A_opst, B_opst, high_eq = turn(np.array(winrate_matrix_sim))
for i in range(1,101):
    print(i,strategy_matrix[i][i])


for i in range(1,101):
    for j in range(1,101):
        if i*j == 0:
            backup_s_matrix[i][j] = [array([0.53082899, 0.30156925, 0.16760177]), array([0.53082899, 0.30156924, 0.16760177])]

for i in range(1,101):
    for j in range(1,101):
        fight.strategy_matrix[i][j], fight.winrate_matrix[i][j] = fight.DP(i,j)