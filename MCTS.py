import random
import pulp
import numpy as np
import time
import sys
import math
import json
import matplotlib.pyplot as plt
from winrate_matrix_str2 import winrate_matrix
from strategy_matrix import strategy_matrix
# Create a LP problem object
mach_low = 18
mach_high = 21
focus_low = 67
focus_high = 79
upper_low = 29
upper_high = 35
winrate_matrix_100 = [[0.5,0.62,0.28],[0.38,0.5,0.88],[0.72,0.12,0.5]]
winrate_matrix_sim = [[0,20,-32],[-20,0,73],[32,-73,0]]
epsilon = 0.001

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


def move_value(high_dam, low_dam, hp, KO_value = 1000):
    if hp <= low_dam:
        return KO_value
    if hp >= high_dam:
        return (high_dam+low_dam)/2
    return ((high_dam -hp)*KO_value+(hp-low_dam)*(hp+low_dam)/2)/(high_dam-low_dam)


def resolve(hp_a, hp_b, A_move, B_move):
    speed_roll = random.randint(0,1)
    if A_move == 0 and B_move == 0:
        if speed_roll == 0:
            hp_b -= random.randint(mach_low, mach_high)
            if hp_b > 0:
                hp_a -= random.randint(mach_low, mach_high)
        if speed_roll == 1:
            hp_a -= random.randint(mach_low, mach_high)
            if hp_a > 0:
                hp_b -= random.randint(mach_low, mach_high)
    elif A_move == 0 and B_move == 1:
        hp_b -= random.randint(mach_low, mach_high)
    elif A_move == 0 and B_move == 2:
        hp_a -= random.randint(upper_low, upper_high)
    elif A_move == 1 and B_move == 0:
        hp_a -= random.randint(mach_low, mach_high)
    elif A_move == 1 and B_move == 1:
        if speed_roll == 0:
            hp_b -= random.randint(focus_low, focus_high)
        if speed_roll == 1:
            hp_a -= random.randint(focus_low, focus_high)
    elif A_move == 1 and B_move == 2:
        hp_b -= random.randint(focus_low, focus_low)
    elif A_move == 2 and B_move == 0:
        hp_b -= random.randint(upper_low, upper_high)
    elif A_move == 2 and B_move == 1:
        hp_a -= random.randint(focus_low, focus_low)
    elif A_move == 2 and B_move == 2:
        if speed_roll == 0:
            hp_b -= random.randint(upper_low, upper_high)
        if speed_roll == 1:
            hp_a -= random.randint(upper_low, upper_high)
    return hp_a, hp_b


def choose_move(strategy):
    draw = random.random()
    thershhold = [strategy[0], strategy[0] + strategy[1]]
    if draw < thershhold[0]:
        move = 0
    elif  thershhold[1] > draw >= thershhold[0]:
        move = 1
    else:
        move = 2
    return move


def sim_AI(hp_a, hp_b, KO_value = 1000):
    res = [[0,0,0],[0,0,0],[0,0,0]]
    res[0][0] = (-move_value(mach_high, mach_low, hp_a, KO_value)+move_value(mach_high, mach_low, hp_b, KO_value))/2
    res[0][1] = move_value(mach_high, mach_low, hp_b, KO_value)
    res[0][2] = -move_value(upper_high, upper_low, hp_a, KO_value)
    res[1][0] = -move_value(mach_high, mach_low, hp_a, KO_value)
    res[1][1] = (-move_value(focus_high, focus_low, hp_a, KO_value)+move_value(focus_high, focus_low, hp_b, KO_value))
    res[1][2] = move_value(focus_high, focus_low, hp_b, KO_value)
    res[2][0] = move_value(upper_high, upper_low, hp_b, KO_value)
    res[2][1] = -move_value(focus_high, focus_low, hp_a, KO_value)
    res[2][2] = (-move_value(upper_high, upper_low, hp_a, KO_value)+move_value(upper_high, upper_low, hp_b, KO_value))/2
    R_matrix = np.array(res)
    A_opst, B_opst, high_eq = turn(R_matrix)
    return(A_opst, B_opst)


def run_turn(hp_a, hp_b, KO_value = 1000):
    A_opst, B_opst = sim_AI(hp_a, hp_b, KO_value)
    A_thershhold = [A_opst[0], A_opst[0] + A_opst[1]]
    B_thershhold = [B_opst[0], B_opst[0] + B_opst[1]]
    A_draw = random.random()
    if A_draw < A_thershhold[0]:
        A_move = 0
    elif  A_thershhold[1] > A_draw >= A_thershhold[0]:
        A_move = 1
    else:
        A_move = 2
    B_draw = random.random()
    if B_draw < B_thershhold[0]:
        B_move = 0
    elif  B_thershhold[1] > B_draw >= B_thershhold[0]:
        B_move = 1
    else:
        B_move = 2
    #print(A_move, B_move)
    return resolve(hp_a, hp_b, A_move, B_move)


def trail(hp_a, hp_b, KO_value = 1000):
    while hp_a > 0 and hp_b > 0:
        hp_a, hp_b = run_turn(hp_a, hp_b, KO_value = 1000)
    if hp_a <= 0:
        return 'b'
    elif hp_b <= 0:
        return 'a'
    return


def cross_entropy(p,q):
    return -(p*math.log(p)+(1-p)*math.log(1-q+epsilon))


def fight_MCTS(hp_a, hp_b, KO_value = 1000, trails = 5):
    mach_low = 18
    mach_high = 21
    focus_low = 67
    focus_high = 79
    upper_low = 29
    upper_high = 35
    res_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    res_matrixs = []
    start_time = time.time()
    A_count = [[0,0,0],[0,0,0],[0,0,0]]
    CE = []
    for k in range(trails):
        for i in range(3):
            for j in range(3):
                A_move, B_move = i,j
                trail_a_hp, trail_b_hp = resolve(hp_a, hp_b, A_move, B_move)
                winner = trail(trail_a_hp, trail_b_hp, KO_value = 1000)
                if winner == 'a':
                    A_count[i][j] += 1
        current_winrate = np.array(A_count)/(1+k)
        current_CE = 0
        for i in range(3):
            for j in range(3):
                current_CE += cross_entropy(winrate_matrix_100[i][j],current_winrate[i][j])
        CE.append(current_CE)
        #if k%5 == 4:
         #   res_matrixs.append(current_winrate)
    R_matrix = np.array(A_count)/(1+k)
    A_opst, B_opst, high_eq = turn(R_matrix)
    end_time = time.time()
    runtime = end_time - start_time
    return A_opst, B_opst, high_eq, R_matrix, runtime, CE


def choose_MCTS(hp_a, hp_b, KO_value = 1000, trails = 5):
    mach_low = 18
    mach_high = 21
    focus_low = 67
    focus_high = 79
    upper_low = 29
    upper_high = 35
    res_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    res_matrixs = []
    A_count = [[0,0,0],[0,0,0],[0,0,0]]
    CE = []
    for k in range(4):
        for i in range(3):
            for j in range(3):
                A_move, B_move = i,j
                trail_a_hp, trail_b_hp = resolve(hp_a, hp_b, A_move, B_move)
                winner = trail(trail_a_hp, trail_b_hp, KO_value = 1000)
                if winner == 'a':
                    A_count[i][j] += 1
        current_winrate = np.array(A_count)/(1+k)
        current_CE = 0
        for i in range(3):
            for j in range(3):
                current_CE += cross_entropy(winrate_matrix_100[i][j],current_winrate[i][j])
        CE.append(current_CE)
        #if k%5 == 4:
         #   res_matrixs.append(current_winrate)
    R_matrix = np.array(A_count)/(1+k)
    A_opst, B_opst, high_eq = turn(R_matrix)
    return A_opst


def sim_battle(AI_for_A = 'random', trails = 5):
    A_count = 0
    start_time = time.time()
    for i in range(trails):
        hp_a, hp_b = 100,100
        A_strategy, B_strategy = [0.333,0.333,0.334], [0.333,0.333,0.334]
        while hp_a > 0 and hp_b > 0:
            if AI_for_A == 'MCTS':#56s per run, 10/20
                A_strategy = choose_MCTS(hp_a, hp_b, KO_value = 1000)
            elif AI_for_A == 'sim':#1.2s per run 52/100 42/100
                A_strategy, _ = sim_AI(hp_a, hp_b, KO_value = 1000)
            elif AI_for_A == 'optimal':
                A_strategy = strategy_matrix[hp_a][hp_b][0]
            else:
                A_strategy = [0.333,0.333,0.334]#winrate: 452/1000
            B_strategy = strategy_matrix[hp_a][hp_b][1]
            hp_a, hp_b = resolve(hp_a, hp_b, choose_move(A_strategy), choose_move(B_strategy))
        if hp_a <= 0:
            continue
        elif hp_b <= 0:
            A_count += 1
            continue
    end_time = time.time()
    runtime = end_time - start_time
    return A_count/trails, runtime

#if name == '__main__':
hp_a, hp_b = 50,50
A_opst, B_opst, high_eq, R_matrix, runtime, CE = fight_MCTS(hp_a, hp_b, KO_value = 300, trails = 10)
print(A_opst, B_opst, high_eq, R_matrix, runtime, CE)

A_winrate, runtime = sim_battle(AI_for_A = 'sim', trails = 100)
#49s for 5 trails, 
#[0.50862069 0.22413793 0.26724138] [0.48275862 0.3362069  0.18103448] 0.5129310349999999 [[0.5 0.7 0.2][0.2 0.7 1. ] for 50