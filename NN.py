import torch
import torch.nn as nn
import random
#import pulp
import numpy as np
import time
import sys
import math
import json
import nashpy as nash
import matplotlib.pyplot as plt
from winrate_matrix_str2 import winrate_matrix
from strategy_matrix import strategy_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# Defining the MLP network
class value_MLP(nn.Module):
    def __init__(self):
        super(value_MLP, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return x

def convert_to_dataset(winrate_matrix):
    res = []
    for i in range(1,101):
        for j in range(1,101):
            res.append([[i/100,j/100], (winrate_matrix[i][j]+100)/200])
    return res

winrate_dataset = convert_to_dataset(winrate_matrix)



# Assuming `dataset` is a list of tuples ((a, b, c, d), label)
# Convert your dataset to a PyTorch TensorDataset
inputs, labels = zip(*winrate_dataset)
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
tensor_dataset = TensorDataset(inputs, labels)

# DataLoader
train_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)
value_network = value_MLP()
# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 80
training_losses = []
# Training Loop
for epoch in range(num_epochs):
    curtime = time.time()
    epoch_loss = 0.0
    num_batches = 0
    for inputs, labels in train_loader:
        # Forward pass
        outputs = value_network(inputs)
        loss = criterion(outputs.squeeze(), labels)
    
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        #for name, param in value_network.named_parameters():
         #   if param.requires_grad:
          #      print(name, param.grad)
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
        #print('current loss:', loss.item())
    print('runtime for this epoch:', time.time()-curtime)
    average_loss = epoch_loss / num_batches
    training_losses.append(average_loss)
    print('average loss for this epoch:', average_loss)
    #scheduler.step()
    # Optionally evaluate on validation set here

plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # Setting the y-axis to logarithmic scale
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


winrate_dataset[1267]
input_tensor = torch.tensor([0.13, 0.68], dtype=torch.float32).unsqueeze(0)
output = value_network(input_tensor)
output
torch.save(value_network.state_dict(), 'D:/thesis/value_network_ref.pth')
torch.tensor([100,1], dtype=torch.float32)


for name, param in value_network.named_parameters():
    print(f"{name}: {param.size()}")
    print(param.data



class policy_MLP(nn.Module):
    def __init__(self):
        super(value_MLP, self).__init__()
        self.layer1 = nn.Linear(4, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return x

def winrate(a, b):
    if a <= 0:
        return -100
    if b <= 0:
        return 100
    else:
        return winrate_matrix[a][b]


def average_winrate(a, b, highdam, lowdam, winner):
    l = highdam-lowdam+1
    res = 0
    if winner == 'a':
        for i in range(l):
            res += winrate(a,b-i-lowdam)
    if winner == 'b':
        for i in range(l):
            res += winrate(a-i-lowdam,b)
    return res/l


def speed_roll(a, b, a_high, a_low, b_high, b_low, winner):
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
                res += winrate(i,j)
    return res/(a_range*b_range)

def winrate_eva(hp_a, hp_b, k):#k means which profile is being checked
    ref_win = [0,[21,18,'a'], [35,29,'b'], [21,18,'b'], 0, [79,67,'a'], [35,29,'a'], [79,67,'b'], 0]
    ref_roll = [[21,18], [79,67], [35,29]]
    if k == 0:
        rate = (speed_roll(hp_a, hp_b, ref_roll[0][0], ref_roll[0][1], ref_roll[0][0], ref_roll[0][1], 'a')+speed_roll(hp_a, hp_b, ref_roll[0][0], ref_roll[0][1], ref_roll[0][0], ref_roll[0][1], 'b')+200)/400
    elif k % 4 == 0:
        l = k//4
        rng = random.randint(0,1)
        if rng:
            rate = (average_winrate(hp_a, hp_b, ref_roll[l][0], ref_roll[l][1], 'a')+100)/200
        else:
            rate = (average_winrate(hp_a, hp_b, ref_roll[l][0], ref_roll[l][1], 'b')+100)/200
    else:
        rate = (average_winrate(hp_a, hp_b, ref_win[k][0], ref_win[k][1], ref_win[k][2])+100)/200
    return rate

def dam_eva(k):
    ref_dam = [[21,21], [21,0], [0,35], [0,21], [21,0], [79,0], [35,0], [0,79], [21,0]]
    ref_win = [0,[21,18,'a'], [35,29,'b'], [21,18,'b'], 0, [79,67,'a'], [35,29,'a'], [79,67,'b'], 0]
    ref_roll = [[21,18], [79,67], [35,29]]
    if k == 0:
        dam_a, dam_b = 0.21,0.21
    elif k % 4 == 0:
        l = k//4
        rng = random.randint(0,1)
        if rng:
            dam_a, dam_b = ref_roll[l][0]/100,0
        else:
            dam_a, dam_b = 0,ref_roll[l][0]/100
    else:
        dam = [0,0]
        if ref_win[k][2] == 'a':
            dam[0] = ref_win[k][0]
        else:
            dam[1] = ref_win[k][0]
        dam_a, dam_b = dam
        dam_a, dam_b = dam_a/100, dam_b/100
    return dam_a, dam_b



def policy_sample(winrate_matrix, winrate_dataset, batch_size):
    ref_win = [0,[21,18,'a'], [35,29,'b'], [21,18,'b'], 0, [79,67,'a'], [35,29,'a'], [79,67,'b'], 0]
    ref_roll = [[21,18], [79,67], [35,29]]
    ref_dam = [[21,21], [21,0], [0,35], [0,21], [21,0], [79,0], [35,0], [0,79], [21,0]]
    random.shuffle(winrate_dataset)
    cur = 0
    batches = []
    #start_time = time.time()
    for i in range(10000//batch_size):
        batch = []
        for j in range(batch_size//9):
            for k in range(9):
                hp_a, hp_b =  winrate_dataset[cur][0]
                hp_a, hp_b = int(hp_a*100), int(hp_b*100)
                rate = winrate_eva(hp_a, hp_b, k)
                dam_a, dam_b = dam_eva(k)
                rng = random.random()
                if rng <= rate:
                    win = 1
                else:
                    win = 0
                batch.append([[hp_a/100, hp_b/100, dam_a, dam_b],win])
                cur += 1
        batches.append(batch)
    #print('runtime:', time.time()-start_time)
    return batches



def policy_validate(policy_network):
    epoch_loss = 0
    for i in range(10):
        hp_a, hp_b = random.randint(1,100),random.randint(1,100)
        k = random.randint(0,8)
        dam_a, dam_b = dam_eva(k)
        model_wr = policy_network(np.array([hp_a/100, hp_b/100, dam_a, dam_b]))
        optimal_wr = winrate_eva(hp_a, hp_b, k)
        epoch_loss+=(model_wr-optimal_wr)**2
    average_loss = epoch_loss / 10
    training_losses.append(average_loss)
    print('loss:', average_loss)
    return

def cosine_similarity(array1, array2):
    dot_product = np.dot(array1, array2)
    norm_array1 = np.linalg.norm(array1)
    norm_array2 = np.linalg.norm(array2)
    similarity = dot_product / (norm_array1 * norm_array2)
    return similarity

def strategy_validate(policy_network):
    similarity_sum = 0
    for i in range(10):
        hp_a, hp_b = random.randint(1,100),random.randint(1,100)
        R_matrix = np.array([[0,1,-1], [-1,0,1],[1,-1,0]])
        for i in range(9):
            dam_a, dam_b = dam_eva(i)
            R_matrix[i//3][i%3] = policy_network(np.array([hp_a/100, hp_b/100, dam_a, dam_b]))
        A=R_matrix
        game = nash.Game(A, B)
        equilibria = game.support_enumeration()
        equilibria_list = list(equilibria)
        optimal_strategy = np.array(strategy_matrix[hp_a][hp_b])
        similarity_sum += cosine_similarity(optimal_strategy[0], equilibria[0])*cosine_similarity(optimal_strategy[1], equilibria[1])
    average_similarity = similarity_sum/10
    training_similarity.append(average_similarity)
    print('similarity:', average_similarity)
    return
        

sample_batches = policy_sample(winrate_matrix, winrate_dataset, batch_size = 36)
policy_network = policy_MLP()
# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 20
training_losses = []
training_similarity = []
# Training Loop
for epoch in range(num_epochs):
    curtime = time.time()
    sample_batches = policy_sample(winrate_matrix, winrate_dataset, batch_size = 36)
    for inputs, labels in sample_batches:
        # Forward pass
        outputs = policy_network(inputs)
        loss = criterion(outputs.squeeze(), labels)
    
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        #for name, param in value_network.named_parameters():
         #   if param.requires_grad:
          #      print(name, param.grad)
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
        #print('current loss:', loss.item())
    policy_validate(policy_network)
    print('runtime for this epoch:', time.time()-curtime)
    
    
        
        
    
    #scheduler.step()
    # Optionally evaluate on validation set here
 
 
 
 