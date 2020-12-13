"""temporal_discounting controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor


import random
import numpy as np
import sys
import pdb
import math
import time


from controller import Robot, Motor, DistanceSensor, Compass, Supervisor, Node, Field

# create the Robot instance.

robot = Supervisor()

# degrees for left, right, up, down
head_directions = [90, 270, 0, 180]

desired_direction = 1


# set random seed for reproducibility
np.random.seed(5)


# get the time step of the current world.
timestep = 2000

# Initialize devices
ps = []
psNames = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 
            'ps6', 'ps7']
            
for i in range(8):
    ps.append(robot.getDistanceSensor(psNames[i]))
    ps[i].enable(timestep)
    
cmpXY1 = robot.createCompass("compassXY_01")
cmpXY1.enable(timestep)
cmpZ1 = robot.createCompass("compassZ_01")
cmpZ1.enable(timestep)    

    
# Initialize motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
leftMotor.setVelocity(0.0)

# Configuring parameters

t = 0
turning = False
dir = 0 
vpre = 0 
trial = 0 
tLatency = 0 
found = False
HD_RES = 5.0#1.0
TURN_RATE = 100
GAIN = 2.5
threshold = 0.5#0.25 #0.4#0.25
max_speed = 2#6.28
radius = 1
alpha = 0.5
time_unit = 10

forward = True

# do this once only

startingTranslation = [[0.7, 0, -4],
                        [-0.7, 0, -4]]
                        
# startingTranslation = [[8.15, 0.125, 0.4],
                        # [8.15, 0.125, 0.4]] # Testing

startingRotation = [0, 1, 0, 3]


root = robot.getRoot()
root_children_field = root.getField("children") 


node = root_children_field.getMFNode(-1) 
trans = node.getField("translation")
rot = node.getField("rotation") 




# Functions relevant to task


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Getting distance
def getDistance(x1, y1, x2, y2): 
    x = (x1-x2)**2
    y = (y1-y2)**2
    return (x+y)**0.5
    
# Temporal discounting



def modulated_impulsivity(kval, hunger):
    return kval * hunger
    

def discounted_value(kval, delay, value):
    DV = 1 / (1+ (kval * delay))
    discounted_value = DV * value
    return discounted_value
    
# Upper confidence bound (UCB) for exploration
def UCB(t, LLSS_values, Nt, c, a):
    # c = 2 
    # t = trial
    # LLSS_Nt = [0,0]
    return LLSS_values[a] + c * np.sqrt(np.log(t + 1) / (Nt[a] + 1)) # add 1 to avoid division by zero

    
# Reward prediction error
def RPE(alpha, previous_value, actual_reward):
    current_value = previous_value + alpha * (actual_reward - previous_value)
    return current_value


def motivation(value, hunger):
    effort = sigmoid(value / hunger)
    return effort

satiety = 10 #[1, 5, 10] # hunger = 10 / satiety
kval = 0.01 #[0.01, 0.05, 0.1]

# red apple (LL) = 100, green apple (SS) = 50


LL_translation = [0.7, 0.2, 5.3]
SS_translation = [-0.7, 0.2, 0.5]

LLSS_translations = [LL_translation, SS_translation]


# LL_value = 0
# SS_value = 0
LLSS_values = [1e-3,1e-3] # LL, SS integrated ; avoid zero?
LLSS_Nt = [0,0] 
conditions = ['LL', 'SS']

LL_magnitude = 100
SS_magnitude = 75


rot.setSFRotation(startingRotation)
if random.random() < 0.5: 
    random_start = 0
else: 
    random_start = 1

LLSS_Nt[random_start] += 1 # because randomly chosen at first

trans.setSFVec3f(startingTranslation[random_start])

LL_table = []
SS_table = []

# Main loop:

while robot.step(timestep) != -1:
    time0 = time.time()

    # current trial's satiety and hunger: 
    satiety *= 0.9**trial
    hunger = 10 / satiety

    # calculate motivation to determine baseline speed
    if trial == 0:
        effort = motivation(LLSS_values[random_start], hunger)
    else:
        effort = motivation(LLSS_values[next], hunger)
    
    baseline_speed = effort * max_speed
    
    
    
    
    #init speeds
    left_speed = 1 #1#max_speed * 0.5
    right_speed = 1#max_speed * 0.5
    
    

    
    # calculate k-value modulated by current level of hunger
    mk = modulated_impulsivity(kval, hunger)
    
    
    
    psValues=[]
    for i in range(8):
        psValues.append(ps[i].getValue())  


    # Check if the robot hit the wall
    
    left_obstacle = psValues[1] < threshold or psValues[0] < threshold
    right_obstacle = psValues[3] < threshold or psValues[4] < threshold
    front_obstacle = psValues[2] < threshold
 
    

    
    trans_value = trans.getSFVec3f()
    # print(trans_value)
 



   # read compass
    cmpXY = cmpXY1.getValues()
    cmpZ = cmpZ1.getValues()

    
    # calculate bearing
    rad = math.atan2(cmpXY[0], cmpZ[2])
    bearing = rad / 3.1428 * 180.0
    if bearing < 0.0: 
        bearing = bearing + 360.0

    heading_error = head_directions[desired_direction] - bearing
    if abs(heading_error) < HD_RES: 
        # print('a')
        # turning = False
        t=1
        forward = True
        # t += 1
    elif head_directions[desired_direction] == 0 and bearing > 355.0: 
        # print('b')
        # turning = False
        forward = True
        t=1
        # t += 1
    elif heading_error > 0: 
        # turning= True
        forward = False
        if abs(heading_error) < 180.0:
            # print('c') 
            left_speed += 0.5 #* max_speed
            right_speed -= 0.5 #* max_speed            
            # left_speed = 3.0
            # right_speed = -1.0
        else: 
            # print('d')
            left_speed -= 0.5 #* max_speed
            right_speed += 0.5 #* max_speed 
            # left_speed = -1.0
            # right_speed = 3.0
    else: 
        # turning = True
        forward = False
        if abs(heading_error) < 180.0: 
            # turn left
            # print('e')
            left_speed -= 0.5 #* max_speed
            right_speed += 0.5 #* max_speed                 
            # left_speed = -1.0
            # right_speed = 3.0
        else: 
            # turn right
            # print('f')
            left_speed += 0.5 #* max_speed
            right_speed -= 0.5 #* max_speed   

    # check if the robot hit a wall
    if front_obstacle or left_obstacle or right_obstacle:
        forward = False
        t=1 
        # modify speeds according to obstacles
        if front_obstacle:
            # print('front_obstacle')
            left_speed = 0 
            right_speed = -1.0
    
            # print('front obstacle')
    
        elif left_obstacle: 
            # print('left_obstacle')
            # turn right
            left_speed += 0.5 #* max_speed
            right_speed -= 0.5 #* max_speed
    
        elif right_obstacle: 
            # print('right_obstacle')
            left_speed -= 0.5 #* max_speed
            right_speed += 0.5 #* max_speed
    # elif turning: 
        # print('turning')
         # t += 1
     
    
        # pointing close to the desired heading 
        # print(bearing)
    # else: 
    else: 
        # left_speed = 2
        # right_speed = 2
        forward = True
        t += 1           
    
    
    if forward:
        left_speed = baseline_speed
        right_speed = baseline_speed
    



    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)
    
    # print(t)
    
    
    # if found or (not turning and (t % TURN_RATE == 0)):
        # turning = True



    distance_LL = getDistance(LL_translation[0], LL_translation[2], trans_value[0], trans_value[2])
    distance_SS = getDistance(SS_translation[0], SS_translation[2], trans_value[0], trans_value[2])


    
    # print(distance_LL)
    if distance_LL < radius:
        elapsed_time = time.time() - time0
        found = True #def discounted_value(kval, delay, value):
        TD_value = discounted_value(mk, elapsed_time/time_unit, LL_magnitude)
        LLSS_values[0] = RPE(alpha, LLSS_values[0], TD_value)
        LL_table.append(LLSS_values[0])
    elif distance_SS < radius: 
        elapsed_time = time.time() - time0
        found = True
        TD_value = discounted_value(mk, elapsed_time/time_unit, SS_magnitude)
        LLSS_values[1] = RPE(alpha, LLSS_values[1], TD_value)
        SS_table.append(LLSS_values[1])


  
   

 
    if found: 
        # turning = True
        # set new starting position for the robot
        
        
        next = np.argmax([UCB(trial, LLSS_values, LLSS_Nt, 2, 0),
                            UCB(trial, LLSS_values, LLSS_Nt, 2, 1)]) 
            
        LLSS_Nt[next] += 1
        print("Choose {}: value of {}".format(conditions[next], LLSS_values[next]))         
            
        
        
        trans.setSFVec3f(startingTranslation[next])
        rot.setSFRotation(startingRotation)
        trial += 1
        found = False




    if trial > 100:
        break    



