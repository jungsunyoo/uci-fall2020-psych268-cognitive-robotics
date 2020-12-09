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

desired_direction = 0


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
HD_RES = 1.0
TURN_RATE = 100
GAIN = 2.5
threshold = 0.5#0.25 #0.4#0.25
max_speed = 6.28
radius = 1
alpha = 0.5
time_unit = 10

# do this once only

startingTranslation = [[0.7, 0, -4],
                        [-0.7, 0, -4]]

startingRotation = [0, 1, 0, 3]


root = robot.getRoot()
root_children_field = root.getField("children") 


node = root_children_field.getMFNode(-1) 
trans = node.getField("translation")
rot = node.getField("rotation") 




# Functions relevant to task

# Getting distance
def getDistance(x1, y1, x2, y2): 
    x = (x1-x2)**2
    y = (y1-y2)**2
    return (x+y)**0.5
    
# Temporal discounting
def discounted_value(kval, delay, value):
    DV = 1 / (1+ (kval * delay))
    discounted_value = DV * value
    return discounted_value
    
# Reward prediction error
def RPE(alpha, previous_value, actual_reward):
    current_value = previous_value + alpha * (actual_reward - previous_value)
    return current_value

hunger_levels = [1, 10, 100]
kval = 0.01 #[0.01, 0.05, 0.1]

# red apple (LL) = 100, green apple (SS) = 50


LL_translation = [0.7, 0.2, 5.3]
SS_translation = [-0.7, 0.2, 0.5]

LLSS_translations = [LL_translation, SS_translation]


# LL_value = 0
# SS_value = 0
LLSS_values = [0,0] # LL, SS integrated


rot.setSFRotation(startingRotation)
if random.random() < 0.5: 
    trans.setSFVec3f(startingTranslation[0])
else: 
    trans.setSFVec3f(startingTranslation[1])


LL_table = []
SS_table = []

# Main loop:

while robot.step(timestep) != -1:
    time0 = time.time()




    psValues=[]
    for i in range(8):
        psValues.append(ps[i].getValue())  


    # Check if the robot hit the wall
    
    left_obstacle = psValues[1] < threshold or psValues[0] < threshold
    right_obstacle = psValues[3] < threshold or psValues[4] < threshold
    front_obstacle = psValues[2] < threshold
 
    
    #init speeds
    left_speed = 1#max_speed * 0.5
    right_speed = 1#max_speed * 0.5
    
    trans_value = trans.getSFVec3f()
    # print(trans_value)
    distance_LL = getDistance(LL_translation[0], LL_translation[2], trans_value[0], trans_value[2])
    distance_SS = getDistance(SS_translation[0], SS_translation[2], trans_value[0], trans_value[2])

    


    ofLeft = 0
    ofRight = 0 

    for i in range(4):
        ofRight += psValues[i]
        ofLeft += psValues[i+4]

    # check if the robot hit a wall
    if front_obstacle or left_obstacle or right_obstacle:
        t=1 
        # modify speeds according to obstacles
        if front_obstacle:
            print('front_obstacle')
            left_speed = 0 
            right_speed = -1.0
    
            # print('front obstacle')
    
        elif left_obstacle: 
            print('left_obstacle')
            # turn right
            left_speed += 0.5 * max_speed
            right_speed -= 0.5 * max_speed
    
        elif right_obstacle: 
            print('right_obstacle')
            left_speed -= 0.5 * max_speed
            right_speed += 0.5 * max_speed
    elif turning: 
        print('turning')
         # t += 1
        # read compass
        cmpXY = cmpXY1.getValues()
        cmpZ = cmpZ1.getValues()

        
        # calculate bearing
        rad = math.atan2(cmpXY[0], cmpZ[2])
        bearing = rad / 3.1428 * 180.0
        if bearing < 0.0: 
            bearing = bearing + 360.0
    
    
        # pointing close to the desired heading 
        print(bearing)
        heading_error = head_directions[desired_direction] - bearing
        if abs(heading_error) < HD_RES: 
            turning = False
            t=1
            # t += 1
        elif head_directions[desired_direction] == 0 and bearing > 355.0: 
            turning = False
            t=1
            # t += 1
        elif heading_error > 0: 
            if abs(heading_error) < 180.0: 
                left_speed += 0.5 * max_speed
                right_speed -= 0.5 * max_speed            
                # left_speed = 3.0
                # right_speed = -1.0
            else: 
                left_speed -= 0.5 * max_speed
                right_speed += 0.5 * max_speed 
                # left_speed = -1.0
                # right_speed = 3.0
        else: 
            if abs(heading_error) < 180.0: 
                # turn left
                left_speed -= 0.5 * max_speed
                right_speed += 0.5 * max_speed                 
                # left_speed = -1.0
                # right_speed = 3.0
            else: 
                # turn right
                left_speed += 0.5 * max_speed
                right_speed -= 0.5 * max_speed   
    # else: 
    else: 
        left_speed = 2
        right_speed = 2
        t += 1           

    velo = GAIN * ((ofLeft - ofRight)/(ofLeft + ofRight))

    # left_speed += velo * left_speed;
    # right_speed += -velo*right_speed;

    # if left_speed > max_speed: 
        # left_speed = max_speed
    # elif left_speed < 0: 
        # left_speed = 0 
    # if right_speed > max_speed: 
        # right_speed = max_speed
    # elif right_speed < 0: 
        # right_speed = 0 


    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)
    
    print(t)
    
    
    if found or (not turning and (t % TURN_RATE == 0)):
        turning = True
    
    # print(distance_LL)
    if distance_LL < radius:
        elapsed_time = time.time() - time0
        found = True #def discounted_value(kval, delay, value):
        TD_value = discounted_value(kval, elapsed_time/time_unit, 100)
        LLSS_values[0] = RPE(alpha, LLSS_values[0], TD_value)
        LL_table.append(LLSS_values[0])
    elif distance_SS < radius: 
        elapsed_time = time.time() - time0
        found = True
        TD_value = discounted_value(kval, elapsed_time/time_unit, 50)
        LLSS_values[1] = RPE(alpha, LLSS_values[1], TD_value)
        SS_table.append(LLSS_values[1])


    # Valuation: decide whether to go for LL or SS
    if LLSS_values[0] > LLSS_values[1]: 
        # go for LL for most of the time
        if random.random() < 0.7:
            next = 0
        else:
            next = 1
    elif LLSS_values[0] < LLSS_values[1]: 
        # go for SS for most of the time
        if random.random() < 0.7:
            next = 1
        else: 
            next = 0
        
    else: # if two values are equivalent
        if random.random() < 0.5: 
            next = 0
            # go for LL
        else:
            next = 1
            # go for SS

    if found: 
        turning = True
        # set new starting position for the robot
        trans.setSFVec3f(startingTranslation[next])
        rot.setSFRotation(startingRotation)
        trial += 1
        found = False




    if trial > 100:
        break    



