"""temporal_discounting controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor


import random
import numpy as np
import sys
import pdb
import math



from controller import Robot, Motor, DistanceSensor, Compass, Supervisor, Node, Field

# create the Robot instance.

robot = Supervisor()


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
threshold = 0.25 #0.4#0.25
max_speed = 6.28
radius = 1

def getDistance(x1, y1, x2, y2): 
    x = (x1-x2)**2
    y = (y1-y2)**2
    return (x+y)**0.5

# do this once only

startingTranslation = [[0.7, 0, -4]]
# [[1,0,2.5],
                       # [-1,0,2.5]]
                        #[1,0,2.5] #[0, 0, 2.25]
startingRotation = [0, 1, 0, -3]
# [[0, 1, 0, -1.8], 
                    # [0, 1, 0, -2.0], 
                    # [0, 1, 0, -3.4], 
                    # [0, 1, 0, 2.0]];
# robot_node = robot.getFromDef("firebird")
# trans_field = robot_node.getField("translation")
# rot_field = robot_node.getField("rotation")

root = robot.getRoot()
root_children_field = root.getField("children") 


node = root_children_field.getMFNode(-1) 
trans = node.getField("translation")
rot = node.getField("rotation") 

trans.setSFVec3f(startingTranslation[0])

rot.setSFRotation(startingRotation)

# LL_translation = [0.7, 0.2, -2.5]
# SS_translation = [-0.7, 0.2, 0.1]

LL_translation = [0.7, 0.2, 5.3]
SS_translation = [-0.7, 0.2, 0.5]
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    # Read sensors outputs
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
    print(trans_value)
    distance_LL = getDistance(LL_translation[0], LL_translation[2], trans_value[0], trans_value[2])
    distance_SS = getDistance(SS_translation[0], SS_translation[2], trans_value[0], trans_value[2])
    # check if the robot hit a wall
    # if front_obstacle or left_obstacle or right_obstacle:
    


    ofLeft = 0
    ofRight = 0 

    for i in range(4):
        ofRight += psValues[i]
        ofLeft += psValues[i+4]

    # velo = GAIN * ((ofLeft - ofRight)/(ofLeft + ofRight))

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
         
        # modify speeds according to obstacles
        if front_obstacle:
            left_speed -= 1.0
            right_speed -=2.0
            # if left_obstacle: 

                # left_speed += 0.1 * max_speed
                # right_speed -= 0.1 * max_speed   
            # elif right_obstacle: 
                # left_speed -= 0.1 * max_speed
                # right_speed += 0.1 * max_speed                       
            # left_speed += -1.0 * max_speed
            # right_speed += 0
            print('front obstacle')

        elif left_obstacle: 
            # turn right
            left_speed += 2.0
            right_speed -= 2.0
            # left_speed += 0.1 * max_speed
            # right_speed -= 0.1 * max_speed
            # if psValues[1] < 0.1 or psValues[0] < 0.1:
            #     left_speed += 0
            #     right_speed += -2.0
            #     print('left and front')
            # print('left obstacle')
        elif right_obstacle: 
            left_speed -= 2.0
            right_speed += 2.0
            # left_speed -= 0.1 * max_speed
            # right_speed += 0.1 * max_speed
            # print('right obstacle')
            # if psValues[3] < 0.1 or psValues[4] < 0.1:
            #     left_speed += -2.0
            #     right_speed += 0
            #     print('right and front')

    # else:
        # left_speed = 2.0
        # right_speed = 2.0
    # write actuators inputs
    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)
    print(distance_LL)
    if distance_LL < radius or distance_SS < radius: 
        found = True
    if found: 
        turning = True
        # set new starting position for the robot
        trans.setSFVec3f(startingTranslation[0])
        rot.setSFRotation(startingRotation)
        trial += 1
        found = False
    if trial > 1000:
        break    
    
