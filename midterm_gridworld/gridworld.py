"""gridworld controller."""
# Jungsun Yoo
# University of California, Irvine
# Neurorobotics Course (PSYCH 268R)

# Midterm challenge

# Description: I implemented a grid-world RL paradigm using the square arena. 
# Specifically, there are 16 "states" which refer to each checkerboard. 
# The objective of this task is to start at a location ([1,1] or 5th grid) and
# find the way to the last grid ([3,3]). Firebird robot is used here and this
# robot learns through an epsilon-greedy policy (epsilon greedy Q-learning).

# Also, the left and right collision avoiding module is controlled by a small neural 
# network in which the speed is the output neuron (motor command) and the 
# distance to the obstacle is the input neuron. 

# Each episode is 2000 timepoints long, and each episode ends if the robot finds the
# platform or the timepoint has exceeded 2,000 timepoints, whichever shorter. 
# After each episode, the supervisor places the robot in the starting point ([1,1]).
# The total experiment is consisted of 1,000 episodes.
# (Due to time constraint, the reported results are based on approximately
# 650 trials (episodes))

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

# degrees for left, right, up, down
head_directions = [90, 270, 0, 180]


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






def getDistance(x1, y1, x2, y2): 
    x = (x1-x2)**2
    y = (y1-y2)**2
    return (x+y)**0.5
state_coords = [[-1.8, 0, -1.8], #0
                [-0.8, 0, -1.8], #1
                [0.2, 0, -1.8], #2
                [1.2, 0, -1.8], #3
                [-1.8, 0, -0.8], #4
                [-0.8, 0, -0.8], #5
                [0.2, 0, -0.8], #6
                [1.2, 0, -0.8], #7
                [-1.8, 0, 0.2], #8
                [-0.8, 0, 0.2], #9
                [0.2, 0, 0.2], #10
                [1.2, 0, 0.2], #11
                [-1.8, 0, 1.2], #12
                [-0.8, 0, 1.2], #13
                [0.2, 0, 1.2], #14
                [1.2, 0, 1.2]] #15

startingTranslation = state_coords[5]; # Each trial, the robot starts in the red (start) tile
goalTranslation = state_coords[-1]; # Goal (platform) is the green tile
startingRotation = [[0, 1, 0, -1.8], 
                    [0, 1, 0, -2.0], 
                    [0, 1, 0, -3.4], 
                    [0, 1, 0, 2.0]];

def getState(curr_coord, state_coords):
    # state_coords is an array of tuples
    arr = []
    for i in range(len(state_coords)):
        arr.append(getDistance(curr_coord[0], curr_coord[2], state_coords[i][0], state_coords[i][2]))
    currentState = np.argmin(arr)
    return currentState

def limit_coordinates(coord):

    coord[0] = min(coord[0], 3)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], 3)
    coord[1] = max(coord[1], 0)
    return coord


def update_q_table(table,s1,s2,action,possible_actions,reward,alpha,gamma):

	
	s1_ = s1[0] * 4 + s1[1]
	s2_ = s2[0] * 4 + s2[1]
	
	
	max_q =  -sys.maxint # instantiate max value to smallest number possible
	
	# Picking best next action
	for a in possible_actions:
		q  = table[s2_,a]
		if q > max_q:
			max_q = q
	# update table
	table[s1_,action] = table[s1_,action] + alpha * (reward + gamma*max_q - table[s1_,action])
	return table

# do this once only



robot_node = robot.getFromDef("firebird")
trans_field = robot_node.getField("translation")
rot_field = robot_node.getField("rotation")

root = robot.getRoot()
root_children_field = root.getField("children") 


node = root_children_field.getMFNode(-3) 
field = node.getField("translation")
rot = node.getField("rotation") 

field.setSFVec3f(startingTranslation)

rot.setSFRotation(startingRotation[random.randint(0,len(startingRotation)-1)])


def list_difference(list1, list2):
    difference = []
    for list1_i, list2_i in zip(list1, list2):
        difference.append(list1_i - list2_i)
    return difference

def checking_offlimit(coord):
    check = False
    if coord[0] < 0 or coord[1]<0 or coord[0] > 3 or coord[1] > 1:
        check = True
    return check



epsilon = 0.1
alpha = 1
gamma = .5
possible_actions = [0,1,2,3] # 0 = left, 1 = right, 2 = up, 3 = down

def make_epsilon_greedy_policy(Q, epsilon, nA): 
  """
  Creates an epsilon-greedy policy based on a given Q-function and epsilon

  Args:
    Q: A dictionary that maps from state -> action-values. 
        Each value is a numpy array of length nA (see below)
    epsilon: The probability to select a random action. Float between 0 and 1
    nA: Number of actions in the environment.

  Returns:
    A function that takes the observation as an argument and returns
    the probabilities for each action in the form of a numpy array of length nA. 
  """
  def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA 
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
  return policy_fn    


# Set the initial position of the robot using the supervisor

# field(trans_field, startingTranslation)
# field(rot_field, startingRotation)
# robot.simulationReset()
print("Reinforcement learning using Nex Fire Bird 6 robot\n")


# platform_radius = 0.33
recording_Q_table = []
recording_episode_len=[]
isRewarded = []
max_speed = 6.28
curr_state=5
s1 = [1,1]
Q = np.zeros((16,4)) # 16 states, 4 actions
policy = make_epsilon_greedy_policy(Q, epsilon, len(possible_actions))
# Main loop:
# - perform simulation steps until Webots is stopping the controller



# Neural network for detecting obstacles
# initial weights
left_input_w1 = 1.0
left_input_w2 = 1.0
left_output_w1 = 1.0
left_output_w2 = 1.0

right_input_w1 = 1.0
right_input_w2 = 1.0
right_output_w1 = 1.0
right_output_w2 = 1.0


learning_rate = 0.25

gain = 1
episode_length=0
while robot.step(timestep) != -1:
    # initialize position
    # choose action based on 
    
    
    
    
    action_probs = policy(curr_state)
    
    
    
    

    a = True
    while a == True: 
        
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


        if action == 0: # if left
            s2 = list_difference(s1, [0,1])
        elif action == 1: # if right

            s2 = list_difference(s1, [0,-1])
        elif action == 2: # if up
 
            s2 = list_difference(s1, [1,0])
        else:             # down
            s2 = list_difference(s1, [-1, 0])
        a = checking_offlimit(s2) # checking if the move is possible and end when possible
        
    

    
    # Read sensors outputs
    psValues=[]
    for i in range(8):
        psValues.append(ps[i].getValue())   
    

    # Check if the robot hit the arena wall
    threshold = 0.25
    left_obstacle = psValues[1] < threshold or psValues[0] < threshold
    right_obstacle = psValues[3] < threshold or psValues[4] < threshold
    front_obstacle = psValues[2] < threshold
    
    
    # Neural network
    
    left_input_w1 = psValues[1] + psValues[0]
    right_input_w1 = psValues[3] + psValues[4]
    
    wLeft = learning_rate * (left_input_w1 * left_output_w1)
    left_output_w1 = 1 / (1 + np.exp(-1 * gain * wLeft * left_input_w1))
    
    wRight = learning_rate * (right_input_w1 * right_output_w1)
    right_output_w1 = 1 / (1 + np.exp(-1 * gain * wRight * right_input_w1))

    
    # init speeds
    left_speed = max_speed * 0.5
    right_speed = max_speed * 0.5
    t+=1
    trans_value = trans_field.getSFVec3f()
    # distance = getDistance(goalTranslation[0], trans_value[0], goalTranslation[2], trans_value[2])
    
    curr_state = getState(trans_value, state_coords) # current "State" - one of 16 states

    next_state = s2 # next state according to selected action

    
    # check if the robot hit a wall
    if front_obstacle or left_obstacle or right_obstacle:
         
        # modify speeds according to obstacles
        if front_obstacle: 
            # turn back, but slightly right to not block the robot
            left_speed = 0.0
            right_speed = -1.0
            # left_speed = left_output_w1
            # left_speed += front_output_w1 * max_speed
            # right_speed -= front_output_w1 * max_speed
            
        elif left_obstacle: 
            # turn right
            left_speed += left_output_w1 * max_speed
            right_speed -= left_output_w1 * max_speed
            # left_speed += 0.5 * max_speed
            # right_speed -= 0.5 * max_speed
            # left_speed = 3.0
            # right_speed = -1.0
        elif right_obstacle: 
            left_speed -= right_output_w1 * max_speed
            right_speed += right_output_w1 * max_speed
            # left_speed -= 0.5 * max_speed
            # right_speed += 0.5 * max_speed 
            # left_speed = -1.0
            # right_speed = 3.0
        
    
    # check if the robot is in the middle of a turn. Use the compass to rotate 
    # to the desired heading. rotate in the direction that is shortest to the desired heading
    
    elif turning: 
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
        heading_error = head_directions[action] - bearing
        if abs(heading_error) < HD_RES: 
            turning = False
            # t += 1
        elif head_directions[action] == 0 and bearing > 355.0: 
            turning = False
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
                

    
    # move in the desired heading. update counter for forward movement
    else:
        left_speed = 2.0
        right_speed = 2.0

        
    
    # write actuators inputs
    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)

    # update when state changes
    curr_1 = curr_state // 4 # code to transform scalar states into coordinates
    curr_2 = curr_state % 4

    if [curr_1, curr_2] == s2: # robot moved from current state to the next state
        # update q table 
        reward=-1.0
        Q = update_q_table(Q,s1,s2,action,possible_actions,reward,alpha,gamma)
        s1 = s2
        episode_length+=1

    
    if [curr_1, curr_2] == [3,3]: #found the goal (platform)
        
        found = True    

    
    

    if found or (t % TURN_RATE == 0 ): # finishing up this episode upon finding or out of time
        turning = True

           
        
        if found: 
            print("Found platform on trial {}".format(trial))            
            reward = -1.0           
            Q = update_q_table(Q,s1,s2,action,possible_actions,reward,alpha,gamma)
            isrewarded=1 # found the platform            

        else: 
            reward = -1.0 
            Q = update_q_table(Q,s1,s2,action,possible_actions,reward,alpha,gamma)          
            isrewarded=0 # didn't find the platform


        
        # Recording results into a txt file
        
        Q_outfile = open('Q.txt', 'a')
        Q_outfile.write("%s\n" %Q)
        
        Reward_outfile = open('isRewarded.txt', 'a')
        Reward_outfile.write("%s\n" %isrewarded)
        
        Episode_outfile = open('episodeLen.txt', 'a')
        Episode_outfile.write("%s\n" %episode_length)
        
        
        # set new starting position for the robot
        field.setSFVec3f(startingTranslation)
        rot.setSFRotation(startingRotation[random.randint(0,len(startingRotation)-1)])

        # initializing parameters for next episode
        trial += 1
        found = False
        t = 1
        s1 = [1,1]        
        episode_length=0
    
    if trial>1000: # finishing while loop after 1000 trials
        break
    
Q_outfile.close
Rewarded_outfile.close
Episode_outfile.close
# Enter here exit cleanup code.
