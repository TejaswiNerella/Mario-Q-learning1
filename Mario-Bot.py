import math
import random
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import threading 
import numpy as np

import pytesseract

import cv2

from PIL import ImageGrab

from pynput.keyboard import Key, Controller,Listener
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#initial setup
pytesseract.pytesseract.tesseract_cmd=r'/opt/homebrew/bin/tesseract'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
keyboard=Controller()
#initialize movement variables
right=False
left=False
jump=False
down=False
shift=False
#initialize time, lives
timed=400
lives=9999999999
#set up parallel events
death_event=threading.Event()
stop_event=threading.Event()
stop_event1=threading.Event()
#set up memory tuples
Transition=namedtuple("Transition",('state','action','next_state','reward'))
#create ReplayMemory
class ReplayMemory(object):
    def __init__(self,capacity): #self initialization
        self.memory=deque([],maxlen=capacity)
    def push(self,*args): #pushing a tuple onto memory
        self.memory.append(Transition(*args))
    def sample(self,batch_size): #taking a random sample of memory
        return random.sample(self.memory,batch_size)
    def __len__(self): #length of memory
        return len(self.memory)
#neural net model
class QLearning(nn.Module):
    def __init__(self, n_observations,n_actions): #initialization
        super(QLearning,self).__init__()
        self.l1=nn.Linear(n_observations,256) #linear layer from observations to 256
        self.l2=nn.Linear(256,256) #linear layer from 256 to 256
        self.l3=nn.Linear(256,128) #linear layer from 256 to 128
        self.l4=nn.Linear(128,n_actions) #linear layer from 128 to number of actions

    def forward(self,x): #forward pass of the neural net
        x=F.relu(self.l1(x)) #ReLu activation function between layer 1
        x=F.relu(self.l2(x)) #ReLu activation function between layer 2
        x=F.relu(self.l3(x)) #ReLu activation function between layer 3
        return self.l4(x) #return output of the final layer


def randAct(length): #random action function
    ActList=[0]*length #intialize list based on how many actions
    # the following loop chooses a random amount of times to choose a random action
    for i in range(random.randrange(1,length)): 
        ActList[random.randrange(0,length)]=1 
    return ActList #return the random action list


BATCH_SIZE=16 #Training loop's batch size
GAMMA=0.90 #Discount Rate-lower is more present focused, higher is future focused
EPS_START=0.9 #Starting random action rate
EPS_END=0.05 #ending random action rate
EPS_DECAY=1000 #random action decay rate
TAU=0.005 #update rate of the target network
LR=0.01 #learning rate of the optimizer


cap=ImageGrab.grab(bbox=(0,160,1439,530)) #initialize the screen grab
n_observations=4 #the screenshots have a final layer of size 4
n_actions=5 #there are 5 actions: right, left, jump, down, run/fire

policy_net=QLearning(n_observations,n_actions).to(device) #create the policy net
target_net=QLearning(n_observations,n_actions).to(device) #create the target net
target_net.load_state_dict(policy_net.state_dict()) #target net gets the weights of policy net

optimizer=optim.AdamW(policy_net.parameters()) #optimizer for policy net
memory=ReplayMemory(100) #memory stores 100 transitions

steps_done=0 #initialize steps done to 0

def select_action(state):
    global steps_done #steps done is global
    sample=random.random() #sample gets a random number
    #calculating the random action threshold:
    eps_threshold=EPS_END + (EPS_START-EPS_END) * math.exp(-1*steps_done/EPS_DECAY)
    steps_done+=1 #increment steps done
    #if the sample is above the threshold then use the model's action
    if sample >eps_threshold:
        with torch.no_grad():
            return (policy_net(state).max(1)[1]).div(100).type(torch.long)
    #otherwise use a random action
    else:
        return torch.tensor([[randAct(n_actions)]],device=device,dtype=torch.long)

episode_durations = [] #list to store how long each episode lasts

def optimize_model():
    if len(memory)<BATCH_SIZE: #if there isn't enough memory to form a batch return
        return
    transitions=memory.sample(BATCH_SIZE) #get a sample of memory
    batch=Transition(*zip(*transitions)) #create a batch out of the memory
    #creating a boolean Tensor based on batch
    #for each next state value of batch, if it is not None, it is True, otherwise it's False
    non_final_mask=torch.tensor(tuple(map(lambda s:s is not None,batch.next_state)),device=device,dtype=torch.bool)
    #stores all of the next states in the batch if next state is not None
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None])
    #put all states in the batch in one tensor
    state_batch=torch.cat(batch.state)
    #put all actions in the batch in one tensor
    action_batch=torch.cat((batch.action))
    #put all rewards in the batch in one tensor
    reward_batch=torch.cat(batch.reward)
    
    state_batch_t=policy_net(state_batch).max(1)[1] #get the neural net's output of the batch
    state_batch_t=torch.max(state_batch_t,1)[1] #reduce the output to the correct shape
    state_action_values=state_batch_t.gather(dim=1,index=action_batch) #Q-value generation
    next_state_values=torch.zeros(BATCH_SIZE,device=device) #crate a tensor for next state values in the batch
    with torch.no_grad(): #do not affect the neural net's values
        #next state tensor gets target net's output of the next states
        next_stateo=target_net(non_final_next_states).max(1)[0]
        next_stateo=next_stateo.max(1)[0] #reduce tensor to correct shape
        #next_state_values gets the tensor created above
        next_state_values[non_final_mask]=next_stateo.max(1)[0]
    #calculates the expected Q function based on the next states
    expected_state_action_values=(next_state_values*GAMMA)+reward_batch
    #Using Huber Loss calculating the loss of the Q function
    criterion=nn.SmoothL1Loss()
    print("input shape: ",state_action_values.size())
    print("target shape: ", expected_state_action_values.size())
    loss=criterion(state_action_values,expected_state_action_values.unsqueeze(1).repeat(1,5))
    optimizer.zero_grad() #reset the optimizer's gradient for the current step
    loss.requires_grad=True
    loss.backward() #do a backwards pass from the loss function
    #clip the gradient value between -100 and 100
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step() #optimizer changes weights of policy net to reduce loss

scroll_cap=np.array(ImageGrab.grab(bbox=(1090,590,1190,750)))
def move(action): #function to make mario move based on the action
    #set global variables
    global right
    global left
    global jump
    global down
    global shift
    action=action.numpy()[0][0] #get the actual action array
    print(action)
    #[right,left,jump,down,shift]
    if action[0]: #if going right
        if not right: #if not already going right 
            keyboard.press(Key.right) #press right
            right=True #right is being pressed
        keyboard.release(Key.left) #don't press left
        left=False #let go of left
    elif action[1]: #if not going right but going left
        if not left: #if not already going left
            keyboard.press(Key.left) #press left
            left=True #left is being pressed
        keyboard.release(Key.right) #release right
        right=False #right is not being pressed
    if action[2]: #if jumping
        if not jump: #if not already jumping
            keyboard.press(Key.up) #press up
            jump=True #up is being pressed
        keyboard.release(Key.down) #release down
        down=False #down is not being pressed
    elif action[3]: #if going down
        if not down: #if not already going down
            keyboard.press(Key.down) #press down
            down=True #down is being pressed
        keyboard.release(Key.up) #release up
        up=False #up is not being pressed
    if action[4]: #if running
        if not shift: #if not already running
            keyboard.press(Key.shift) #press shift
            shift=True #shift is being pressed
    else: #if not running
        keyboard.release(Key.shift) #release shift
        shift=False #shift is not being pressed
    if not action[0]: #if not going right
        right=False #right isn't being pressed
        keyboard.release(Key.right) #release right
    if not action[1]: #if not going left
        left=False #left isn't being pressed
        keyboard.release(Key.left) #release left
    if not action[2]: #if not jumping
        jump=False #up isn't being pressed
        keyboard.release(Key.up) #release up
    if not action[3]: #if not going down
        down=False #down isn't being pressed
        keyboard.release(Key.down) #release down

#function to check if mario has gained a life
def life_gain(): 
    global lives #stores lives
    cap=ImageGrab.grab(bbox=(1120,145,1280,170))#image of lives
    livestr=pytesseract.image_to_string( #image to string
        cv2.cvtColor(np.array(cap),cv2.COLOR_BGR2GRAY),
            config='digits')
    if(int(livestr)>lives): #if lives have been gained
        lives=int(livestr) #set lives to the correct amount
        
#function to check whether mario has died
def death_check():
    global timed #stores time
    global lives #stores lives
    while True:
        cap= ImageGrab.grab(bbox=(830,145,895,180)) #takes image of time
        teststr=pytesseract.image_to_string( #converts image into string
            cv2.cvtColor(np.array(cap),cv2.COLOR_BGR2GRAY),
                config='digits')
        if teststr=='': #if nothing is detected set string to 0
            teststr='0'
        if teststr[-1]=='\n':
                teststr=str('0')
        if int(teststr)==timed: #if the time hasn't changed
            stop_event.set()
            time.sleep(2) #wait 2 seconds
            cap= ImageGrab.grab(bbox=(830,145,885,170)) #takes image of time
            teststr=pytesseract.image_to_string( #converts image to string
                cv2.cvtColor(np.array(cap),cv2.COLOR_BGR2GRAY),
                    config='digits')
            if teststr=='':
                teststr='0'
            if teststr[-1]=='\n':
                teststr=str('0')
            if int(teststr)==timed: #if time hasn't changed (wasn't a pipe)
                stop_event1.set() #stop
                while int(teststr)<=timed: #while time hasn't reset
                    cap= ImageGrab.grab(bbox=(830,145,885,170))#image of time
                    teststr=pytesseract.image_to_string( #time to string
                        cv2.cvtColor(np.array(cap),cv2.COLOR_BGR2GRAY),
                            config='digits')
                    if teststr=='': #set blank to 0
                        teststr="0"
                    if teststr=='-\n': #common error is set to 0
                        teststr="0"
                        print("it happened in loop")
                cap=ImageGrab.grab(bbox=(1120,145,1280,170))#image of lives
                livestr=pytesseract.image_to_string( #image to string
                    cv2.cvtColor(np.array(cap),cv2.COLOR_BGR2GRAY),
                        config='digits')
                if livestr=="":
                    livestr=str(lives)
                if livestr=="-\n":
                    livestr=str(lives)
                if int(livestr)<lives: #if death
                    lives=int(livestr)
                    death_event.set() #trigger the event
        if teststr=='':
            teststr='0'
        timed=int(teststr)
        time.sleep(0.2)

def step(action):#make one step in the environment
    #set up global variables
    global scroll_cap
    global right
    global left
    global jump
    global down
    global shift
    move(action)#move 
    reward=0#initialize reward to 0
    observation=np.array(ImageGrab.grab(bbox=(200,0,1350,560))) #take observation
    #The model receives a reward for moving forward so it is determined by
    #checking if the bottom of the screen has changed, indicating the model
    #has moved to the right and made the screen scroll
    cur_scroll=np.array(ImageGrab.grab(bbox=(1090,590,1190,750))) #check current scroll screenshot
    #if the model has moved the screen increase reward and set the
    #baseline to the current scroll level
    if not np.array_equal(scroll_cap,cur_scroll):
        scroll_cap=cur_scroll
        reward=reward+10
        
    terminated=False #set death to false

    #If the timer has stopped release all keys and wait for a signal
    if stop_event.is_set(): 
        keyboard.release(Key.right)
        keyboard.release(Key.left)
        keyboard.release(Key.down)
        keyboard.release(Key.up)
        keyboard.release(Key.shift)
        right=False
        left=False
        jump=False
        down=False
        shift=False
        #Pipe transitions stop the time for 2 seconds
        if stop_event1.wait(2): #if it isn't a pipe transition
            if death_event.wait(5): #if a terminal state has been reached
                terminated=True #set terminated to True
                reward=reward-50 #decrease reward by 50
                death_event.clear() #reset the death checker
            #if a terminal state hasn't been reached the level has been completed
            else: 
                reward=reward+275 #increase reward by 275 for level completion
            stop_event1.clear() #reset the stop checker
        stop_event.clear() #reset the stop checker
        
    #return the screenshot, reward, and whether mario died on this step
    return observation,reward,terminated

def optimThread():
    while True:
        #perform a step of optimization
        optimize_model()
        #update the weights
        target_net_state_dict=target_net.state_dict()
        policy_net_state_dict=policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key]=policy_net_state_dict[key]*TAU+target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
    
    
#if gpu is available train for 1200 episodes, else train for 250 episodes
if torch.cuda.is_available:
    num_episodes=1200
else:
    num_episodes=250



#start training
death_thread=threading.Thread(target=death_check) #create a thread to check for death
optimization_thread=threading.Thread(target=optimThread) #create a thread to optimize model 
death_thread.daemon=True #if the main thread has an exception the program ends
optimization_thread.daemon=True #if the main thread has an exception the program ends
time.sleep(3) #wait 3 seconds before beginning training
death_thread.start() #start checking for deaths
optimization_thread.start() #begin Optimization process
scroll_cap=np.array(ImageGrab.grab(bbox=(1090,590,1190,750))) #Initialize the scroll check

#begin training loop
for i_episode in range(num_episodes):
    #take the initial screenshot and converts it to tensor
    state=np.array(ImageGrab.grab(bbox=(200,0,1350,560)))
    
    state=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
    for t in count(): #until death
        action=select_action(state) #select an action
        observation,reward,terminated=step(action) #take a step in the environment
        reward=torch.tensor([reward],device=device)# convert reward to tensor
        if terminated:
            #there is no next state if death
            next_state=None
        else:
            #store next state if mario survives
            next_state=torch.tensor(observation,dtype=torch.float32,device=device).unsqueeze(0)
        memory.push(state,action[0][0].view(1,-1).clamp(min=0,max=4),next_state,reward)#store the step
        #move to the next state
        state=next_state #state stores the next state
        if terminated: #if mario dies
            episode_durations.append(t+1) #stores the amount of steps in the episode
            break #end the loop
FILE="policy.pth"
torch.save(policy_net.state_dict(),FILE)
FILE2="target.pth"
torch.save(target_net.state_dict(),FILE2)
print('Complete')
        

        
