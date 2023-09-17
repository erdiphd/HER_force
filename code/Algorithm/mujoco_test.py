import gym
import threading
import numpy as np
env = gym.make("FetchPickAndPlace-v1", reward_type="sparse")

def continuous_run():
    while True:
        env.render()


sim_thread = threading.Thread(target=continuous_run)
sim_thread.start()


obs = env.reset()



print(gym.__file__)

counter = 0
obs = env.reset()
action_test = [0,0,0,1]
while True:
    action_test = env.action_space.sample()
    tmp = env.step(np.zeros(8))
