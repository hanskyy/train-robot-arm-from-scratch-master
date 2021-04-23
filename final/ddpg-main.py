from ddpgrl import DDPG
import numpy as np
import gym
import subprocess
import matplotlib.pyplot as plt
from PIL import Image

ON_TRAIN = False


# set env as Fetch and set dim
env = gym.make('FetchReach-v1')
desiredgoal_shape = env.observation_space['desired_goal'].shape[0]    #desired_goal is 3
achievedgoal_shape = env.observation_space['achieved_goal'].shape[0]   # 3
observation_shape = env.observation_space['observation'].shape[0]      # 10
a_dim = env.action_space.shape[0]                               # 4
s_dim = observation_shape + achievedgoal_shape + desiredgoal_shape  # 16
#s_dim = 13
#a_dim = 4
a_bound = env.action_space.high[1]

max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

# state = env.reset().values()
# obs, curr_pos, goal_pos = state
# exp = np.concatenate((obs, curr_pos, goal_pos))
# exp.reshape((s_dim,))
# obs_log = []
# curr_pos_log = []
# next_obs_log = []
# next_curr_pos_log = []
rl = DDPG(input_dims=s_dim, env=env, n_actions=a_dim)

MAX_EPISODES = 20000
MAX_EP_STEPS = 200

def train():
    # start training
    for i in range(MAX_EPISODES):

        state = env.reset().values()
        obs, curr_pos, goal_pos = state
        init_pos = curr_pos
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            # env.render()
            action = rl.choose_action(np.concatenate((obs, curr_pos, goal_pos)))

            next_state, reward, done, info = env.step(action)
            next_obs, next_curr_pos, goal_pos = next_state.values()

            #send new experience to replay buffer
            exp = np.concatenate((obs, curr_pos, goal_pos))
            next_exp = np.concatenate((next_obs, next_curr_pos, goal_pos))
            rl.store_transition(exp, action, reward, next_exp, done)

            # start to learn once has fulfilled the memory
            if rl.memory_full:
                rl.learn()

            # update the states
            curr_pos = next_curr_pos
            obs = next_obs
            ep_reward += reward

            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_reward, j))
                break
    rl.save()


def eval():
    rl.restore()
    state = env.reset().values()
    obs, curr_pos, goal_pos = state
    subprocess.call(['rm', '-rf', 'frames'])
    subprocess.call(['mkdir', '-p', 'frames'])
    time_step_counter = 0
    while True:
        #env.render()
        action = rl.choose_action(np.concatenate((obs, curr_pos, goal_pos)))
        next_state, reward, done, info = env.step(action)
        next_obs, next_curr_pos, goal_pos = next_state.values()

        #img = env.render(height=480, width=480)
        image_data = env.render(mode="rgb_array")
        # plt.imshow(img)
        # plt.show()
        # image_data = env.render(height=480, width=480,camera_id=-1)
        img = Image.fromarray(image_data, 'RGB')
        img.save("frames/frame-%.10d.png" % time_step_counter)
        time_step_counter += 1
        curr_pos = next_curr_pos
        obs = next_obs
        if done:
            break
    subprocess.call([
        'ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])

if ON_TRAIN:
    train()
else:
    eval()