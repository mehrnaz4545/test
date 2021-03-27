import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image
import cv2
import math



DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MODEL_NAME = 'trainallSL'

MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
MIN_REWARD = -500  # For model save
MEMORY_FRACTION = 0.20
# Environment settings
EPISODES = 200
####
Thr_cost = 700
EPI_STEP = 500
PRF_Reward = 300
PRF_PENALTY = 3
MV_PENALTY = 1

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

SIZE = 40
NET_NUM = 8
T = 0
POS_array = np.array([[8, 7, 6], [4, 5, -1], [1, 2, 3]])

# DIM_FILE= "C:/Users/mehrnaza/Documents/simulations/my_ML_RF_codes/dimentions.txt"
NET_FILE = "C:/Users/mehrnaza/Documents/simulations/my_ML_RF_codes/network.txt"
SYM_FILE = "C:/Users/mehrnaza/Documents/simulations/my_ML_RF_codes/symmetry.txt"
FIN_FILE = "C:/Users/mehrnaza/Documents/simulations/my_ML_RF_codes/finnf.txt"

network = np.zeros((NET_NUM, NET_NUM), dtype=int)
with open(NET_FILE, "rb") as f:
    network = np.genfromtxt(f)

symmetry = np.zeros((NET_NUM, NET_NUM), dtype=int)  #########   1:x-symmetry    ,   2:x & y symmetry    ,   3:y symmetry
with open(SYM_FILE, "rb") as f:
    symmetry = np.genfromtxt(f)

fin_prp = np.zeros((NET_NUM, NET_NUM), dtype=int)  #########   1:x-symmetry    ,   2:x & y symmetry    ,   3:y symmetry
with open(FIN_FILE, "rb") as f:
    fin_prp = np.genfromtxt(f)


def Dim(fin_prp):
    dim = np.zeros((len(fin_prp), 2), dtype=int)
    for i in range(len(fin_prp)):
        dim[i][0] = (fin_prp[i][0] - 1) + 2 + fin_prp[i][
            2] * 2  ###########fin_prp[i][2]= dummy,     fin_prp[i][0]= gate fingers
        dim[i][1] = fin_prp[i][1] + 2
    return (dim)


def Tot_wire_l(transistors, network):
    sum = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    lada = SIZE / 70
    for j in range(len(transistors)):
        s1 += np.exp(transistors[j].x / (2 * lada))
        s2 += np.exp((-transistors[j].x) / (2 * lada))
        s3 += np.exp(transistors[j].y / lada)
        s4 += np.exp((-transistors[j].y) / lada)
        sum += (math.log(s1) + math.log(s2) + math.log(s3) + math.log(s4))
    return (sum)


def Tot_overlap(transistors):
    sumov = 0
    for i in range(len(transistors)):
        for j in range(i + 1, len(transistors)):
            sumov += transistors[i].overlap(transistors[j])  # overlap(transistors[i], transistors[j])
    return (sumov)


def Area(transistors):
    Max_x = 0
    Min_x = 0
    Min_y = 0
    Max_y = 0
    for i in range(len(transistors)):
        if transistors[i].x > Max_x:
            Max_x = transistors[i].x
        if transistors[i].y > Max_y:
            Max_y = transistors[i].y
        if transistors[i].x < Min_x:
            Min_x = transistors[i].x
        if transistors[i].y < Min_y:
            Min_y = transistors[i].y
    return ((Max_x - Min_x) / 2) * (Max_y - Min_y)


def sym_function(transistors):
    cost = 0
    xmax = 0
    xmin = SIZE
    ymax = 0
    ymin = SIZE
    x = 0
    y = 0
    for i in range(len(transistors)):
        # x +=transistors[i].x
        # y+= transistors[i].y
        # x=x/len(transistors)
        # y=y/len(transistors)
        xmin = min(transistors[i].x, xmin)
        ymin = min(transistors[i].y, ymin)
        xmax = max(transistors[i].x, xmax)
        ymax = max(transistors[i].y, ymax)

    x = (xmax + xmin) / 2
    y = (ymax + ymin) / 2

    for i in range(len(transistors)):
        for s in range(len(transistors)):
            if symmetry[i, s] == 1:
                cost += max(1 * x - ((transistors[i].x + transistors[s].x) / 2), 0)
            if symmetry[i, s] == 3:
                cost += max(2 * y - (transistors[i].y + transistors[s].y), 0)
    return (cost)


def N_Cost(transistors):
    Ncost = 120 * np.sqrt(Tot_overlap(transistors))
    if Ncost < 2:
        Ncost = 2
    return (Ncost)


def P_Cost(transistors, network):
    Pcost = 6 * np.sqrt(Area(transistors)) + 1 * Tot_wire_l(transistors, network)
    if Pcost < 2:
        Pcost = 2
    return (Pcost)


def Tot_cost(transistors, network):
    return (P_Cost(transistors, network) + N_Cost(transistors) + 50 * sym_function(transistors))


class trans:
    def __init__(self, PN):
        """self.x = SIZE / 2
        if PN == 1:  #############Pmos
            self.y = (3 * SIZE) / 4
        else:
            self.y = (1 * SIZE) / 4"""
        self.x = 0
        self.y = 0

        self.di_x = 2
        self.di_y = 1

    def __str__(self):
        return f"{self.x}, {self.y}, {self.di_x}, {self.di_y}, {self.PN}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return (self.x + other.x, self.y + other.y)

    def overlap(self, other):

        ov_y_1 = self.y + (self.di_y / 2) - (other.y - (other.di_y / 2))
        ov_y_2 = other.y + (other.di_y / 2) - (self.y - (self.di_y / 2))
        fy = 0.5 * (-np.sqrt(((ov_y_2 - ov_y_1) ** 2) + (T ** 2)) + ov_y_1 + ov_y_2)
        ov_y = 0.5 * (np.sqrt((fy ** 2) + (T ** 2)) + fy)
        ov_x_1 = self.x + (self.di_x / 2) - (other.x - (other.di_x / 2))
        ov_x_2 = other.x + (other.di_x / 2) - (self.x - (self.di_x / 2))
        fx = 0.5 * (-np.sqrt((((ov_x_2 / 2) - (ov_x_1 / 2)) ** 2) + (T ** 2)) + (ov_x_1 / 2) + (ov_x_2 / 2))
        ov_x = 0.5 * (np.sqrt((fx ** 2) + (T ** 2)) + fx)

        return (ov_x * ov_y)

    def action(self, choice):
        ######M0 trans

        if choice == 0:
            self.move(x=1, y=0)
        elif choice == 1:
            self.move(x=-1, y=0)
        elif choice == 2:
            self.move(x=0, y=1)
        elif choice == 3:
            self.move(x=0, y=-1)
       # else:
            # self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if not x:
            self.x += 2 * np.random.randint(-1, 2)  # values: -1 0 1
        else:
            self.x += 2 * x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < (self.di_x / 2) + 2:  # self.x < (self.dx/2)
            self.x += 4  # self.x < (self.dx/2)
        elif (self.x + (self.di_x / 2)) > SIZE - 2:  # SIZE - (self.dx/2)
            self.x -= 4

        if self.y < (self.di_y / 2):  # self.y < (self.dy/2)
            self.y += 2  # self.y < (self.dy/2)
        elif (self.y + (self.di_y / 2)) > SIZE - 1:  # SIZE - (self.dy/2)
            self.y -= 2

            # self.y = SIZE - 1 - (self.di_y / 2)


class TransEnv:
    ACTION_SPACE_SIZE = 5
    RETURN_IMAGES = True
    Thr_cost = 280
    PRF_Reward = 300
    PRF_PENALTY = 5
    MV_PENALTY = 1
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    nmos = 1
    pmos = 2
    psym = 3
    d = {1: (255, 175, 0),  # NMOS
         2: (0, 0, 255),  # PMOS is red
         3: (0, 255, 0)}  # symmetries is green

    def reset(self):
        self.M1 = trans(1)
        self.M2 = trans(1)
        self.M3 = trans(1)
        self.M4 = trans(0)
        self.M5 = trans(0)
        self.M6 = trans(0)
        self.M7 = trans(0)
        self.M8 = trans(0)
        self.transistors = [self.M1, self.M2, self.M3, self.M4, self.M5, self.M6, self.M7, self.M8]

        for i in range(len(POS_array)):
            y_status = SIZE / (len(POS_array) * 2) + (i * SIZE) / len(POS_array)
            for j in range(len(POS_array[i])):
                x_status = SIZE / (len(POS_array[i]) * 2) + (j * SIZE) / len(POS_array[i])
                if POS_array[i, j] >= 1:
                    index_pos= POS_array[i, j]-1
                    self.transistors[index_pos].x  = x_status
                    self.transistors[index_pos].y = y_status


        for i in range(len(self.transistors)):
            [self.transistors[i].di_x, self.transistors[i].di_y] = dimentions[i]
        env.render(self.transistors)
        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image(self.transistors))
        else:
            observation = (self.M1 - self.M2) + (self.M3 - self.M4)  ###no idea why!!!!
        return observation

    ##############to be completed
    def step(self, actions):
        self.episode_step += 1
        #index = action // 4
        #act = action % 4
        for ind in range(NET_NUM):
            act =actions[ind]
            self.transistors[ind].action(act)


        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image(self.transistors))
        else:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            new_observation = Image.fromarray(env, 'RGB')

        Best_cost = Tot_cost(self.transistors, network)
        if Best_cost <= Thr_cost:
            reward = PRF_Reward
        elif Thr_cost < Best_cost and Best_cost < 1.3 * Thr_cost:
            reward = 5
        elif 1.3 * Thr_cost <= Best_cost and Best_cost < 1.5 * Thr_cost:
            reward = 2
        # elif Best_cost < Org_cost:
        #    reward = -PRF_PENALTY
        else:
            reward = -MV_PENALTY

        done = False
        if Best_cost <= Thr_cost or self.episode_step >= EPI_STEP:
            done = True

        return new_observation, reward, done

    def render(self, transistors):
        img = self.get_image(transistors)
        img = img.resize((400, 400))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(3000)

    def get_image(self, transistors):
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        for yi in range(int(round(transistors[0].y - (transistors[0].di_y) / 2)),
                        int(round(transistors[0].y + (transistors[0].di_y) / 2))):
            for xi in range(int(round(transistors[0].x - (transistors[0].di_x) / 2)),
                            int(round(transistors[0].x + (transistors[0].di_x) / 2))):
                env[yi][xi] = self.d[self.pmos]
        for yi in range(int(round(transistors[1].y - (transistors[1].di_y) / 2)),
                        int(round(transistors[1].y + (transistors[1].di_y) / 2))):
            for xi in range(int(round(transistors[1].x - (transistors[1].di_x) / 2)),
                            int(round(transistors[1].x + (transistors[1].di_x) / 2))):
                env[yi][xi] = self.d[self.pmos]
        for yi in range(int(round(transistors[2].y - (transistors[2].di_y) / 2)),
                        int(round(transistors[2].y + (transistors[2].di_y) / 2))):
            for xi in range(int(round(transistors[2].x - (transistors[2].di_x) / 2)),
                            int(round(transistors[2].x + (transistors[2].di_x) / 2))):
                env[yi][xi] = self.d[self.pmos]
        for yi in range(int(round(transistors[3].y - (transistors[3].di_y) / 2)),
                        int(round(transistors[3].y + (transistors[3].di_y) / 2))):
            for xi in range(int(round(transistors[3].x - (transistors[3].di_x) / 2)),
                            int(round(transistors[3].x + (transistors[3].di_x) / 2))):
                env[yi][xi] = self.d[self.psym]
        for yi in range(int(round(transistors[4].y - (transistors[4].di_y) / 2)),
                        int(round(transistors[4].y + transistors[4].di_y / 2))):
            for xi in range(int(round(transistors[4].x - (transistors[4].di_x) / 2)),
                            int(round(transistors[4].x + (transistors[4].di_x) / 2))):
                env[yi][xi] = self.d[self.psym]
        for yi in range(int(round(transistors[5].y - (transistors[5].di_y) / 2)),
                        int(round(transistors[5].y + (transistors[5].di_y) / 2))):
            for xi in range(int(round(transistors[5].x - (transistors[5].di_x) / 2)),
                            int(round(transistors[5].x + (transistors[5].di_x) / 2))):
                env[yi][xi] = self.d[self.nmos]
        for yi in range(int(round(transistors[6].y - (transistors[6].di_y) / 2)),
                        int(round(transistors[6].y + (transistors[6].di_y) / 2))):
            for xi in range(int(round(transistors[6].x - (transistors[6].di_x) / 2)),
                            int(round(transistors[6].x + (transistors[6].di_x) / 2))):
                env[yi][xi] = self.d[self.nmos]
        for yi in range(int(round(transistors[7].y - (transistors[7].di_y) / 2)),
                        int(round(transistors[7].y + (transistors[7].di_y) / 2))):
            for xi in range(int(round(transistors[7].x - (transistors[7].di_x) / 2)),
                            int(round(transistors[7].x + (transistors[7].di_x) / 2))):
                env[yi][xi] = self.d[self.nmos]
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = TransEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

#checkpoint = ModelCheckpoint(filepath='trainallSL2.ckpt',save_best_only=False,save_weights_only=False,verbose=0)
checkpoint = ModelCheckpoint(filepath='trainmodel.ckpt',save_best_only=False,save_weights_only=False,verbose=0)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()
        #self.model.load_weights('trainallSL1.ckpt')
        self.model.load_weights('trainallSL2.ckpt')



        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3),
                         input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE * NET_NUM, activation='linear'))
        # ACTION_SPACE_SIZE * NET_NUM = how many choices (4) * transistor_numbers
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        # print(f"replay appen")

    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        # images frm the game
        X = []
        # actions like up, down, left, ...
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                argtochng = np.argmax(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
                argtochng = np.argmax(current_qs_list[index])
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[argtochng] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
            # print("state app")

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,callbacks=[self.tensorboard,checkpoint] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # Queries main network for Q values given current observation space (environment state)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]


agent = DQNAgent()
dimentions = np.zeros((NET_NUM, 2), dtype=int)
# Iterate over episodes
# Reset environment and get initial state
current_state = env.reset()
for i in range(len(env.transistors)):
    if fin_prp[i][0] % 2 == 1:
        env.transistors[i].x += 1
Best_trans = env.transistors
env.render(env.transistors)
print(f"total original cost {Tot_cost(Best_trans, network)}")
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    # adjust the position for x for gate alignment
    for i in range(len(env.transistors)):
        if fin_prp[i][0] % 2 == 1:
            env.transistors[i].x += 1
    # get the dimentions of the transistors
    dimentions = Dim(fin_prp)
    # Reset flag and start iterating until episode ends
    done = False

    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        j = 0
        action =np.zeros(NET_NUM , dtype=int)
        for i in range(NET_NUM):
            if np.random.random() > epsilon:
                # Get action from Q table
                action[i] = np.argmax(agent.get_qs(current_state)[j:j+5])
            else:
                # Get random action
                #action[i] = np.random.randint(0, env.ACTION_SPACE_SIZE * NET_NUM)
                action[i] = np.random.randint(0,5)
            j += 5
       # print (f"action is: {action}")
        new_state, reward, done = env.step(action)
        if Tot_cost(env.transistors, network) < Tot_cost(Best_trans, network):
            Best_trans = env.transistors
            print(f"best cost:{Tot_cost(Best_trans, network)} in episode {episode})")
            print(
                f"best dimentions: T1: {Best_trans[0].x}, {Best_trans[0].y},  T2:{Best_trans[1].x}, {Best_trans[1].y},  T3:{Best_trans[2].x}, {Best_trans[2].y}, T4:{Best_trans[3].x}, {Best_trans[3].y},  T5:{Best_trans[4].x}, {Best_trans[4].y}, T6:{Best_trans[5].x}, {Best_trans[5].y},T7:{Best_trans[6].x}, {Best_trans[6].y}, T8:{Best_trans[7].x}, {Best_trans[7].y}")
            # print(f"reward is:{reward}")
            #if reward == PRF_Reward:
            # if Tot_cost(env.transistors,network) <= Tot_cost(Best_trans,network):
            #Best_trans = env.transistors;
            #print("found best")

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render(env.transistors)

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    # print("reward app")
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        # print(f"stucked1: {episode}")
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            # print(f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model")
            agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        # print(f"stucked2")
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

print(
    f"best dimentions: T1: {Best_trans[0].x}, {Best_trans[0].y},  T2:{Best_trans[1].x}, {Best_trans[1].y},  T3:{Best_trans[2].x}, {Best_trans[2].y}, T4:{Best_trans[3].x}, {Best_trans[3].y},  T5:{Best_trans[4].x}, {Best_trans[4].y}, T6:{Best_trans[5].x}, {Best_trans[5].y},T7:{Best_trans[6].x}, {Best_trans[6].y}, T8:{Best_trans[7].x}, {Best_trans[7].y}")
print(f" best cost ever:{Tot_cost(Best_trans, network)}")
