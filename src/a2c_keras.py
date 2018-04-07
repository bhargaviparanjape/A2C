from __future__ import print_function

import os

import sys
import argparse
import numpy as np
np.random.seed(0)
import pickle as pkl

import keras
from keras.optimizers import Adam

import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl

import math

class A2C():
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n

        with open(self.args.model_config_path, 'r') as f:
            shared_model = keras.models.model_from_json(f.read()) 
        shared_model.pop()

        kernel = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
                                                   distribution='normal', seed=None)
        actor_model = keras.layers.Dense(self.nA, activation='softmax', name='act_output', 
                                 kernel_initializer= kernel)
        critic_model = keras.layers.Dense(1, activation='linear', name='crit_output', 
                                  kernel_initializer= kernel)

        state_input = keras.layers.Input(shape=(self.nS,), name='state_input')
        shared_output = shared_model(state_input)
        actor_output = actor_model(shared_output)
        critic_output = critic_model(shared_output)

        self.model = keras.models.Model(inputs=state_input, outputs=[actor_output, critic_output])

        if self.args.mode=='train' and self.args.resume:
            print('*'*80)
            print ('Resuming Training from last Checkpoint')
            print('*'*80)
            best_model_path = sorted([int(ep) for ep in os.listdir(self.args.model_path)])[-1]
            self.model.load_weights(os.path.join(self.args.model_path, str(best_model_path)))
            self.episodes_done = best_model_path
        else:
            self.episodes_done = 0

        self.n = args.n
        self.lr = args.lr
        self.gamma = args.gamma

    def truncated_discounted_rewards(self, rewards):
        
        batch_size = len(rewards)-self.n
        truncated_rewards = np.zeros(batch_size)
        for t in range(batch_size):
            cumulative = 0
            for i in range(0, self.n):
                cumulative += math.pow(self.gamma, i)*rewards[t + i]
            truncated_rewards[t] = cumulative
        return truncated_rewards

    def get_value_reward(self, states, rewards, values):

        extended_values = values + [0]*self.n
        extended_rewards = rewards + [0]*self.n
        truncated_discounted_rewards = self.truncated_discounted_rewards(extended_rewards)
        batch_size = len(rewards)
        discounted_rewards = np.zeros_like(rewards)
        for t in reversed(range(batch_size)):
            discounted_rewards[t] = math.pow(self.gamma,self.n)*extended_values[t+self.n] + \
                                    truncated_discounted_rewards[t]
        return discounted_rewards

    def train(self):
        
        if self.args.optimizer == "adam":
            self.optimizer = Adam(lr = self.args.lr)
        
        self.model.compile(optimizer='rmsprop',
              loss={'act_output': 'categorical_crossentropy', 'crit_output': 'mean_squared_error'},
              loss_weights={'act_output': 1., 'crit_output': 0.2})
                
        total_act_loss = []
        total_crit_loss = []
        all_rewards = []
        
        for episode in range(self.episodes_done ,self.args.num_episodes):
            
            ## Generate Episode
            states, actions, rewards, values = self.generate_episode(self.env, self.args.render)
            discounted_rewards = self.get_value_reward(states, rewards, values)

            states = np.array(states)
            act_target = np.zeros((len(states),self.nA))
            act_target[np.arange(len(states)), np.array(actions)] = (np.array(discounted_rewards) 
                                                                    - np.array(values))

            crit_target = np.array(discounted_rewards)

            _, act_loss, crit_loss = self.model.train_on_batch(states, {'act_output': act_target, 
                                                                    'crit_output': crit_target})

            total_act_loss.append(act_loss)
            total_crit_loss.append(crit_loss)
            all_rewards.append(np.sum(rewards)*1e2)


            if (episode + 1) % self.args.log_every == 0:
                if self.args.verbose:
                    print("Num Episodes: {0}, Train Reward: {1} +/- {2}, Act Loss: {3}, Crit Loss: {4}".
                          format(episode+1, np.mean(all_rewards), np.std(all_rewards), 
                          np.mean(total_act_loss), np.mean(total_crit_loss)))
                total_act_loss = []
                total_crit_loss = []
                all_rewards = []
                          
            if (not self.args.trial) and(episode + 1) % self.args.eval_after == 0:
                if self.args.verbose:
                    print("Saving Model Weights")
                self.model.save_weights(os.path.join(self.args.model_path, str(episode+1)))

            sys.stdout.flush()

        
    def test_episode(self, num_test_episodes = 100):
        
        all_rewards = []                 
        for i in range(num_test_episodes):
            _ ,_ , rewards, _ = self.generate_episode(self.env, self.args.render)
            episode_reward = np.sum(rewards)
            all_rewards.append(episode_reward*1e2)
        
        average_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        return average_reward, std_reward
        
    def test(self):
        trained_episodes = sorted([int(ep) for ep in os.listdir(self.args.model_path)])
        performance = []
        for episode in trained_episodes:
            self.model.load_weights(os.path.join(self.args.model_path, str(episode)))
            average_reward, std_reward = self.test_episode()
            performance.append([episode, average_reward, std_reward])
            print('*'*80)
            print("Average reward after {0} episodes: {1} +/- {2}".format(episode, 
                                                          average_reward, std_reward))
            print('*'*80)
        self.plot_performance(performance)
                          

    def generate_episode(self, env, model, render = False):
        
        states = []
        actions = []
        rewards = []
        values = []
        
        current_state = env.reset()
        is_terminal  = False
        while not is_terminal:
            action_distribution, cricval = self.model.predict(np.expand_dims(current_state, 0))
            action = np.random.choice(env.action_space.n, 1, 
                                      p=action_distribution.squeeze(0))[0]
            next_state, reward, is_terminal, _ = env.step(action)
            states.append(current_state)
            actions.append(action)
            rewards.append(reward*1e-2)
            values.append(cricval.squeeze(0)[0])
            current_state = next_state
            
        return states, actions, rewards, values
    
    def plot_performance(self, performance):
        
        X = [x[0] for x in performance]
        Y = [x[1] for x in performance]
        Z = [x[2] for x in performance]

        plt.figure()
        plt.errorbar(X, Y, yerr = Z, ecolor='r', capsize = 2)
        plt.axhline(y=200, linestyle='--')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title("Performance of A2C Algorithm (n="+str(self.args.n)+") on Lunar Lander")
        plt.savefig(self.args.plot_path,dpi = 200)


def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config-path', dest='model_config_path', type=str,
                        default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--result_path', dest='result_path',type=str, 
                        default='a2c_keras',
                        help="Path to the model.")
    parser.add_argument('--resume', dest='resume', type=int, default=0, 
                        help="Resume the training from last checkpoint")

    parser.add_argument('--num-episodes', dest='num_episodes', type=int, 
                        default=50000, 
                        help="Number of episodes to train on.")
    parser.add_argument('--eval_after', dest='eval_after', type=int, 
                        default=500, 
                        help="Number of episodes to evaluate after.")
    parser.add_argument('--log_every', dest='log_every', type=int, 
                        default=25, 
                        help="Number of episodes to log after.")
    parser.add_argument('--gamma', type=float, dest='gamma', 
                        default = 1)
    
    parser.add_argument('--run', type=int, dest='run', default=1)
    parser.add_argument('--seed', type=int, dest='seed', default=0)
    parser.add_argument('--trial', dest='trial', action='store_true',
                         help="If it is just a trial")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                         help="Whether to print loss after every episode.")
    parser.add_argument('--mode', type=str, dest='mode', 
                        default='train',
                        help="Optimizer to be Used")
    
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render', 
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render', 
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, 
                        help="The learning rate.")
    parser.add_argument('--optimizer', type=str, dest='optimizer', 
                        default='adam', help="Optimizer to be Used")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    args.model_path = os.path.join(args.result_path, 
                              'a2c_model' +str(args.run) + '_' + str(args.n))
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    args.plot_path = os.path.join(args.result_path, 'a2c_plot' +str(args.run) + 
                                  '_' + str(args.n) + '.png')
    
    return args


def main(args):
    
    # Parse command-line arguments.
    args = parse_arguments()
    
    # Create the environment.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    
    a2c = A2C(env, args)
    
    if args.mode == 'train':
        a2c.train()
        
    elif args.mode == 'test':
        a2c.test()    

if __name__ == '__main__':
    main(sys.argv)