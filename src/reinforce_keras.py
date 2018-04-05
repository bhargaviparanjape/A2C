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


'''
 Implementation of the policy gradient method REINFORCE.
'''
class Reinforce(object):

    def __init__(self, env, args):

        self.env = env
        self.args = args
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        
        with open(self.args.model_config_path, 'r') as f:
            self.model = keras.models.model_from_json(f.read())
        
        # Model to be resumed. Update stored states
        if self.args.mode=='train' and self.args.resume:
            print('*'*80)
            print ('Resuming Training from last Checkpoint')
            print('*'*80)
            best_model_path = sorted([int(ep) for ep in os.listdir(self.args.model_path)])[-1]
            self.model.load_weights(os.path.join(self.args.model_path, str(best_model_path)))
            self.episodes_done = best_model_path
        else:
            self.episodes_done = 0

        self.lr = args.lr
        self.gamma = args.gamma
        
    def discounted_rewards(self, rewards):
        
        T = len(rewards)
        discounted_rewards = np.zeros_like(rewards)
        discounted_rewards[T-1] = rewards[T-1]
        for t in reversed(range(0, T-1)):
            discounted_rewards[t] = (discounted_rewards[t+1] * self.gamma + rewards[t])
        return discounted_rewards

    def train(self):

        if self.args.optimizer == "adam":
            self.optimizer = Adam(lr = self.args.lr)
        
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        
        total_loss = []
        all_rewards = []
        
        for episode in range(self.episodes_done, self.args.num_episodes):
            ## Generate Episodes
            states, actions, rewards = self.generate_episode(self.env, self.args.render)
            discounted_rewards = self.discounted_rewards(rewards)

            states = np.array(states)
            target = np.zeros((len(states),self.nA))
            target[np.arange(len(states)), np.array(actions)] = np.array(discounted_rewards)
            
            loss = self.model.train_on_batch(states, target)
            
            total_loss.append(loss)
            all_rewards.append(np.sum(rewards)*1e2)

            if (episode + 1) % self.args.log_every == 0:
                if self.args.verbose:
                    print("Num Episodes: {0}, Train Reward: {1} +/- {2}, Loss: {3}".
                          format(episode+1, np.mean(all_rewards), np.std(all_rewards), np.mean(total_loss)))
                total_loss = []
                all_rewards = []
                          
            if (not self.args.trial) and(episode + 1) % self.args.eval_after == 0:
                if self.args.verbose:
                    print("Saving Model Weights")
                self.model.save_weights(os.path.join(self.args.model_path, str(episode+1)))

            sys.stdout.flush()
                
    def test_episode(self, num_test_episodes = 100):
        
        all_rewards = []                 
        for i in range(num_test_episodes):
            _ ,_ , rewards = self.generate_episode(self.env, self.args.render)
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
        
        current_state = env.reset()
        is_terminal  = False
        while not is_terminal:
            action_distribution = self.model.predict(np.expand_dims(current_state, 0))
            action = np.random.choice(env.action_space.n, 1, 
                                      p=action_distribution.squeeze(0))[0]
            next_state, reward, is_terminal, _ = env.step(action)
            states.append(current_state)
            actions.append(action)
            rewards.append(reward*1e-2)
            current_state = next_state
            
        return states, actions, rewards
    
    def plot_performance(self, performance):
        
        X = [x[0] for x in performance]
        Y = [x[1] for x in performance]
        Z = [x[2] for x in performance]
        
        plt.figure()
        plt.errorbar(X, Y, yerr = Z, ecolor='b')
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.title("Variation of reward with number of training episodes")
        plt.savefig(self.args.plot_path)


def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config-path', dest='model_config_path', type=str,
                        default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--result_path', dest='result_path',type=str, 
                        default='reinforce_keras',
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

    parser.add_argument('--lr', dest='lr', type=float, default=5e-4, 
                        help="The learning rate.")
    parser.add_argument('--optimizer', type=str, dest='optimizer', 
                        default='adam',
                        help="Optimizer to be Used")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    args.model_path = os.path.join(args.result_path, 
                              'reinforce_model' +str(args.run))
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    args.plot_path = os.path.join(args.result_path, 
                              'reinforce_plot' +str(args.run)+ '.png')
    
    return args


def main(args):
    
    # Parse command-line arguments.
    args = parse_arguments()
    
    # Create the environment.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    
    reinforce = Reinforce(env, args)
    
    if args.mode == 'train':
        reinforce.train()
        
    elif args.mode == 'test':
        reinforce.test()    

if __name__ == '__main__':
    main(sys.argv)
