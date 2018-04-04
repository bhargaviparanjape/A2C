from __future__ import print_function

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl


'''
 Matching Keras Model Configuration, MultiLayer Perceptron with (16,16,16)
 Hidden Units and Variance Scaling with Xavier Uniform
'''
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = 16
        self.hidden_dim2 = 16
        self.hidden_dim3 = 16
        self.output_dim = output_dim

        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.weight_init(self.layer1)
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.weight_init(self.layer2)
        self.layer3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.weight_init(self.layer3)
        self.layer4 = nn.Linear(self.hidden_dim3, self.output_dim)
        self.weight_init(self.layer4)
                
    def weight_init(self, layer):
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform(param)

    def forward(self, input_var):
        hidden1 = F.relu(self.layer1(input_var))
        hidden2 = F.relu(self.layer2(hidden1))
        hidden3 = F.relu(self.layer3(hidden2))
        # Batch_size = 1 is assumed always.
        output = F.softmax(self.layer4(hidden3), dim=0)
        return output

'''
 Implementation of the policy gradient method REINFORCE.
'''
class Reinforce(object):

    def __init__(self, env, args):

        self.env = env
        self.args = args
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.model = ActorNetwork(self.nS, self.nA)
        
        # Model to be resumed. Update stored states
        if args.resume and os.path.exists(self.args.model_path):
            print('*'*80)
            print ('Resuming Training from last Checkpoint')
            print('*'*80)
            self.model.load_state_dict(torch.load(self.args.model_path))
            self.recorded_performance, self.episodes_done, self.rden = (
                                        pkl.load(open(self.args.meta_path,'rb')))
        else:
            self.recorded_performance, self.episodes_done, self.rden = [], 0, 0

        self.lr = args.lr
        self.gamma = args.gamma

    def loss(self, action_probabilities, action_mask, discounted_rewards):
        masked_action_probabilities = action_mask*action_probabilities
        episode_action_probabilities = torch.sum(masked_action_probabilities, dim=1)
        batch_loss = -torch.mean(discounted_rewards*torch.log(episode_action_probabilities))
        return batch_loss

    def train(self):
        
        if self.args.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        denominator = self.rden
        total_loss = 0.

        for episode in range(self.episodes_done,self.args.num_episodes):
            ## Generate Episodes
            states, action_probabilities, actions, rewards = (
                                        self.generate_episode(self.env, self.args.render))
            discounted_rewards = self.discounted_rewards(rewards)

            ## Create a Batch
            mask = np.zeros((len(actions), self.env.action_space.n))
            mask[np.arange(len(actions)), actions] = 1
            
            action_mask = Variable(torch.FloatTensor(mask))
            action_probabilities = torch.stack(action_probabilities)
            discounted_reward_variable = Variable(torch.FloatTensor(discounted_rewards))

            ## Batch Train
            loss = self.loss(action_probabilities, action_mask, discounted_reward_variable)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            denominator += len(actions)
            total_loss += loss

            if (episode + 1) % self.args.log_every == 0:
                if self.args.verbose:
                    print("Loss (Negative Likelihood) after {0} episodes, {1} updates: {2}".
                          format(episode+1, denominator, float(total_loss)/(self.args.log_every)))
                total_loss = 0.

            ## Test after every k episodes
            if (episode + 1) % self.args.eval_after == 0:
                average_reward, std_reward = self.test()
                if not self.args.trial and (len(self.recorded_performance) == 0 or
                    average_reward > self.recorded_performance[-1][1]):
                    torch.save(self.model.state_dict(),self.args.model_path)
                    pkl.dump((self.recorded_performance, episode+1, denominator),
                                                    open(self.args.meta_path,'wb'))
                self.recorded_performance.append([episode+1, average_reward, std_reward])
                print('*'*80)
                print("Average reward after {0} episodes: {1} +/- {2}".format(episode+1, 
                                                              average_reward, std_reward))
                print('*'*80)
            
            sys.stdout.flush()
        if not self.args.trial:
            self.plot_performance()
                
    def test(self, num_test_episodes = 100):
        
        all_rewards = []
        for i in range(num_test_episodes):
            _ ,_ ,_ , rewards = self.generate_episode(self.env, self.args.render)
            episode_reward = np.sum(rewards)
            all_rewards.append(episode_reward*1e2)
        
        average_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        return average_reward, std_reward

    def discounted_rewards(self, rewards):
        
        T = len(rewards)
        discounted_rewards = np.zeros_like(rewards)
        discounted_rewards[T-1] = rewards[T -1]
        for t in reversed(range(0, T-1)):
            discounted_rewards[t] = (
                            discounted_rewards[t+1] * self.gamma + rewards[t])
        return discounted_rewards

    def generate_episode(self, env, render = False, test_time = False):
        
        action_distributions = []
        actions = []
        rewards = []
        states = []
        
        current_state = env.reset()
        is_terminal  = False
        
        while not is_terminal:
            
            current_state_variable = Variable(torch.FloatTensor(current_state))
            
            action_distribution = self.model(current_state_variable)
            action = np.random.choice(env.action_space.n, 1, 
                      p=action_distribution.data.numpy())[0]

            next_state, reward, is_terminal, _ = env.step(action)
            
            states.append(current_state_variable)
            action_distributions.append(action_distribution)
            actions.append(action)
            rewards.append(reward*1e-2)  # Reward Downscaled
            
            current_state = next_state
            
        return states, action_distributions, actions, rewards
    
    def plot_performance(self):
        
        X = [x[0] for x in self.recorded_performance]
        Y = [x[1] for x in self.recorded_performance]
        Z = [x[2] for x in self.recorded_performance]
        
        plt.figure()
        plt.errorbar(X, Y, yerr = Z, ecolor='b')
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.title("Variation of reward with number of training episodes")
        plt.savefig(self.args.plot_path)


def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--plot_path', dest='plot_path',type=str, 
                        default='reinforce/reinforce_plot.png',
                        help="Path to the plot.")
    parser.add_argument('--model_path', dest='model_path',type=str, 
                        default='reinforce/reinforce_model.pt',
                        help="Path to the model.")
    parser.add_argument('--meta_path', dest='meta_path',type=str, 
                        default='reinforce/reinforce_meta.p',
                        help="Path to meta information for model.")
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
                        default=0.95)
    parser.add_argument('--seed', type=int, dest='seed', default=0)
    parser.add_argument('--trial', dest='trial', action='store_true',
                         help="If it is just a trial")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                         help="Whether to print loss after every episode.")
    
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
    
    return parser.parse_args()


def main(args):
    
    # Parse command-line arguments.
    args = parse_arguments()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create the environment.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    
    reinforce = Reinforce(env, args)
    reinforce.train()

if __name__ == '__main__':
    main(sys.argv)
