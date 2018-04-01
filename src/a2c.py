import sys
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce, ActorNetwork


## experiment with different critic networks
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(CriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim = output_dim
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        for name, param in self.layer1.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform(param)
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        for name, param in self.layer2.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform(param)
        self.layer3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        for name, param in self.layer3.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform(param)
        self.layer4 = nn.Linear(self.hidden_dim3, self.output_dim)
        for name, param in self.layer4.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform(param)

    def forward(self, input):
        hidden1 = self.layer1(input)
        hidden1 = F.relu(hidden1)
        hidden2 = self.layer2(hidden1)
        hidden2 = F.relu(hidden2)
        hidden3 = self.layer3(hidden2)
        hidden3 = F.relu(hidden3)
        output = self.layer4(hidden3) ## final linear layer projects hidden state into a single value representation
        return output

class A2C(Reinforce):
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.model = ActorNetwork(self.env.observation_space.shape[0], self.args.hidden1, self.args.hidden2,
                             self.args.hidden3, self.env.action_space.n)
        self.lr = args.lr
        self.critic_model = CriticNetwork(self.env.observation_space.shape[0], self.args.hidden1, self.args.hidden2,
                             self.args.hidden3, 1) ##Value function
        self.critic_lr = self.args.critic_lr
        self.gamma = self.args.gamma
        self.n = self.args.n
        self.critic_criterion = nn.MSELoss()

    def actor_loss(self, rewards, values, action_probabilities, action_mask):
        masked_action_probabilities = action_mask * action_probabilities
        episode_action_probabilities = torch.sum(masked_action_probabilities, dim=1)
        advantage = rewards - values
        batch_loss = torch.mean(advantage * torch.log(episode_action_probabilities))
        return -batch_loss

    def critic_loss(self, rewards, values):
        mse_loss = self.critic_criterion(values, rewards)
        return mse_loss

    def truncated_discounted_rewards(self, rewards):
        batch_size = len(rewards)-self.n
        truncated_rewards = np.zeros(batch_size)
        for t in range(batch_size):
            cumulative = 0
            for i in range(0, self.n):
                cumulative += math.pow(self.gamma, i)*rewards[t + i]
            truncated_rewards[t] = cumulative
        return truncated_rewards

    def get_value_reward(self, states, rewards):
        batch_size = len(rewards)
        states_variable = torch.stack(states)
        values = self.critic_model(states_variable)

        ## variable that is used on the actor loss is volatile since value function is considered fixed
        extended_values = Variable(torch.zeros((batch_size + self.n)), volatile=True)
        extended_values[range(batch_size)] = values

        extended_rewards = rewards + [0]*self.n
        truncated_discounted_rewards = self.truncated_discounted_rewards(extended_rewards)
        discounted_rewards = np.zeros_like(rewards)
        for t in reversed(range(batch_size)):
            discounted_rewards[t] = math.pow(self.gamma,self.n)*extended_values[t+self.n] + truncated_discounted_rewards[t]
        ## before function exit, reset volatile
        extended_values.volatile = False
        return extended_values[:-self.n], values, discounted_rewards

    def test(self):
        self.model.train(False)
        self.critic_model.train(False)
        average_reward = 0
        for i in range(100):
            _, _, _, rewards = self.generate_episode(self.env, self.args.render)
            episode_reward = np.sum(rewards)
            average_reward += episode_reward * 1e2
        average_reward = float(average_reward) / 100
        self.model.train(True)
        self.critic_model.train(True)
        return average_reward

    def train(self):
        if self.args.optimizer == "adam":
            self.actor_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
        self.recorded_performance = []
        denominator = 0
        total_actor_loss = 0
        total_critic_loss = 0
        for episode in range(self.args.num_episodes):
            ## egnerate episode
            states, action_probabilities, actions, rewards = self.generate_episode(self.env, self.args.render)
            values_for_actor, values, rewards = self.get_value_reward(states, rewards)

            ##create a batch
            mask = np.zeros((len(actions), self.env.action_space.n))
            mask[np.arange(len(actions)), actions] = 1
            action_mask = Variable(torch.FloatTensor(mask))
            action_probabilities = torch.stack(action_probabilities)
            rewards_variable = Variable(torch.FloatTensor(rewards))

            ## batch backpropogate
            a_loss = self.actor_loss(rewards_variable, values_for_actor, action_probabilities, action_mask)
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            c_loss = self.critic_loss(rewards_variable, values)
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            self.critic_optimizer.step()

            denominator += len(actions)
            total_actor_loss += a_loss
            total_critic_loss += c_loss

            if self.args.verbose:
                print("Actor Loss after {0} updates: {1}".format(denominator, float(total_actor_loss)/denominator))
                print("Critic Loss after {0} updates: {1}".format(denominator, float(total_critic_loss) / denominator))

                ## test after every 100 episodes
            if (episode + 1) % self.args.eval_after == 0:
                average_reward = self.test()
                self.recorded_performance.append([episode + 1, average_reward])
                print("Average reward after {0} episodes: {1}".format(episode + 1, average_reward))


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_path', dest='plot_path',
                        type=str, default='plot.png',
                        help="Path to the plot.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--eval_after', dest='eval_after', type=int,
                        default=500, help="Number of episodes to evaluate after.")
    parser.add_argument('--gamma', type=float, dest='gamma', default=0.95)


    # Action store
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--verbose', dest='verbose',
                              action='store_true',
                              help="Whether to print loss after every episode.")
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)


    ## A2C related arguments
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--optimzer', type=str, dest='optimizer', default='adam')
    parser.add_argument('--hidden1', dest='hidden1', type=int, default=16)
    parser.add_argument('--hidden2', dest='hidden2', type=int, default=16)
    parser.add_argument('--hidden3', dest='hidden3', type=int, default=16)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()

    # Create the environment.
    env = gym.make('LunarLander-v2')
    a2c = A2C(env, args)

    a2c.train()


if __name__ == '__main__':
    main(sys.argv)
