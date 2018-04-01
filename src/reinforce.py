import sys
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        ## network configuration (16,16,16,4)
        ## keras configuration (match Variance Initializer with -1,1 and fan_avg)
        super(Network, self).__init__()
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
        output = self.layer4(hidden3)
        output = F.softmax(output)
        return output

    def loss(self):
        return

    def evaluate(self, input):
        return

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, env, args):
        # TODO: Define any training operations and optimizers here, initialize
        self.env = env
        self.args = args
        self.model = Network(self.env.observation_space.shape[0], self.args.hidden1, self.args.hidden2,
                                   self.args.hidden3, self.env.action_space.n)
        self.lr = args.lr
        self.gamma = args.gamma


    def loss(self, action_probabilities, action_mask, discounted_rewards):
        masked_action_probabilities = action_mask*action_probabilities
        episode_action_probabilities = torch.sum(masked_action_probabilities, dim=1)
        batch_loss = torch.mean(discounted_rewards*torch.log(episode_action_probabilities))
        return -batch_loss

    def train(self):
        if self.args.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.recorded_performance = []
        denominator = 0
        total_loss = 0
        for episode in range(self.args.num_episodes):

            ## generate episode
            action_probabilities, actions, rewards = self.generate_episode(self.env, self.args.render)
            discounted_rewards = self.discounted_rewards(rewards)

            ##create a batch
            mask = np.zeros((len(actions), self.env.action_space.n))
            mask[np.arange(len(actions)), actions] = 1
            action_mask = Variable(torch.FloatTensor(mask))
            action_probabilities = torch.stack(action_probabilities)
            discounted_reward_variable = Variable(torch.FloatTensor(discounted_rewards))

            ## batch backpropogate
            loss = self.loss(action_probabilities, action_mask, discounted_reward_variable)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            denominator += len(actions)
            total_loss += loss
            if self.args.verbose:
                print("Loss(Negative Likelihood) after {0} updates: {1}".format(denominator, float(total_loss)/denominator))

            ## test after every 100 episodes
            if (episode + 1) % self.args.eval_after == 0:
                average_reward = self.test()
                self.recorded_performance.append([episode+1, average_reward])
                print("Average reward after {0} episodes: {1}".format(episode+1, average_reward))

    def plot_performance(self):
        X = [x[0] for x in self.recorded_performance]
        Y = [x[1] for x in self.recorded_performance]
        plt.figure()
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.title("Variation of reward with number of training episodes")
        plt.plot(X, Y)
        plt.savefig(self.args.plot_path)


    def test(self,):
        self.model.train(False)
        average_reward = 0
        for i in range(100):
            states, actions, rewards = self.generate_episode(self.env, self.args.render)
            episode_reward = np.sum(rewards)
            average_reward += episode_reward*1e2
        average_reward = float(average_reward)/100
        self.model.train(True)
        return average_reward

    def discounted_rewards(self, rewards):
        ## implementation different from standards
        T = len(rewards)
        discounted_rewards = np.zeros_like(rewards)
        discounted_rewards[T-1] = rewards[T -1]
        for t in reversed(range(0, T-1)):
            discounted_rewards[t] = discounted_rewards[t+1] * self.gamma + rewards[t]
        return discounted_rewards

    def generate_episode(self, env, render=False, test_time = False):
        action_distributions = []
        actions = []
        rewards = []
        current_state = env.reset()
        current_state_variable = Variable(torch.FloatTensor(current_state))
        episode_over  = False
        while episode_over is not True:
            action_distribution = self.model(current_state_variable)
            action = np.random.choice(env.action_space.n, 1, p=action_distribution.data.numpy())[0]
            # action = torch.multinomial(action_distribution, 1).data.numpy()
            next_state, reward, episode_over, _ = env.step(action)
            action_distributions.append(action_distribution)
            actions.append(action)
            rewards.append(reward*1e-2)
            current_state = next_state
            current_state_variable = Variable(torch.FloatTensor(current_state))
        return action_distributions, actions, rewards


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
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--verbose', dest='verbose',
                              action='store_true',
                              help="Whether to print loss after every episode.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)


    ## policy network specific arguments
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
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
    reinforce = Reinforce(env, args)

    reinforce.train()

if __name__ == '__main__':
    main(sys.argv)
