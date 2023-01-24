import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ICM import ICMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def estm(self, state, action):
        sa = torch.cat([state, action]).to(device)

        q1 = F.relu(self.l1(sa)).to(device)
        q1 = F.relu(self.l2(q1)).to(device)
        q1 = self.l3(q1).to(device)

        q2 = F.relu(self.l4(sa)).to(device)
        q2 = F.relu(self.l5(q2)).to(device)
        q2 = self.l6(q2).to(device)
        return q1, q2


class MA_TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor1 = Actor(state_dim, action_dim, max_action).to(device)
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=3e-4)

        self.actor2 = Actor(state_dim, action_dim, max_action).to(device)
        self.actor2_target = copy.deepcopy(self.actor2)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0



    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature,
                                                 reduction='none').mean(-1)
        return intrinsic_reward.data.cpu().numpy()

    def select_action1(self, state):

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor1(state).cpu().data.numpy().flatten()

    def select_action2(self, state):

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor2(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, w1, w2, batch_size, k, flag, actor_flag=0):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

        k = float(k)
        if flag == 1:
            next_action1 = (
                    self.actor1_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            next_action2 = (
                    self.actor2_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            target_Q11A1, target_Q12A1 = self.critic1_target(next_state, next_action1)
            target_Q21A2, target_Q22A2 = self.critic2_target(next_state, next_action2)
            #print('更新critic1')
            current_Q1, current_Q2 = self.critic1(state, action)
            target_Q1 = torch.min(target_Q11A1, target_Q12A1)
            average_Q = (target_Q21A2 + target_Q22A2) / 2
            # target_QA2 = torch.min(target_Q11A2, target_Q12A2)
            # target_Q1 = torch.max(target_QA1, target_QA2)
            k = float(k)
            #target_Q = torch.min(target_Q11A1, target_Q12A1)
            target_Q1 = target_Q1 * (1-k) + average_Q * k
            target_Q1 = reward + not_done * self.discount * target_Q1
            # Compute critic loss
            critic_loss1 = F.mse_loss(current_Q1, target_Q1) + F.mse_loss(current_Q2, target_Q1)
            self.critic1_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic1_optimizer.step()
            #if self.total_it % self.policy_freq == 0:

            if actor_flag == 1:
        
                actor1_loss = -self.critic1.Q1(state, self.actor1(state)).mean()
            # Optimize the actor
                self.actor1_optimizer.zero_grad()
                actor1_loss.backward()
                self.actor1_optimizer.step()
                for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if flag == 2:
 
            next_action1 = (
                    self.actor1_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            next_action2 = (
                    self.actor2_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q11A1, target_Q12A1 = self.critic1_target(next_state, next_action1)
            target_Q21A2, target_Q22A2 = self.critic2_target(next_state, next_action2)
            current_Q1, current_Q2 = self.critic2(state, action)
            target_Q2 = torch.min(target_Q21A2, target_Q22A2)
            average_Q = (target_Q11A1 + target_Q12A1) / 2
            target_Q2 = target_Q2 *(1-k) + average_Q * k
     
            target_Q2 = reward + not_done * self.discount * target_Q2

            critic_loss2 = F.mse_loss(current_Q1, target_Q2) + F.mse_loss(current_Q2, target_Q2)
            self.critic2_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic2_optimizer.step()

            if actor_flag == 2 :
                #print('更新actor2')
                # Compute actor losse
                actor2_loss = -self.critic2.Q1(state, self.actor2(state)).mean()
         
                self.actor2_optimizer.zero_grad()
                actor2_loss.backward()
                self.actor2_optimizer.step()



                for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def estimate_value(self, state, max_action, cf):
        with torch.no_grad():
            if cf == 1:
                action = self.select_action1(np.array(state)).clip(-max_action, max_action)
                action = torch.FloatTensor(action)
                state = torch.FloatTensor(state)
                q1, q2 = self.critic1.estm(state, action)

            else:
                action = self.select_action2(np.array(state)).clip(-max_action, max_action)
                action = torch.FloatTensor(action)
                state = torch.FloatTensor(state)
                q1, q2 = self.critic2.estm(state, action)
        return min(q1, q2)


def save(self, filename):
    torch.save(self.critic.state_dict(), filename + "_critic")
    torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    torch.save(self.actor.state_dict(), filename + "_actor")
    torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


def load(self, filename):
    self.critic.load_state_dict(torch.load(filename + "_critic"))
    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    self.critic_target = copy.deepcopy(self.critic)

    self.actor.load_state_dict(torch.load(filename + "_actor"))
    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    self.actor_target = copy.deepcopy(self.actor)


