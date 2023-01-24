import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


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

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)

	# def estm(self, state, action):
	# 	state = state.to(device)
	# 	q = F.relu(self.l1(state))
	# 	q = F.relu(self.l2(torch.cat([q, action])))
	# 	return self.l3(q)
	#

	def estm(self, state, action):
		state = state.to(device)
		action = action.to(device)
		q = F.relu(self.l1(torch.cat([state, action])))
		q = F.relu(self.l2(q))
		q = self.l3(q)
		return q


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
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

		self.discount = discount
		self.tau = tau


	def select_action1(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor1(state).cpu().data.numpy().flatten()

	def select_action2(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor2(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size, flag):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		if flag == 1 :

			target_Q = self.critic1_target(next_state, self.actor1_target(next_state))
			target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
			current_Q = self.critic1(state, action)

		# Compute critic loss
			critic1_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
			self.critic1_optimizer.zero_grad()
			critic1_loss.backward()
			self.critic1_optimizer.step()

		# Compute actor loss
			actor_loss = -self.critic1(state, self.actor1(state)).mean()
		
		# Optimize the actor 
			self.actor1_optimizer.zero_grad()
			actor_loss.backward()
			self.actor1_optimizer.step()

		# Update the frozen target models
			for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		elif flag == 2:

			target_Q = self.critic2_target(next_state, self.actor2_target(next_state))
			target_Q = reward + (not_done * self.discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic2(state, action)

			# Compute critic loss
			critic2_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic2_optimizer.zero_grad()
			critic2_loss.backward()
			self.critic2_optimizer.step()

			# Compute actor loss
			actor2_loss = -self.critic2(state, self.actor2(state)).mean()

			# Optimize the actor
			self.actor2_optimizer.zero_grad()
			actor2_loss.backward()
			self.actor2_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




	def save(self, filename,flag):
		if flag == 1 :
			torch.save(self.critic1.state_dict(), filename + "_critic")
			torch.save(self.critic1_optimizer.state_dict(), filename + "_critic_optimizer")

			torch.save(self.actor1.state_dict(), filename + "_actor")
			torch.save(self.actor1_optimizer.state_dict(), filename + "_actor_optimizer")
		elif flag == 2 :

			torch.save(self.critic2.state_dict(), filename + "_critic")
			torch.save(self.critic2_optimizer.state_dict(), filename + "_critic_optimizer")

			torch.save(self.actor2.state_dict(), filename + "_actor")
			torch.save(self.actor2_optimizer.state_dict(), filename + "_actor_optimizer")


	def estimate_value(self, state, max_action,flag):

		with torch.no_grad():
			if flag == 1:
				action = self.select_action1(np.array(state)).clip(-max_action, max_action)
				action = torch.FloatTensor(action)
				state = torch.FloatTensor(state).to(device)
				q1 = self.critic1.estm(state, action)
			elif flag == 2:
				action = self.select_action2(np.array(state)).clip(-max_action, max_action)
				action = torch.FloatTensor(action)
				state = torch.FloatTensor(state).to(device)
				q1 = self.critic2.estm(state, action)
		return q1

