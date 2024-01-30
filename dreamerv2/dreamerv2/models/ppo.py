import gc
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torchvision.models import resnet18
from transformers import AutoModel
from tqdm import tqdm
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super().__init__()
        self.action_dim = action_dim
        self.actors = nn.ModuleList([Actor(state_dim+i, action_dim[i]) for i in range(3)])

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # img, ln = state[0], state[1]
        actions = torch.zeros(state[0].size(0), 3, device=device)
        action_logprobs = torch.zeros(state[0].size(0), 3, device=device)
        states = self.encoder(state)
        state_val = self.critic(self.encoder(state)).detach()
        for i in range(3):

            action_probs = self.actors[i](states)
            dist = Categorical(action_probs)

            action = dist.sample()
            action_logprob = dist.log_prob(action)
            # print("act", states.shape, action.shape)
            states = torch.cat([states, action.unsqueeze(1)], dim=-1)
            
            actions[:, i] = action.detach()
            # print("act", action_logprob.shape, action_logprobs.shape)
            action_logprobs[:, i] = action_logprob.detach()

        return actions, action_logprobs, state_val
    
    def evaluate(self, state, action): #state->(batch, 3, 160, 160), action->(batch, 3)
        action_logprobs = torch.zeros(state[0].size(0), 3, device=device)
        dist_entropies = torch.zeros(state[0].size(0), 3, device=device)
        states = self.encoder(state)
        state_values  = self.critic(states)

        for i in range(3):
            action_prob = self.actors[i](states)
            dist = Categorical(action_prob)

            action_logprob = dist.log_prob(action[:, i])
            dist_entropy = dist.entropy()
            states = torch.cat([states, action[:,i].unsqueeze(1)], dim=-1)
            action_logprobs[:, i] = action_logprob
            dist_entropies[:, i] = dist_entropy
        
        return action_logprobs, state_values, dist_entropies


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # self.buffer = ReplayBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actors.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.update_timestep = 0
    def eval(self):
        self.policy.eval()
        self.policy_old.eval()
    def train(self):
        self.policy.train()
        self.policy_old.train()
    def select_action(self, state):

        with torch.no_grad():
            state = [s.to(device) for s in state]
            action, action_logprob, state_val = self.policy_old.act(state)

        return action, action_logprob, state_val
    
    def one_epoch(self, writer, old_states, old_actions, old_logprobs, old_state_values, buffer_rewards, is_terminals):
        
        # convert list to tensor
        old_states[0] = old_states[0].detach().to(device)
        old_states[1] = old_states[1].detach().to(device).long()
        old_actions = old_actions.detach().to(device)
        old_logprobs = old_logprobs.detach().to(device)
        old_state_values = old_state_values.squeeze().detach().to(device)
        buffer_rewards = buffer_rewards.squeeze().detach().to(device)
        is_terminals = is_terminals.squeeze().detach().to(device)
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer_rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Evaluating old actions and values
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        #logprobs -> (batch, 3), dist_entropy -> (batch, 3)
        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)
        surr_loss = 0
        entropy_loss = 0
        for i in range(3):
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs[:,i] - old_logprobs[:,i].detach())
            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            surr_loss += - torch.min(surr1, surr2)
            entropy_loss += - dist_entropy[:, i] 
        surr_loss *= 1/3
        entropy_loss *= 1/3 *0.01
        value_loss = 0.5 * self.MseLoss(state_values, rewards)
        # final loss of clipped objective PPO
        loss = surr_loss + value_loss + entropy_loss
        writer.add_scalar('loss/main', loss.mean().item(), self.update_timestep)
        writer.add_scalar('loss/surr', surr_loss.mean().item(), self.update_timestep)
        writer.add_scalar('loss/entropy', entropy_loss.mean().item(), self.update_timestep)
        writer.add_scalar('loss/mse', value_loss.mean().item(), self.update_timestep)
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.update_timestep += 1

    def update(self, writer, buffer, n_envs, batch_size, chunk_length): # change sample method
        if buffer.__len__() < batch_size:
            return

        # Optimize policy for K epochs
        for _ in tqdm(range(self.K_epochs)):
            
            old_states, old_actions, old_logprobs, old_state_values, buffer_rewards, is_terminals = buffer.sample(batch_size, chunk_length, is_chunk=False)
            # convert list to tensor
            old_states[0] = old_states[0].detach().to(device)
            old_states[1] = old_states[1].detach().to(device).long()
            old_actions = old_actions.detach().to(device)
            old_logprobs = old_logprobs.detach().to(device)
            old_state_values = old_state_values.squeeze().detach().to(device)
            buffer_rewards = buffer_rewards.squeeze().detach().to(device)
            is_terminals = is_terminals.squeeze().detach().to(device)
            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(buffer_rewards), reversed(is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # calculate advantages
            advantages = rewards.detach() - old_state_values.detach()

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            #logprobs -> (batch, 3), dist_entropy -> (batch, 3)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            surr_loss = 0
            entropy_loss = 0
            for i in range(3):
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs[:,i] - old_logprobs[:,i].detach())
                # Finding Surrogate Loss  
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                surr_loss += - torch.min(surr1, surr2)
                entropy_loss += - dist_entropy[:, i] 
            surr_loss *= 1/3
            entropy_loss *= 1/3 *0.01
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            # final loss of clipped objective PPO
            loss = surr_loss + value_loss + entropy_loss
            writer.add_scalar('loss/main', loss.mean().item(), self.update_timestep)
            writer.add_scalar('loss/surr', surr_loss.mean().item(), self.update_timestep)
            writer.add_scalar('loss/entropy', entropy_loss.mean().item(), self.update_timestep)
            writer.add_scalar('loss/mse', value_loss.mean().item(), self.update_timestep)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.update_timestep += 1
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        # self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
