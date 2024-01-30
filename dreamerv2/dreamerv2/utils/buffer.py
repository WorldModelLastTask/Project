import numpy as np 
import random, pickle
from collections import namedtuple, deque
from typing import Optional, Tuple

import torch
import torch.nn as nn



class MultiEnvTransitionBuffer():
    def __init__(
        self,
        n_envs,
        capacity,
        obs_shape: Tuple[int],
        action_size: int,
        seq_len: int, 
        batch_size: int,
        obs_type=np.float32,
        action_type=np.float32,
        nl_embedding_size = int,
    ):
        self.n_envs = n_envs
        self.buffers = [TransitionBuffer(capacity, obs_shape, action_size, seq_len, batch_size, obs_type, action_type, nl_embedding_size) for _ in range(n_envs)] 
    
    def add(
        self,
        obs: Tuple,
        action: list,
        reward: list,
        done: list,
    ):
        for i in range(self.n_envs):
            self.buffers[i].add([obs[0][i], obs[1][i]], action[i], reward[i], done[i])
    def sample(self):
        obs_img,obs_nl,act,rew,term = [],[],[],[],[]
        for i in range(self.n_envs):
            o,a,r,t = self.buffers[i].sample()
            obs_img.append(o[0])
            obs_nl.append(o[1])
            act.append(a)
            rew.append(r)
            term.append(t)
        # return np.stack(obs, axis=1), np.stack(act, axis=1), np.stack(rew, axis=1), np.stack(term, axis=1)
        return [np.concatenate(obs_img, axis=1), np.concatenate(obs_nl, axis=1)], np.concatenate(act, axis=1), np.concatenate(rew, axis=1), np.concatenate(term, axis=1)
    
    def save(self, filename):
        for i in range(self.n_envs):
            self.buffers[i].save(filename+str(i)+'.pkl')
    def load(self, filename):
        for i in range(self.n_envs):
            self.buffers[i].load(filename+str(i)+'.pkl')
class TransitionBuffer():
    def __init__(
        self,
        capacity,
        obs_shape: Tuple[int],
        action_size: int,
        seq_len: int, 
        batch_size: int,
        obs_type=np.float32,
        action_type=np.float32,
        nl_embedding_size = int,
    ):

        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observation = np.empty((capacity, *obs_shape), dtype=obs_type) 
        # self.nl_obs_list = np.empty((capacity, max_length), dtype=np.int32)
        self.embed_nl_obs = torch.empty((capacity, 1, nl_embedding_size), dtype=torch.float32)
        # self.nl_obs_idx = np.empty((capacity,), dtype=np.int32)
        self.action = np.empty((capacity, action_size), dtype=np.uint8)
        self.reward = np.empty((capacity,), dtype=np.int8) 
        self.terminal = np.empty((capacity,), dtype=bool)
    
    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 
    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()
    

    def add(
        self,
        obs: Tuple,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        assert action.min() > 0, "action should be 1-indexed"
        self.observation[self.idx] = obs[0]
        self.embed_nl_obs[self.idx] = obs[1]
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _sample_idx(self, L):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.capacity
            valid_idx = not self.idx in idxs[1:] 
        return idxs

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.observation[vec_idxs]
        nl_obs = self.embed_nl_obs[vec_idxs]
        return[observation.reshape(l, n, *self.obs_shape), nl_obs.reshape(l,n,-1)], self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), self.terminal[vec_idxs].reshape(l, n)

    def sample(self):
        n = self.batch_size
        l = self.seq_len+1
        obs,act,rew,term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        obs,act,rew,term = self._shift_sequences(obs,act,rew,term)
        return obs,act,rew,term
    
    def _shift_sequences(self, obs, actions, rewards, terminals):
        obs_img = obs[0][1:]
        obs_nl = obs[1][1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]
        return [obs_img, obs_nl], actions, rewards, terminals

#Following objects are not used

Episode = namedtuple('Episode', ['observation', 'action', 'reward', 'terminal', 'length'])  

class EpisodicBuffer():
    """
    :params total_episodes: maximum no of episodes capacity  
    """
    def __init__(
        self,
        total_episodes,
        obs_shape: Tuple[int],
        action_size: int,
        seq_len: int, 
        batch_size: int,
        obs_type=np.float32,
        action_type=np.float32,
    ):
        self.total_episodes = total_episodes
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer = deque([], maxlen=total_episodes)
        self.lengths = deque([], maxlen=total_episodes)
        self._full = False
        self._episode_cnt = 0
        self._init_episode()
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        last_obs: Optional[np.ndarray] = None
    ):
        self.observation.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.terminal.append(done)
        
        if done:
            assert last_obs is not None
            self.observation.append(last_obs)
            self.add_episode()
    
    def sample(self):
        seq_len = self.seq_len
        batch_size = self.batch_size
        episode_list = random.choices(self.buffer, k=batch_size)
        obs_batch = np.empty([seq_len, batch_size, *self.obs_shape], dtype=self.obs_type)
        act_batch = np.empty([seq_len, batch_size, self.action_size], dtype=self.action_type)
        rew_batch = np.empty([seq_len, batch_size], dtype=np.float32)
        term_batch = np.empty([seq_len, batch_size], dtype=bool)
        for ind, episode in enumerate(episode_list):
            obs_batch[:,ind], act_batch[:,ind], rew_batch[:,ind], term_batch[:,ind] = self._sample_seq(episode, seq_len)
        return obs_batch, act_batch, rew_batch , term_batch
    
    def _sample_seq(self, episode, seq_len):
        assert episode.length>=seq_len
        s = min(np.random.choice(episode.length), episode.length-seq_len)
        return (
                episode.observation[s:s+seq_len], 
                episode.action[s:s+seq_len], 
                episode.reward[s:s+seq_len], 
                episode.terminal[s:s+seq_len]
            )
    
    def add_episode(self):
        assert self.terminal[-1] == True
        e = Episode(*self._episode_to_array(), len(self.terminal))
        self.buffer.append(e)
        self._episode_cnt += 1
        if self._episode_cnt == self.total_episodes:
            self.full = True 
        self._init_episode()
    
    def _init_episode(self):
        self.observation = [] 
        self.action = [np.zeros(self.action_size, dtype=self.action_type)] 
        self.reward = [0.0]
        self.terminal = [False]
    
    def _episode_to_array(self):
        o = np.stack(self.observation, axis=0)
        a = np.stack(self.action, axis=0)
        r = np.stack(self.reward, axis=0)
        nt = np.stack(self.terminal, axis=0)
        return o,a,r,nt

    @property
    def episode_count(self):
        return self._episode_cnt

class FluidEpisodicBuffer():
    """
    :params total_episodes: 
    """
    def __init__(
        self,
        total_episodes: int,
        obs_shape: Tuple[int],
        action_size: int,
        seq_len: int, 
        batch_size: int,
        minimum_episode_len: Optional[int] = 1,
        obs_type: Optional[np.dtype] = np.uint8,
        action_type: Optional[np.dtype] = np.float32,
        incr_len: Optional[int] = 5,
    ):
        self.total_episodes = total_episodes
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.obs_type = obs_type
        self.action_type = action_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.incr_len = incr_len

        self.buffer = deque([], maxlen=total_episodes)
        self.lengths = deque([], maxlen=total_episodes)
        
        self._minimum_episode_len = minimum_episode_len
        self.opt_frac = 0.1*total_episodes
        self.opt_seq_len = minimum_episode_len
        self._init_episode()
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        last_obs: Optional[np.ndarray] = None
    ):

        self.observation.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.terminal.append(done)
        
        if done:
            assert last_obs is not None
            self.observation.append(last_obs)
            self.add_episode()
    
    def sample(self):
        seq_len = self.opt_seq_len
        batch_size = self.batch_size
        obs_batch = np.empty([seq_len, batch_size, *self.obs_shape], dtype=self.obs_type)
        act_batch = np.empty([seq_len, batch_size, self.action_size], dtype=self.action_type)
        rew_batch = np.empty([seq_len, batch_size], dtype=np.float32)
        term_batch = np.empty([seq_len, batch_size], dtype=bool)
        episode_idx = np.random.choice(np.where(np.array(self.lengths)>=seq_len)[0], size=batch_size)
        for i,idx in enumerate(episode_idx):
            obs_batch[:,i], act_batch[:,i], rew_batch[:,i], term_batch[:,i] = self._sample_seq(self.buffer[idx], self.lengths[idx], seq_len)
        return obs_batch, act_batch, rew_batch , term_batch
    
    def _sample_seq(self, episode, episode_len, seq_len):
        s = min(episode_len, episode_len-seq_len)
        return (
                episode.observation[s:s+seq_len], 
                episode.action[s:s+seq_len], 
                episode.reward[s:s+seq_len], 
                episode.terminal[s:s+seq_len]
            )
    
    def add_episode(self):
        assert self.terminal[-1] == True
        e = Episode(*self._episode_to_array())
        if len(self.terminal)>=self.opt_seq_len:  
            self.buffer.append(e)
            self.lengths.append(len(self.terminal))
        self._init_episode()
        self._set_opt_len()
    
    def _set_opt_len(self):
        temp_len = np.array(self.lengths)
        if np.sum(temp_len>self.opt_seq_len+self.incr_len)>self.opt_frac:
            self.opt_seq_len = min(self.opt_seq_len+self.incr_len, self.seq_len)
    
    def _init_episode(self):
        self.observation = [] 
        self.action = [np.zeros(self.action_size, dtype=self.action_type)] 
        self.reward = [0.0]
        self.terminal = [False]
    
    def _episode_to_array(self):
        o = np.stack(self.observation, axis=0)
        a = np.stack(self.action, axis=0)
        r = np.stack(self.reward, axis=0)
        nt = np.stack(self.terminal, axis=0)
        return o,a,r,nt

