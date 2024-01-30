import torch 
import torch.nn as nn
import numpy as np

class MultiDiscreteActionModel(nn.Module):
    def __init__(
        self,
        act_shape,
        deter_size,
        stoch_size,
        nl_embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        self.act_shape = act_shape
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.nl_embedding_size = nl_embedding_size
        self.models = self._build_models(actor_info, expl_info)
    
    def _build_models(self, actor_info, expl_info):
        models = nn.ModuleList()
        for action_size, info in zip(self.act_shape, actor_info):
            model = DiscreteActionModel(action_size, self.deter_size, self.stoch_size, self.nl_embedding_size, info.default_factory(), expl_info)
            models.append(model)
        return models
    
    def forward(self, model_state):
        action_list = []
        action_dist_list = []
        for model in self.models:
            action, action_dist = model(model_state)
            action_list.append(action)
            action_dist_list.append(action_dist)
        return action_list, action_dist_list
    
    def get_action_dist(self, model_state):
        action_dist_list = []
        for model in self.models:
            action_dist = model.get_action_dist(model_state)
            action_dist_list.append(action_dist)
        return action_dist_list
    
    def add_exploration(self, action_list: torch.Tensor, itr: int, mode='train'):
        for i, model in enumerate(self.models):
            action_list[i] = model.add_exploration(action_list[i], itr, mode)
        return action_list
    

class DiscreteActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        nl_embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.nl_embedding_size = nl_embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

    def _build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size + self.nl_embedding_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        if self.dist in ['one_hot', 'category']:
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                # nn.init.orthogonal_(m.bias)
        return nn.Sequential(*model) 

    def forward(self, model_state):
        action_dist = self.get_action_dist(model_state)
        action = action_dist.sample()
        action = action + action_dist.log_prob(action) - action_dist.log_prob(action).detach()
        return action+1, action_dist

    def get_action_dist(self, modelstate):
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)
        elif self.dist == 'category':
            return torch.distributions.Categorical(logits=logits)         
        else:
            raise NotImplementedError
            
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                if self.dist == 'one_hot':
                    index = torch.randint(0, self.action_size, (1,), device=action.device)
                    action = torch.zeros_like(action, dtype=torch.int32)
                    action[:, index] = 1
                elif self.dist == 'category':
                    action = torch.randint(0, self.action_size, (action.shape[0],), device=action.device) + 1
            return action

        raise NotImplementedError