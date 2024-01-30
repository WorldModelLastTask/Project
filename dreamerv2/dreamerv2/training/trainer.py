import numpy as np
import torch 
import torch.optim as optim
import torch.nn.functional as F
import os, gc, time
from tqdm import tqdm

from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.utils.algorithm import compute_return

from dreamerv2.models.actor import DiscreteActionModel, MultiDiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM, RSSMDiscState, RSSMContState
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.buffer import TransitionBuffer, MultiEnvTransitionBuffer

class Trainer(object):
    def __init__(
        self, 
        config,
        device,
        writer
    ):
        self.device = device
        self.config = config
        self.writer = writer
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip
        self.obs_loss_scale = -np.inf

        self._model_initialize(config)
        self._optim_initialize(config)

    def collect_seed_episodes(self, env):
        s, done  = env.reset(), False
        scores = np.zeros((len(self.config.env_list),))
        dones= np.zeros((len(self.config.env_list),))
        for i in tqdm(range(self.seed_steps), position=0, leave=True, desc='collect_seed_episodes'):
            if f"{env.__class__.__name__}" == "VecEnvs":
                a = [env.vec_envs[i_env].action_space.sample()+1 for i_env in range(len(env.vec_envs))]
                ns, r, done, _ = env.step(a)
                self.buffer.add(s,a,r,done)
                scores += np.array(r, dtype=np.int16)
                for i_env in range(len(env.vec_envs)):
                    if done[i_env]:
                        dones[i_env] += 1
                        _s = env.vec_envs[i_env].reset(tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True)
                        s[0][i_env], s[1][i_env], done[i_env]  = _s[0], _s[1], False 
                    else:
                        s[0][i_env] = ns[0][i_env]
                        s[1][i_env] = ns[1][i_env]
            else:
                a = env.action_space.sample()
                ns, r, done, _ = env.step(a)
                if done:
                    self.buffer.add(s,a,r,done)
                    s, done  = env.reset(), False 
                else:
                    self.buffer.add(s,a,r,done)
                    s = ns    
        for i_env in range(len(self.config.env_list)):
            self.writer.add_scalar(f'seed_score/{self.config.env_list[i_env]}', scores[i_env]/dones[i_env])
        env.close()

    def train_batch(self, train_metrics, iter):
        """ 
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []
        scaler = torch.cuda.amp.GradScaler()
        for _ in tqdm(range(self.collect_intervals), position=0, leave=False, desc='train_batch'):
        
            obs, actions, rewards, terms = self.buffer.sample()

            obs_img = torch.tensor(obs[0], dtype=torch.float32) 
            obs_nl = torch.tensor(obs[1], dtype=torch.float32) #t, t+seq_len 
            actions = torch.tensor(actions, dtype=torch.float32)                 #t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)   #t-1 to t+seq_len-1
            nonterms = torch.tensor(1-terms, dtype=torch.float32).unsqueeze(-1)  #t-1 to t+seq_len-1
            model_loss_list = torch.zeros(self.config.accumulated_loss)
            kl_loss_list = torch.zeros(self.config.accumulated_loss)
            obs_loss_list = torch.zeros(self.config.accumulated_loss)
            reward_loss_list = torch.zeros(self.config.accumulated_loss)
            pcont_loss_list = torch.zeros(self.config.accumulated_loss)
            prior_ent_list = torch.zeros(self.config.accumulated_loss)
            post_ent_list = torch.zeros(self.config.accumulated_loss)
            posterior_logit_list = []
            posterior_stoch_list = []
            posterior_deter_list = []
            sum_b = 0
            for i_acc in range(self.config.accumulated_loss):
                batch_size = ((self.batch_size*len(self.config.env_list))//self.config.accumulated_loss)
                o_i = obs_img[:,i_acc*batch_size:(i_acc+1)*batch_size].to(self.device)
                o_n = obs_nl[:,i_acc*batch_size:(i_acc+1)*batch_size].to(self.device)
                a = actions[:,i_acc*batch_size:(i_acc+1)*batch_size].to(self.device)
                r = rewards[:,i_acc*batch_size:(i_acc+1)*batch_size].to(self.device)
                nt = nonterms[:,i_acc*batch_size:(i_acc+1)*batch_size].to(self.device)
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss([o_i, o_n], a, r, nt, batch_size=batch_size)
                del o_i, o_n, a, r, nt
                gc.collect, torch.cuda.empty_cache()
                model_loss_list[i_acc] = model_loss.cpu()
                kl_loss_list[i_acc] = kl_loss.cpu().detach()
                obs_loss_list[i_acc] = obs_loss.cpu().detach()
                reward_loss_list[i_acc] = reward_loss.cpu().detach()
                pcont_loss_list[i_acc] = pcont_loss.cpu().detach()
                with torch.no_grad():
                    prior_ent_list[i_acc] = prior_dist.entropy().mean().cpu()
                    post_ent_list[i_acc] = post_dist.entropy().mean().cpu()
                    
                posterior_logit_list.append(posterior.logit.cpu())
                posterior_stoch_list.append(posterior.stoch.cpu())
                posterior_deter_list.append(posterior.deter.cpu())
                gc.collect, torch.cuda.empty_cache()
            model_loss = model_loss_list.mean().to(self.device)
            kl_loss = kl_loss_list.mean()
            obs_loss = obs_loss_list.mean()
            reward_loss = reward_loss_list.mean()
            pcont_loss = pcont_loss_list.mean()
            prior_ent = prior_ent_list.mean()
            post_ent = post_ent_list.mean()
            posterior_logit = torch.cat(posterior_logit_list, dim=1).to(self.device).float()
            posterior_stoch = torch.cat(posterior_stoch_list, dim=1).to(self.device).float()
            posterior_deter = torch.cat(posterior_deter_list, dim=1).to(self.device).float()
            posterior = RSSMDiscState(posterior_logit, posterior_stoch, posterior_deter)
            del model_loss_list, kl_loss_list, obs_loss_list, reward_loss_list, pcont_loss_list, prior_ent_list, post_ent_list, posterior_logit_list, posterior_stoch_list, posterior_deter_list
            gc.collect, torch.cuda.empty_cache()
            
            self.model_optimizer.zero_grad()

            # model_loss.backward()
            scaler.scale(model_loss).backward()
            scaler.unscale_(self.model_optimizer)
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
            # self.model_optimizer.step()
            scaler.step(self.model_optimizer)
            scaler.update()
            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                actor_loss, value_loss, target_info = self.actorcritc_loss(posterior, obs_nl.to(self.device))
            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_list), self.grad_clip_norm)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_list), self.grad_clip_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])
            
            del model_loss, value_loss, target_info, prior_dist, post_dist, posterior, prior_ent, post_ent, actor_loss, obs_loss, reward_loss, pcont_loss, kl_loss
            gc.collect, torch.cuda.empty_cache()

        # train_metrics['model_loss'] = np.mean(model_l)
        # train_metrics['kl_loss']=np.mean(kl_l)
        # train_metrics['reward_loss']=np.mean(reward_l)
        # train_metrics['obs_loss']=np.mean(obs_l)
        # train_metrics['value_loss']=np.mean(value_l)
        # train_metrics['actor_loss']=np.mean(actor_l)
        # train_metrics['prior_entropy']=np.mean(prior_ent_l)
        # train_metrics['posterior_entropy']=np.mean(post_ent_l)
        # train_metrics['pcont_loss']=np.mean(pcont_l)
        # train_metrics['mean_targ']=np.mean(mean_targ)
        # train_metrics['min_targ']=np.mean(min_targ)
        # train_metrics['max_targ']=np.mean(max_targ)
        # train_metrics['std_targ']=np.mean(std_targ)
        
        self.writer.add_scalar('train/model_loss', np.mean(model_l), iter)
        self.writer.add_scalar('train/kl_loss', np.mean(kl_l), iter)
        self.writer.add_scalar('train/reward_loss', np.mean(reward_l), iter)
        self.writer.add_scalar('train/obs_loss', np.mean(obs_l), iter)
        self.writer.add_scalar('train/value_loss', np.mean(value_l), iter)
        self.writer.add_scalar('train/actor_loss', np.mean(actor_l), iter)
        self.writer.add_scalar('train/prior_entropy', np.mean(prior_ent_l), iter)
        self.writer.add_scalar('train/posterior_entropy', np.mean(post_ent_l), iter)
        self.writer.add_scalar('train/pcont_loss', np.mean(pcont_l), iter)
        self.writer.add_scalar('train/mean_targ', np.mean(mean_targ), iter)
        self.writer.add_scalar('train/min_targ', np.mean(min_targ), iter)
        self.writer.add_scalar('train/max_targ', np.mean(max_targ), iter)
        
        return train_metrics

    def actorcritc_loss(self, posterior, embed_nl):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior, embed_nl)
        
        # imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        imag_modelstates = torch.cat([self.RSSM.get_model_state(imag_rssm_states), embed_nl[:-1].flatten(0,1).repeat(self.horizon,1,1)], dim=-1)
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]+[self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)
        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)     

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
        }

        return actor_loss, value_loss, target_info

    def representation_loss(self, obs, actions, rewards, nonterms, batch_size):
        embed = self.ObsEncoder(obs[0])                                         #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(batch_size)   
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len     
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        
        post_modelstate = torch.cat([post_modelstate, obs[1]], dim=-1) 
        reward_dist = self.RewardDecoder(post_modelstate[:-1])               #t to t+seq_len-1  
        pcont_dist = self.DiscountModel(post_modelstate[:-1])                #t to t+seq_len-1   
        
        obs_loss = self._obs_loss(obs_dist, obs[0][:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount']*pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)
        
        if self.config.actor_grad == 'reinforce':
            advantage = (lambda_returns-imag_value[:-1]).detach()
            objective = imag_log_prob[1:] * advantage

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:]
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates) 
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss
            
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
        # if self.obs_loss_scale==-np.inf:
        #     self.obs_loss_scale = obs_loss.detach().cpu().item()
        # return obs_loss/self.obs_loss_scale
        # return F.mse_loss(obs_dist.sample(), obs)
        # return F.mse_loss(obs_dist, obs)
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_latest.pth')
        save_dict["model_optimizer"] = self.model_optimizer.state_dict()
        save_dict["actor_optimizer"] = self.actor_optimizer.state_dict()
        save_dict["value_optimizer"] = self.value_optimizer.state_dict()
        save_dict["iter"] = iter
        torch.save(save_dict, save_path)
        del save_dict
        gc.collect, torch.cuda.empty_cache()

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])
        self._optim_initialize(self.config)
    def restart_from_checkpoint(self, checkpoint_path):
        saved_dict = torch.load(checkpoint_path)
        self.load_save_dict(saved_dict)
        self.model_optimizer.load_state_dict(saved_dict["model_optimizer"])
        self.actor_optimizer.load_state_dict(saved_dict["actor_optimizer"])
        self.value_optimizer.load_state_dict(saved_dict["value_optimizer"])
        return saved_dict["iter"]
    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        act_shape = config.act_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size
        nl_embedding_size = config.nl_embedding_size 
        if len(config.env_list) > 1:
            self.buffer = MultiEnvTransitionBuffer(len(config.env_list),config.capacity, obs_shape, action_size, config.seq_len, config.batch_size, config.obs_dtype, config.action_dtype, config.nl_embedding_size)
        else:
            self.buffer = TransitionBuffer(config.capacity, obs_shape, action_size, config.seq_len, config.batch_size, config.obs_dtype, config.action_dtype)
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.config.max_length, self.device, config.rssm_type, config.rssm_info).to(self.device)
        if isinstance(config.actor, tuple):
            self.ActionModel = MultiDiscreteActionModel(act_shape, deter_size, stoch_size, nl_embedding_size, config.actor, config.expl).to(self.device)
        else:
            self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, nl_embedding_size, config.actor, config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size + nl_embedding_size, config.reward).to(self.device)
        self.ValueModel = DenseModel((1,), modelstate_size + nl_embedding_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), modelstate_size + nl_embedding_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        
        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size + nl_embedding_size, config.discount).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
        print('\n Actor: \n', self.ActionModel)
        print('\n Critic: \n', self.ValueModel)