import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from dreamerv2.models.actor import DiscreteActionModel, MultiDiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from transformers import AutoModel
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


class LM(nn.Module):
    def __init__(self, state_dim=32, is_linear=False):
        super(LM, self).__init__()
        self.lm = AutoModel.from_pretrained("microsoft/deberta-base")
        if is_linear:
            self.fc = nn.Linear(state_dim*768, 512)
        else:
            self.fc = nn.Identity()
    def forward(self, x, attention_mask=None):
        emb = self.lm(x, attention_mask=attention_mask).last_hidden_state
        emb = emb.reshape(emb.shape[0], -1)
        return self.fc(emb)
class Evaluator(object):
    '''
    used this only for minigrid envs
    '''
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
        self.lm = LM(state_dim=config.max_length, is_linear=False).eval()

    def load_model(self, config, model_path):
        saved_dict = torch.load(model_path)
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

        if config.pixel:
                self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
                self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()

        if isinstance(config.actor, tuple):
            self.ActionModel = MultiDiscreteActionModel(act_shape, deter_size, stoch_size, nl_embedding_size, config.actor, config.expl).to(self.device)
        else:
            self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device)
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.config.max_length, self.device, config.rssm_type, config.rssm_info).to(self.device).eval()

        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])

    def eval_saved_agent(self, env, model_path):
        self.load_model(self.config, model_path)
        eval_episode = self.config.eval_episode
        if f"{env.__class__.__name__}" == "VecEnvs":
            mean_scores = []
            eval_scores = [[] for _ in range(len(self.config.env_list))]
            for i_env in range(len(self.config.env_list)):
                n_episode = 0
                prev_rssmstate = self.RSSM._init_rssm_state(1)
                prev_action = torch.zeros((1, self.config.action_size)).to(self.device)
                while n_episode < eval_episode:
                    obs, done = env.vec_envs[i_env].reset(tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True), False
                    embed = self.ObsEncoder(obs[0].to(self.device)).unsqueeze(0)
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, 1-done*1, prev_rssmstate)
                    model_state = self.RSSM.get_model_state(posterior_rssm_state)
                    action, _ = self.ActionModel(torch.cat([model_state, obs[1].to(self.device)], dim=-1))
                    action = torch.stack(action, dim=1).detach()
                    next_obs, reward, done, _ = env.vec_envs[i_env].step(action.squeeze(0),tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True)
                    if done:
                        n_episode += 1
                        prev_rssmstate = self.RSSM._init_rssm_state(1)
                        prev_action = torch.zeros((1, self.config.action_size)).to(self.device)
                        eval_scores[i_env].append(reward)
                        obs, done = env.vec_envs[i_env].reset(tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True), False
                    else:
                        obs = next_obs
                        prev_rssmstate = posterior_rssm_state
                        prev_action = action # init action is 0
                mean_scores.append(np.sum(eval_scores[i_env]))
                self.writer.add_scalar(f'eval_score/{self.config.env_list[i_env]}', np.sum(eval_scores[i_env]))
            self.writer.add_scalar('eval_score/total_mean', np.mean(mean_scores))
            print('average evaluation score for model at ' + model_path + ' = ' +str(np.mean(mean_scores)))
            env.close()
            return np.mean(np.mean(mean_scores))
    def eval_imagine(self, env, model_path, is_ablation=False):
        self.load_model(self.config, model_path)
        eval_episode = self.config.eval_episode
        if f"{env.__class__.__name__}" == "VecEnvs":
            mean_scores = []
            eval_scores = [[] for _ in range(len(self.config.env_list))]
            for i_env in range(len(self.config.env_list)):
                n_episode = 0
                prev_rssmstate = self.RSSM._init_rssm_state(1)
                prev_action = torch.zeros((1, self.config.action_size)).to(self.device)
                raw_frames, prep_frames,imagine_frames = [], [],[]
                done = False
                # while n_episode < eval_episode:
                with torch.no_grad():
                    while True:
                        obs, raw_img = env.vec_envs[i_env].reset(tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True, render=True)
                        embed = self.ObsEncoder(obs[0].to(self.device)).unsqueeze(0)
                        _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, 1-done*1, prev_rssmstate)
                        model_state = self.RSSM.get_model_state(posterior_rssm_state)
                        if is_ablation:
                            # obs[1] = torch.randn_like(obs[1])
                            obs[1] = torch.zeros_like(obs[1])
                        action, _ = self.ActionModel(torch.cat([model_state, obs[1].to(self.device)], dim=-1))
                        action = torch.stack(action, dim=1).detach()
                        next_obs, reward, done, _, next_raw_img = env.vec_envs[i_env].step(action.squeeze(0),tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True, render=True)
                        if done:
                            n_episode += 1
                            prev_rssmstate = self.RSSM._init_rssm_state(1)
                            prev_action = torch.zeros((1, self.config.action_size)).to(self.device)
                            eval_scores[i_env].append(reward)
                            obs, done = env.vec_envs[i_env].reset(tokenizer=env.tokenizer, ocr=env.ocr, lm=env.lm, is_embed=True), False
                            print("##"*10,self.config.env_list[i_env],"##"*10)
                            if raw_frames == []:
                                print("no frames, score:", reward)
                            else:
                                print("score:", reward)
                                self.display_video(raw_frames)
                                self.display_video(prep_frames)
                                self.display_video(imagine_frames)
                            break
                        else:
                            denorm = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                                            std=[1/0.5, 1/0.5, 1/0.5]),
                                                        transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                                            std=[1., 1., 1.]),
                                                        ])
                            recon_obs = denorm(self.ObsDecoder(model_state).sample()).cpu().numpy().squeeze(0).transpose(1,2,0)
                            obs_0 = denorm(obs[0].cpu()).numpy().transpose(1,2,0)
                            print(raw_img.min(), raw_img.max())
                            print(recon_obs.min(), recon_obs.max())
                            print(obs_0.min(), obs_0.max())
                            raw_frames.append(raw_img)
                            prep_frames.append(obs_0)
                            imagine_frames.append(recon_obs)
                            obs = next_obs
                            raw_img = next_raw_img
                            prev_rssmstate = posterior_rssm_state
                            prev_action = action
            env.close()

    def display_video(self, frames):
        plt.figure(figsize=(8, 8), dpi=50)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])
            plt.title("Step %d" % (i))

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        display(HTML(anim.to_jshtml(default_mode='once')))
        plt.close()   
