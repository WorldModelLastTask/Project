import gymnasium as gym
import numpy as np
from miniwob.action import ActionTypes
from miniwob.reward import get_binary_reward


import time
import os
import gc
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal
import datetime
import logging



def set_seed(seed: int) -> None:
    """
    Pytorch, NumPyのシード値を固定します．これによりモデル学習の再現性を担保できます．

    Parameters
    ----------
    seed : int
        シード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(1234)


import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple
from miniwob.action import ActionTypes, ActionSpaceConfig, ActionTypes, Action
from miniwob.reward import get_binary_reward
from transformers import DebertaTokenizer, AutoTokenizer
import sys

from PaddleOCR.paddleocr import PaddleOCR
import argparse

class OCR(PaddleOCR):
    def __init__(self):
        self.config = argparse.Namespace(use_gpu=True, use_xpu=False, use_npu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, page_num=0, det_algorithm='DB', det_model_dir=None, det_limit_side_len=960, det_limit_type='max', det_box_type='quad', det_db_thresh=0.3, det_db_box_thresh=0.6, det_db_unclip_ratio=1.5, max_batch_size=10, use_dilation=False, det_db_score_mode='fast', det_east_score_thresh=0.8, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_sast_score_thresh=0.5, det_sast_nms_thresh=0.2, det_pse_thresh=0, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, scales=[8, 16, 32], alpha=1.0, beta=1.0, fourier_degree=5, rec_algorithm='SVTR_LCNet', rec_model_dir=None, rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_batch_num=6, max_text_length=25, rec_char_dict_path=None, use_space_char=True, vis_font_path='./doc/fonts/simfang.ttf', drop_score=0.5, e2e_algorithm='PGNet', e2e_model_dir=None, e2e_limit_side_len=768, e2e_limit_type='max', e2e_pgnet_score_thresh=0.5, e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_pgnet_valid_set='totaltext', e2e_pgnet_mode='fast', use_angle_cls=False, cls_model_dir=None, cls_image_shape='3, 48, 192', label_list=['0', '180'], cls_batch_num=6, cls_thresh=0.9, enable_mkldnn=False, cpu_threads=10, use_pdserving=False, warmup=False, sr_model_dir=None, sr_image_shape='3, 32, 128', sr_batch_num=1, draw_img_save_dir='./inference_results', save_crop_res=False, crop_res_save_dir='./output', use_mp=False, total_process_num=1, process_id=0, benchmark=False, save_log_path='./log_output/', show_log=False, use_onnx=False, output='./output', table_max_len=488, table_algorithm='TableAttn', table_model_dir=None, merge_no_span_structure=True, table_char_dict_path=None, layout_model_dir=None, layout_dict_path=None, layout_score_threshold=0.5, layout_nms_threshold=0.5, kie_algorithm='LayoutXLM', ser_model_dir=None, re_model_dir=None, use_visual_backbone=True, ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ocr_order_method=None, mode='structure', image_orientation=False, layout=True, table=True, ocr=True, recovery=False, use_pdf2docx_api=False, invert=False, binarize=False, alphacolor=(255, 255, 255), lang='en', det=True, rec=True, type='ocr', ocr_version='PP-OCRv4', structure_version='PP-StructureV2')
        super().__init__(**(self.config.__dict__))
    def ocr(self, img):
        result = super().ocr(
            img,
            det=self.config.det,
            rec=self.config.rec,
            cls=self.config.use_angle_cls,
            bin=self.config.binarize,
            inv=self.config.invert,
            alpha_color=self.config.alphacolor
            )
        return result
    
from transformers import AutoModel

class LM(nn.Module):
    def __init__(self, is_linear=False, model_name=None):
        super(LM, self).__init__()
        self.lm = AutoModel.from_pretrained(model_name).cuda()
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad = False
        # if is_linear:
        #     self.fc = nn.Linear(384, 512)
        # else:
        #     self.fc = nn.Identity()
    def forward(self, x, attention_mask=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        emb = self.lm(x.cuda(), attention_mask=attention_mask.cuda()).last_hidden_state.cpu().detach()
        last_idx = (attention_mask.sum(dim=1)-1).long().unsqueeze(1).unsqueeze(2).expand(-1, -1, emb.shape[-1])
        emb = emb.gather(1, last_idx).squeeze(1)
        # return self.fc(emb)
        del x, attention_mask, last_idx
        gc.collect(), torch.cuda.empty_cache()
        return emb

def make_env(env_name, seed=None, render_mode=None, conf=None):
    if conf is None:
        conf = ActionSpaceConfig(ActionTypes).get_preset(name="humphreys22")
        conf.coord_bins = None
    env =gym.make("miniwob/"+env_name, render_mode = render_mode,reward_processor=get_binary_reward, action_space_config=conf) 

    # env.seed(seed)
    if seed is not None:
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        np.random.seed(seed)

    return env
class MiniWob_Env:
    def __init__(self, env_name, seed=0, render_mode=None, 
        action_types = [ActionTypes.CLICK_COORDS,], 
        coord_bins=(40,40), max_length=32, clip_reward=True, screen_width=160, screen_height=160, step_limit=10, seed_fix=False):
        
        self.seed = seed
        self.seed_fix = seed_fix
        self.coord_bins = coord_bins
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.env_name = env_name
        self.bins = np.linspace(0, self.screen_width, self.coord_bins[0], dtype=np.int32)
        if seed_fix:
            self.env = make_env(env_name, seed=0, render_mode=render_mode)
        else:
            self.env = make_env(env_name, seed=seed, render_mode=render_mode)
        self.action_types = action_types
        
        ## config
        self.clip_reward = clip_reward
        self.step_limit = step_limit
        self.count_step = 0
        
        ## preprocess
        self.preprocess_img = transforms.Compose([
            # lambda x: (x/255.0).astype(np.uint8),
            transforms.ToTensor(),
            transforms.Resize((screen_width, screen_height),antialias=True),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])
        
        def preprocess_nl(img, nl, tokenizer, ocr, lm, is_embed=False):
            # img = (img.numpy().transpose(1, 2, 0) + 1) / 2 * 255
            ocr_result = ocr.ocr(img)
            prompt = "instruction: " + nl + ", ocr: "
            if ocr_result == [None]:
                prompt += "None"
            else:
                prompt += "(coordinates,content)={"
                for line in ocr_result[0]:
                    prompt += f"({line[0]},{line[1][0]}),"
                prompt += "}"
            result = tokenizer(prompt, padding='max_length', max_length=max_length)
            embed, attention_mask = result["input_ids"],result["attention_mask"]
            del result, prompt, ocr_result
            gc.collect(), torch.cuda.empty_cache()
            if is_embed and lm is not None:
                return lm(torch.tensor(embed), torch.tensor(attention_mask))
            return [torch.tensor(embed), torch.tensor(attention_mask)]
        self.preprocess_nl = preprocess_nl
        
    def reset(self, seed=None, train=True, tokenizer=None, ocr=None, lm=None, is_embed=False, render=False):
        # self.time=time.time()
        if self.seed_fix:
            obs, _ = self.env.reset(seed = 0)
        else:
            if train and seed is None:
                seed = np.random.randint(0, 5)
            elif not train and seed is None:
                seed = np.random.randint(5, 10)
            if seed is not None:
                obs, _ = self.env.reset(seed = seed)
            else:
                obs, _ = self.env.reset()
        img = self.preprocess_img(obs["screenshot"][50:, ])
        nl = self.preprocess_nl(obs["screenshot"][50:, ], obs["utterance"], tokenizer, ocr, lm, is_embed)
        if render:
            return [img, nl], obs["screenshot"][50:, ]
        return [img, nl]
    def step(self, action, tokenizer=None, ocr=None, lm=None, is_embed=False, render=False): #action -> (action type, (x_bin, y_bin)), actor->(1~3, (x_bin, y_bin)
        coords = self.bin2coord((action[1]-1, action[2]-1))
        action = self.env.unwrapped.create_action(self.action_types[int(action[0])-1], coords = coords)
        # print("time",time.time() - self.time)
        # self.time = time.time()
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if self.clip_reward:
            if reward <= 0:
                reward = 0.0
        img = self.preprocess_img(obs["screenshot"][50:, ])
        nl = self.preprocess_nl(obs["screenshot"][50:, ], obs["utterance"], tokenizer, ocr, lm, is_embed) # "insturction: ~, ocr: ~"
        
        self.count_step += 1
        if self.count_step >= self.step_limit or done:
            done = True
            self.count_step = 0
        if truncated and reward <= 0:
            reward = 0
        if render:
            return [img, nl], reward, done, None, obs["screenshot"][50:, ]
        return [img, nl], reward, done, None
    # def bin2coord(self, bin) -> Tuple[float, float]:
    #     """Extract the left and top coordinates from the action."""
    #     assert bin[0] < self.coord_bins[0] and bin[1] < self.coord_bins[1], f"error bin {bin}"
    #     left = self.bins[int(bin[0])]+(self.bins[1]-self.bins[0])/2
    #     top = self.bins[int(bin[1])]+(self.bins[1]-self.bins[0])/2
    #     return left, top
    # def coord2bin(self, coord):#coods -> bins
    #     if not self.screen_width or not self.screen_height:
    #         raise ValueError("screen_width and screen_height must be specified.")
    #     return np.digitize(coord, self.bins)
    def bin2coord(self, bin) -> Tuple[float, float]:
        """Extract the left and top coordinates from the action."""
        if self.coord_bins:
            # Add 0.5 to click at the middle of the partition.

            left = (0.5 + int(bin[0])) * (
                160 / self.coord_bins[0]
            )
            top = (0.5 + int(bin[1])) * (
                160 / self.coord_bins[1]
            ) + 50
            assert left >= 0 and left <= 160, f"error left {left}"
            assert top >= 50 and top <= 210, f"error top {top}"
        else:
            raise ValueError("coord_bins must be specified.")
        return left, top
    @property
    def action_space(self):
        space_shape = (len(self.action_types), *self.coord_bins)
        return gym.spaces.MultiDiscrete(space_shape)
    @property
    def observation_space(self): 
        return gym.spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height , 3),dtype=np.uint8)    
class VecEnvs:
    def __init__(self, env_list, render_mode=None, action_types = [ActionTypes.CLICK_COORDS,],coord_bins=(40,40), max_length=32, screen_width=160, screen_height=160, step_limit=10, seed_fix=False,tokenizer_name=None,ocr_name=None):
        self.env_list = env_list
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.lm = LM(model_name=tokenizer_name)
        self.ocr = OCR()
        self.vec_envs = [MiniWob_Env(env_name, render_mode = render_mode, action_types=action_types, coord_bins=coord_bins, max_length=max_length, screen_width=screen_width, screen_height=screen_height, step_limit=step_limit, seed_fix=seed_fix) for env_name in env_list]
    def reset(self, seed=None):
        obs_list = []
        for env in self.vec_envs:
            obs = env.reset(seed, tokenizer=self.tokenizer, ocr=self.ocr, lm=self.lm)
            obs_list.append(obs)
        img = torch.stack([obs_list[i][0] for i in range(len(obs_list))])
        ln  = torch.stack([obs_list[i][1][0] for i in range(len(obs_list))])
        at  = torch.stack([obs_list[i][1][1] for i in range(len(obs_list))])
        state = [img, self.lm(ln, at)]
        return state
    def step(self, action_list, is_reset=False):
        obs_list = []
        reward_list = []
        done_list = []
        for env, action in zip(self.vec_envs, action_list):
            obs, reward, done, _ = env.step(action, self.tokenizer, self.ocr, self.lm)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            if done and is_reset:
                env.reset()
        img = torch.stack([obs_list[i][0] for i in range(len(obs_list))])
        ln  = torch.stack([obs_list[i][1][0] for i in range(len(obs_list))])
        am = torch.stack([obs_list[i][1][1] for i in range(len(obs_list))])
        
        state = [img, self.lm(ln, am)]
        return state, reward_list, done_list, None
    
    def onehot2action(self, onehot):
        action = []
        for i in range(len(onehot)):
            action.append(torch.argmax(onehot[i],dim=-1))
        return action
    
    def close(self):
        for env in self.vec_envs:
            env.env.close()