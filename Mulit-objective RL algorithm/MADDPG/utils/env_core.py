import math
import random
import numpy as np
import json


from .Unet import UNet
import torch


class EnvCore(object):
    

    def __init__(self):
        
        self.time = 25000
        self.interval = 1
        self.w = [0.3, 0.3, 0.3]

        self.aiz = 3  
        self.tilt = 3  
        self.power = 3  
        self.agent_num = 39  
        self.row = 177
        self.column = 156
        self.grid_num = self.row * self.column  
        
        self.obs_dim = 71  
        self.buffer_obs_dim = (3 + 4 + self.grid_num * 6)  
        self.action_dim = self.aiz * self.tilt * self.power  

        self.cover_his = []
        self.throughput_his = []
        self.power_his = []
        self.reward_his = []

        with open('utils/out_bs_info.json', 'r') as file:
            js = file.read()
            bs = json.loads(js)
        self.bs = bs  

        with open('utils/grid_info.json', 'r') as file:
            js = file.read()
            grid = json.loads(js)
        self.grid = grid  

        with open('utils/out_is_grid_bs.json', 'r') as file:
            js = file.read()
            is_grid_bs = json.loads(js)
        self.is_grid_bs = np.array(is_grid_bs["is_grid_bs"])  

        with open('utils/is_grid_indoor.json', 'r') as file:
            js = file.read()
            is_grid_indoor = json.loads(js)
        self.is_indoor = np.array(is_grid_indoor["is_indoor"])  

        self.people_1 = np.zeros(self.grid_num)  
        self.people_2 = np.zeros(self.grid_num)  
        self.people_3 = np.zeros(self.grid_num)  

        self.n = math.pow(10, (-174 / 10 + math.log10(1.8e5)))  

        self.Unet = UNet(6, 64)
        self.Unet.load_state_dict(
            torch.load("utils/Unet.pt", map_location=torch.device('cpu')))

    def reset(self):

        sub_agent_obs = []
        s3 = np.zeros(64)  
        for i in self.bs:
            self.bs[i]['angle'] = [random.choice(range(0, 361, 10)), random.choice(range(0, 91, 5))]
        for i in self.bs:
            s1 = np.append(self.bs[i]["longlat_position"], self.bs[i]["transmit_power"])
            s2 = np.concatenate((self.bs[i]['angle'], np.array([63, 6.5])))
            s = np.concatenate((s1, s2, s3))
            sub_agent_obs.append(s)
        return sub_agent_obs

    def step(self, actions):

        actions = np.array(actions)
        coverage = np.zeros(self.grid_num)
        sig_store = np.zeros(self.grid_num)
        rate = np.zeros(self.grid_num)

        self.people_3 = self.people_2
        self.people_2 = self.people_1
        self.people_1 = np.zeros(self.grid_num)

        num_bs = 0
        for j in self.bs:  
            act = np.argmax(actions[num_bs])
            bs_azi = (act // (self.tilt * self.power))
            bs_tilt = ((act % (self.power * self.tilt)) // self.power)
            power = act % self.power * 15
            
            angle_adjustments = {0: -10, 1: 0, 2: 10}
            tilt_adjustments = {0: -5, 1: 0, 2: 5}

            new_azi = self.bs[j]['angle'][0] + angle_adjustments[bs_azi]
            new_tilt = self.bs[j]['angle'][1] + tilt_adjustments[bs_tilt]

            if 0 <= new_azi <= 360:
                self.bs[j]['angle'][0] = new_azi

            if 0 <= new_tilt <= 90:
                self.bs[j]['angle'][1] = new_tilt

            self.bs[j]['power'] = power
            num_bs += 1

        action = []
        for i in self.bs:
            d = {}
            d['id'] = i
            d['bs_azi'] = self.bs[i]['angle'][0]
            d['bs_tilt'] = self.bs[i]['angle'][1]
            d['bw_azi'] = 63
            d['bw_tilt'] = 6.5
            d['power'] = self.bs[i]['power']
            
            action.append(d)

        cover = (np.sum(sig_store > -120)) / self.grid_num
        coverage = sig_store > -120  

        
        rate_average = sum(rate) / len(rate)
        rate_average = rate_average / 7e4  

        
        energy = 0
        for j in self.bs:  
            energy += self.bs[j]['power']
        energy = (1 - (energy / (self.agent_num * 30))) 

        
        reward = cover * self.w[0] + rate_average * self.w[1] + energy * self.w[2]
        print("覆盖率:{},吞吐量：{},能耗：{},奖励：{}".format(cover, rate_average, energy, reward))

        
        self.cover_his.append(cover)
        self.throughput_his.append(rate_average)
        self.power_his.append(energy)
        self.reward_his.append(reward)
        json.dump({'cover_his': self.cover_his}, fp=open('./cover_his' + '.json', 'w'))
        json.dump({'throughput_his': self.throughput_his}, fp=open('./throughput_his' + '.json', 'w'))
        json.dump({'power_his': self.power_his}, fp=open('./power_his' + '.json', 'w'))
        json.dump({'reward_his': self.reward_his}, fp=open('./reward_his' + '.json', 'w'))

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []  
        sub_agent_info = []  

        s3 = np.concatenate((coverage, self.is_indoor, self.is_grid_bs, self.people_1/np.max(self.people_1), self.people_2, self.people_3))

        unet_obs = s3.reshape((1, 6, self.row, self.column))
        unet_result = self.Unet(torch.from_numpy(unet_obs).float())
        unet_result = unet_result.detach().cpu().numpy()

        
        bs_info = np.zeros(64)
        for i, info in enumerate(self.bs):
            row = int(self.bs[info]["poi_grid"] / self.column)
            col = self.bs[info]["poi_grid"] % self.column
            bs_info = np.vstack((bs_info, unet_result[0, :, row, col]))

        bs_info = np.delete(bs_info, 0, axis=0)

        num_bs = 0
        for i in self.bs:
            s1 = np.append(self.bs[i]["longlat_position"], self.bs[i]["power"])
            s2 = np.concatenate((self.bs[i]['angle'], np.array([63, 6.5])))
            
            s = np.concatenate((s1, s2, bs_info[num_bs]))
            sub_agent_obs.append(s)
            sub_agent_reward.append(reward)
            sub_agent_done.append(False)
            sub_agent_info.append({})
            num_bs = num_bs + 1
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
