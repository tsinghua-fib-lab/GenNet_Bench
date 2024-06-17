import time
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

import torch
from torch.optim import Adam
import torch.nn as nn
from utils.util import init


class NNPerformancePredictor:
    def __init__(self, d_reward, size=100, device='cuda:0'):
        hidden_size = 16
        gain = nn.init.calculate_gain('relu')

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.device = device
        self.nn = nn.Sequential(
            init_(nn.Linear(d_reward, hidden_size)),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            init_(nn.Linear(hidden_size, d_reward))).to(self.device)
        self.optim = Adam(self.nn.parameters(), lr=1e-3)

        self.mini_buffer = {'weights': [], 'evaluations': []}
        self.size = size

    def _predict(self, weight):
        return self.nn(weight).detach()

    def _train(self, weight, evaluation):
        self.optim.zero_grad()
        loss = (0.5 * (self.nn(weight) - evaluation) ** 2.).mean()
        loss.backward()
        self.optim.step()

    def add(self, weight, evaluation):
        self.mini_buffer['weights'].append(weight)
        self.mini_buffer['evaluations'].append(evaluation)
        if len(self.mini_buffer['weights']) > self.size:
            del self.mini_buffer['weights'][0]
            del self.mini_buffer['evaluations'][0]

        for _ in range(100):
            idx = np.random.choice(len(self.mini_buffer['weights']), 16)
            weights, evaluations = (
                torch.Tensor([self.mini_buffer['weights'][i] for i in idx]),
                torch.Tensor([self.mini_buffer['evaluations'][i] for i in idx]))
            self._train(weights.to(self.device), evaluations.to(self.device))

    def predict_next_evaluation(self, weight_candidate: np.ndarray, policy_eval: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        weight_candidate = torch.from_numpy(weight_candidate).float().to(self.device)
        evaluation = self._predict(weight_candidate).cpu().numpy()
        return np.zeros_like(evaluation), evaluation


