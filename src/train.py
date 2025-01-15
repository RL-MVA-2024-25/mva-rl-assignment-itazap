import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

import itertools

MAX_EPISODES = 200

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

main_path = os.path.join(os.getcwd(), 'src/grid_search_models')

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[int(self.index)] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        self.model = None
        self.target_model = None
        self.memory = None
        self.best_model = None
        self.optimizer = None
        self.gamma = None
        self.batch_size = None

    def act(self, observation, use_random=False):
        device = self._get_device()
        if use_random:
            return env.action_space.sample()
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        save_path = os.path.join(main_path, path)
        torch.save(self.model.state_dict(), save_path)

    def load(self):
        filename='best_model_ita.pt'
        load_path = os.path.join(os.getcwd(), filename)
        device = self._get_device()
        self.model = self._build_model({}, device)
        self.model.load_state_dict(torch.load(load_path, map_location=device))
        self.model.eval()

    def train(self, filename):
        config = self._default_config()
        self._initialize_training(config)
        epsilon, step, episode, episode_cum_reward = config["eps_max"], 0, 0, 0
        state, _ = env.reset()
        episode_returns = []

        while episode < MAX_EPISODES:
            epsilon = self._update_epsilon(epsilon, step, config)
            action = self._choose_action(state, epsilon)
            next_state, reward, done, trunc, _ = env.step(action)
            self._store_transition(state, action, reward, next_state, done)
            episode_cum_reward += reward

            for _ in range(config["gradient_steps"]):
                self._gradient_step()

            self._update_target_network(step, config)

            if done or trunc:
                episode = self._finalize_episode(
                    episode, episode_cum_reward, episode_returns,filename
                )
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state

            step += 1

        self.model.load_state_dict(self.best_model.state_dict())
        self.save(path=filename)
        return self.best_score

    def _build_model(self, config, device):
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        hidden_units = 256

        return nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_actions),

        ).to(device)

    def _initialize_training(self, config):
        device = self._get_device()
        self.model = self._build_model(config, device)
        self.target_model = deepcopy(self.model).to(device)
        self.memory = ReplayBuffer(1e5, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _default_config(self):
        return {
            "lr": 0.001,
            "gamma": 0.95,
            "batch_size": 20,
            "eps_min": 0.01,
            "eps_max": 1.0,
            "eps_decay_period": 1000,
            "eps_delay_decay": 20,
            "gradient_steps": 1,
            "update_target_freq": 50,
            "update_target_tau": 0.005,
            "criterion": nn.SmoothL1Loss(),
        }
   
    def _update_epsilon(self, epsilon, step, config):
        if step > config["eps_delay_decay"]:
            epsilon_step = (config["eps_max"] - config["eps_min"]) / config["eps_decay_period"]
            epsilon = max(config["eps_min"], epsilon - epsilon_step)
        return epsilon

    def _choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        return self.act(state)

    def _store_transition(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def _gradient_step(self):
        if len(self.memory) < self.batch_size:
            return
        X, A, R, Y, D = self.memory.sample(self.batch_size)
        QYmax = self.target_model(Y).max(1)[0].detach()
        target = R + (1 - D) * self.gamma * QYmax
        QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = nn.SmoothL1Loss()(QXA, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target_network(self, step, config):
        if step % config["update_target_freq"] == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def _finalize_episode(self, episode, reward, returns, filename):
        episode += 1
        score = evaluate_HIV(agent=self, nb_episode=1)
        print(f"Episode {episode} | Reward {reward:.2e} | Score {score:.2e}")
        if score > getattr(self, "best_score", float("-inf")):
            self.best_model = deepcopy(self.model)
            self.best_score = score
            self.save(path=filename)
        returns.append(reward)
        return episode



def main():
    param_grid = {
        "lr": [1e-3],
        "gamma": [0.97],
        "batch_size": [768],
        "gradient_steps": [4],
        "eps_min": [3e-2],
        "eps_max": [1.0],
        "eps_decay_period": [1e4],
        "eps_delay_decay": [1e2],
        "update_target_freq": [400],
        "update_target_tau": [5e-3],
        "criterion": nn.SmoothL1Loss(),
    }

    param_combinations = list(itertools.product(
        param_grid["lr"],
        param_grid["gamma"],
        param_grid["batch_size"],
        param_grid["gradient_steps"],
        param_grid["eps_min"],
        param_grid["eps_max"],
        param_grid["eps_decay_period"],
        param_grid["eps_delay_decay"],
        param_grid["update_target_freq"],
        param_grid["update_target_tau"],
    ))

    best_return = float("-inf")
    best_params = None

    results_file = "grid_search_results.txt"
    with open(results_file, "a") as file:
        if file.tell() == 0:
            file.write("lr,gamma,batch_size,gradient_steps,eps_min,eps_max,eps_decay_period,eps_delay_decay,update_target_freq,update_target_tau,best_score\n")

        for params in param_combinations:
            (lr, gamma, batch_size, gradient_steps, eps_min, eps_max,
             eps_decay_period, eps_delay_decay, update_target_freq, update_target_tau) = params
            
            # print param combinations to console
            print(f"TRAINING: lr: {lr}, gamma: {gamma}, batch_size: {batch_size}, gradient_steps: {gradient_steps}, eps_min: {eps_min}, eps_max: {eps_max}, eps_decay_period: {eps_decay_period}, eps_delay_decay: {eps_delay_decay}, update_target_freq: {update_target_freq}, update_target_tau: {update_target_tau}")

            agent = ProjectAgent()
            config = agent._default_config()
            config.update({
                "lr": lr,
                "gamma": gamma,
                "batch_size": batch_size,
                "gradient_steps": gradient_steps,
                "eps_min": eps_min,
                "eps_max": eps_max,
                "eps_decay_period": eps_decay_period,
                "eps_delay_decay": eps_delay_decay,
                "update_target_freq": update_target_freq,
                "update_target_tau": update_target_tau,
            })

            agent._initialize_training(config)

            model_filename = (f"model_lr{lr}_g{gamma}_bs{batch_size}_gs{gradient_steps}_"
                              f"epsmin{eps_min}_epsmax{eps_max}_"
                              f"epsdecay{eps_decay_period}_epsdelay{eps_delay_decay}_"
                              f"utfreq{update_target_freq}_uttau{update_target_tau}.pt")

            best_score = agent.train(filename=model_filename)

            file.write(f"{lr},{gamma},{batch_size},{gradient_steps},{eps_min},"
                       f"{eps_max},{eps_decay_period},{eps_delay_decay},{update_target_freq},"
                       f"{update_target_tau},{best_score}\n")

            if best_score > best_return:
                best_return = best_score
                best_params = params

    print(f"Best Parameters: {best_params}")
    print(f"Best Return: {best_return}")


if __name__ == "__main__":
    main()
