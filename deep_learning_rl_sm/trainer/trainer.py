import random
import numpy as np
import torch
import wandb
from torch.optim import Optimizer
import torch.nn as nn
import time
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model : nn.Module, optimizer: Optimizer, scheduler, batch_size: int = 32,
                 learning_rate: float = 1e-3, num_epochs: int = 10, device=None,dataset=None,data_loader = None,parsed_args=None, acc_grad = 1):
        self.tau = parsed_args["tau"]
        self.grad_norm = parsed_args["grad_norm"]
        self.model = model.to(device)
        self.dataset = dataset
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_start = time.time()
        # Define DataLoader
        """if dataset is not None:
            self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)"""
        self.acc_grad = acc_grad

        # Define optimizer and loss function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_tmp],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    def train_step(
            self,
            timesteps,
            states,
            actions,
            returns_to_go,
            rewards,
            traj_mask
    ):
        self.model.train()
        # data to gpu ------------------------------------------------
        timesteps = timesteps.to(self.device)  # B x T
        states = states[:, 1:, :].float()
        states = states.to(self.device)  # B x T x state_dim
        actions = actions.float()
        actions = actions.to(self.device)  # B x T x act_dim
        returns_to_go = returns_to_go.to(self.device).unsqueeze(
            dim=-1
        )  # B x T x 1
        returns_to_go = returns_to_go.float()
        min_timestep = timesteps.min()
        max_timestep = timesteps.max()
        rewards = rewards.to(self.device).unsqueeze(
            dim=-1
        )  # B x T x 1
        traj_mask = traj_mask.to(self.device)  # B x T

        # model forward ----------------------------------------------
        (
            returns_to_go_preds,
            actions_dist_preds,
            _,
        ) = self.model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
        )

        returns_to_go_target = torch.clone(returns_to_go).view(
            -1, 1
        )[
            (traj_mask.view(-1, ) == 0) | (traj_mask.view(-1, ) == 1)
            ]
        returns_to_go_preds = returns_to_go_preds.view(-1, 1)[
            (traj_mask.view(-1, ) == 0) | (traj_mask.view(-1, ) == 1)
            ]

        # returns_to_go_loss -----------------------------------------
        norm = returns_to_go_target.abs().mean()
        u = (returns_to_go_target - returns_to_go_preds) / norm
        returns_to_go_loss = torch.mean(
            torch.abs(
                self.tau - (u < 0).float()
            ) * u ** 2
        )

        # TODO this section will likely throw errors when dealing with non discrete envs like in D4RL.
        #  Taking a look at previous implementation or original Reinformer code will help you
        #  to make adjustments to D4RL Rasmus!!!
        #  (we have a categorical dist output from the actor net, chosen by setting arg in parseargs to discrete)
        # action_loss ------------------------------------------------
        # cannot calculate logprobs of out of dist actions need to fill with viable value for padding and then mask to 0
        actions_target = torch.clone(actions)
        mask = actions_target != -100  # padding for actions is -100 (better padding scheme probably necessary)
        actions_target_4logprob = actions_target.clone()
        actions_target_4logprob[~mask] = 0
        actions_target_4logprob = actions_target_4logprob.squeeze()

        log_likelihood = (actions_dist_preds.log_prob(
            actions_target_4logprob
        )).unsqueeze(-1) * mask
        # TODO why is batch dim sometimes 232
        print(log_likelihood.shape)
        log_likelihood = log_likelihood.sum(axis=2).view(-1)[
            (traj_mask.view(-1) == 0) | (traj_mask.view(-1) == 1)
            ].mean()

        entropy = actions_dist_preds.entropy().unsqueeze(-1).sum(axis=2).mean()
        action_loss = -(log_likelihood + self.model.temp().detach() * entropy)

        loss = returns_to_go_loss + action_loss
        print("Loss: " + str(loss))
        # optimization -----------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_norm
        )
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
                self.model.temp() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        self.scheduler.step()

        return loss.detach().cpu().item(), \
                returns_to_go_loss.detach().cpu().item(), \
                action_loss.detach().cpu().item(), \
                torch.mean(u.detach().cpu().item()), \
                log_likelihood.detach().cpu().item(), \
                temperature_loss.detach().cpu().item()

    def train(self, parsed_args):
        self.model.train()
        seed = parsed_args["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = parsed_args["env"]
        dataset = self.dataset  # parsed_args["dataset"]
        # parsed_args["batch_size"] = 16 if dataset == "complete" else 256
        if env in ["kitchen", "maze2d", "antmaze"]:
            parsed_args["num_eval_ep"] = 100
        # TODO set data path
        # dataset_path = os.path.join(variant["dataset_dir"], f"{d4rl_env}.pkl")
        device = torch.device(parsed_args["device"])

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=parsed_args["batch_size"],
            shuffle=True,

        )
        # TODO iterate data ret batch size 128 or [104 (rarely)] make so only 128!(shouldn't massively effect training)
        iterate_data = iter(data_loader)
        # data_test = next(iterate_data)
        """for i, tens in enumerate(data_test):
            print("data batch shape:")
            print(next(iterate_data)[i].shape)
        print()"""
        # TODO implement get_state_stats for our envs???
        # state_mean, state_std = dataset.get_state_stats()

        # TODO initialize env random seed

        # state_dim = env.observation_space.shape[0]
        # act_dim = env.action_space.shape[0]

        model_type = parsed_args["model_type"]

        max_train_iters = parsed_args["max_train_iters"]
        num_updates_per_iter = parsed_args["num_updates_per_iter"]
        score_list_normalized = []

        num_updates = 0
        for _ in range(1, max_train_iters + 1):
            for epoch in range(num_updates_per_iter):
                print(epoch)
                try:
                    print("Trying to get batch")
                    (
                        states,
                        actions,
                        rewards,
                        traj_mask,
                        timesteps,
                        action_masks,
                        returns_to_go

                    ) = next(iterate_data)
                except StopIteration:
                    iterate_data = iter(data_loader)  # start again with original load
                    (
                        states,
                        actions,
                        rewards,
                        traj_mask,
                        timesteps,
                        action_masks,
                        returns_to_go

                    ) = next(iterate_data)

                loss, returns_to_go_loss, action_loss, u, log_likelihood, temperature_loss = self.train_step(
                    timesteps=timesteps.squeeze(2),
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go.squeeze(2),
                    rewards=rewards,
                    traj_mask=traj_mask
                )
                if parsed_args["use_wandb"]:
                    wandb.log(
                        data={
                            "training/loss": loss,
                            "training/rtg_loss": returns_to_go_loss,
                            "training/action_loss": action_loss,
                            "training/u": u,
                            "training/log_likelihood": log_likelihood,
                            "training/temperature_loss": temperature_loss
                        }, 
                        step=num_updates
                    )

                if num_updates % 50 == 0 and num_updates != 0:
                    win_loss_ratio_test = self.evaluate_online(env=env)
                    print("win loss ratio: " + str(win_loss_ratio_test))
                    if parsed_args["use_wandb"]:
                        wandb.log(
                            data={
                                "evaluation/win_loss_ratio": win_loss_ratio_test
                            },
                            step=num_updates
                        )
                num_updates += 1

            # TODO win_loss_ratio_test instead of normalized score or both?
            normalized_score = self.evaluate(
                dataset=None
            )
            score_list_normalized.append(normalized_score)
            if parsed_args["use_wandb"]:
                wandb.log(
                    data={
                        "evaluation/score": normalized_score
                    }, 
                    step=num_updates
                )

        if parsed_args["use_wandb"]:
            wandb.log(
                data={
                    "evaluation/max_score": max(score_list_normalized),
                    "evaluation/last_score": score_list_normalized[-1]
                }, 
                step=num_updates
            )
            wandb.finish()
        print(score_list_normalized)
        print("finished training!")

    def evaluate_online(self, env):
        ratio = 0
        for _ in range(50):
            # states vector is 1 timestep larger because of init state
            # flatten board state
            s0 = torch.flatten(torch.tensor(env.reset()[0]))
            print("s0 shape: " + str(s0.shape[0]))
            states = torch.zeros((1, env.max_two_p_game_length + 1, s0.shape[0]))
            states[0, 0] = s0  # (B=1, T=1, State)
            timesteps = torch.tensor([i for i in range(21)]) + 1  # remember 0 is used for padding
            actions = torch.full((21, 1), -100)
            returns_to_go = torch.full((21, 1), -100)
            while True:

                (
                    returns_to_go_preds,
                    actions_dist_preds,
                    _,
                ) = self.model.forward(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go,
                )
                print(actions_dist_preds.mean)
                mean_action = actions_dist_preds.mean
                state, reward, done, _ = env.step(mean_action)
                ratio += reward if reward == 1 else 0
                if done:
                    break
        ratio /= 50

        return ratio

    def evaluate(self, dataset):
        """Evaluate the model on a given dataset."""
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_states, batch_actions, batch_rewards in data_loader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)

                action_logits = self.model(batch_states)

                # Flatten tensors for computing loss
                action_logits = action_logits.view(-1, action_logits.size(-1))
                batch_actions = batch_actions.view(-1)

                # Compute loss
                loss = self.criterion(action_logits, batch_actions)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_iteration_benchmark(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # training -----------------------------------------------
        self.model.train()
        for i in range(num_steps):
            if i%50 == 0:
                print(f"Iteration {i}, time: {time.time() - train_start}")
            train_loss = self.train_step_benchmark((i+1) % self.acc_grad == 0)
            #wandb.log({"train_loss":np.mean(train_loss)})
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()


        eval_start = time.time()

        # evaluation -----------------------------------------------
        """self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start"""
        
        
        # timing -----------------------------------------------
        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.time_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        
        wandb.log({"train_loss_mean":np.mean(train_losses)})

        
        # diagnostics -----------------------------------------------
        """for k in self.diagnostics:
            logs[k] = self.diagnostics[k]"""


        # prints ---------------------------------------------------
        if print_logs:
            print('=' * 80)
            print(f'Epoch {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs
    
    def get_next(self):
        try:
                (
                    timesteps,
                    states,
                    actions,
                    returns_to_go,
                    traj_mask,
                ) = next(self.data_iter)
        except StopIteration:
                del self.data_iter
                self.data_iter = iter(self.data_loader)
                (
                    timesteps,
                    states,
                    actions,
                    returns_to_go,
                    traj_mask,
                ) = next(self.data_iter)
        return (
                    timesteps,
                    states,
                    actions,
                    returns_to_go,
                    traj_mask,
                )
        
    def train_step_benchmark(self, update):
            timesteps, states, actions, rtg, traj_mask = self.get_next()
            timesteps = timesteps.to(self.device)      # B x T
            states = states.to(self.device)            # B x T x state_dim
            actions = actions.to(self.device)          # B x T x act_dim
            rtg = rtg.to(self.device).unsqueeze(
                dim=-1
            )                                       # B x T x 1
            traj_mask = traj_mask.to(self.device)      # B x T
            # model forward ----------------------------------------------
            (
                returns_to_go_preds,
                actions_dist_preds
            ) = self.model.forward(
                timesteps=timesteps,
                states=states,
                actions=actions,
                returns_to_go=rtg,
            )
            returns_to_go_target = torch.clone(rtg).view(
                -1, 1
            )[
                traj_mask.view(-1,) > 0
            ]
            returns_to_go_preds = returns_to_go_preds.view(-1, 1)[
                traj_mask.view(-1,) > 0
            ]

            # returns_to_go_loss -----------------------------------------
            norm = returns_to_go_target.abs().mean()
            u = (returns_to_go_target - returns_to_go_preds) / norm
            returns_to_go_loss = torch.mean(
                torch.abs(
                    self.tau - (u < 0).float()
                ) * u ** 2
            )
            # action_loss ------------------------------------------------
            actions_target = torch.clone(actions)
            log_likelihood = actions_dist_preds.log_prob(
                actions_target
                ).sum(axis=2)[
                traj_mask > 0
            ].mean()
            entropy = actions_dist_preds.entropy().sum(axis=2).mean()
            action_loss = -(log_likelihood + self.model.temp().detach() * entropy)
            wandb.log({"rtg_loss": returns_to_go_loss, "act_log_likelihood":-log_likelihood,"temperature_loss":self.model.temp().detach() * entropy})
            loss = (returns_to_go_loss + action_loss) / self.acc_grad
            
            # optimizer -----------------------------------------------
            loss.backward()
            if update:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_norm
                )
                self.optimizer.step()
                
                self.optimizer.zero_grad()
            # scheduler -----------------------------------------------
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (
                    self.model.temp() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

            self.scheduler.step()
            
            
            # diagnostics -----------------------------------------------
            """with torch.no_grad():
                self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()"""

            return loss.detach().cpu().item()