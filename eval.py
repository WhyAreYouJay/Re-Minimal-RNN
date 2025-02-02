import torch
import numpy as np


def Reinformer_eval(
    seed,
    model,
    device,
    context_len,
    env,
    state_mean,
    state_std,
    num_eval_ep=10,
    max_test_ep_len=1000,
):
    eval_batch_size = 1
    returns = []
    lengths = []

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.reset(seed = seed)
    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)

    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(num_eval_ep):
            # zeros place holders
            actions = torch.zeros(
                (eval_batch_size, max_test_ep_len, act_dim),
                dtype=torch.float32,
                device=device,
            )
            states = torch.zeros(
                (eval_batch_size, max_test_ep_len, state_dim),
                dtype=torch.float32,
                device=device,
            )
            returns_to_go = torch.zeros(
                (eval_batch_size, max_test_ep_len, 1),
                dtype=torch.float32,
                device=device,
            )

            # init episode
            running_state = env.reset()[0]
            episode_return = 0
            episode_length = 0

            for t in range(max_test_ep_len):
                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std
                #print(f"State {t} : {states[0,t]}")
                # predict rtg by model
                if t < context_len:
                    #print(f"FW1<:{states[:, :context_len]}")
                    rtg_preds, _ = model.forward(
                        timesteps[:, :context_len],
                        states[:, :context_len],
                        actions[:, :context_len],
                        returns_to_go[:, :context_len],
                    )
                    rtg = rtg_preds[0, t].detach()
                else:
                    #print(f"FW1>:{states[:, :context_len]}")
                    rtg_preds, _  = model.forward(
                        timesteps[:, t - context_len + 1 : t + 1],
                        states[:, t - context_len + 1 : t + 1],
                        actions[:, t - context_len + 1 : t + 1],
                        returns_to_go[:, t - context_len + 1 : t + 1],
                    )
                    rtg = rtg_preds[0, -1].detach()
                #print(f"RTG {t} : {rtg}")
                # add rtg in placeholder
                returns_to_go[0, t] = rtg
                # take action by model
                if t < context_len:
                    #print(f"FW2<:{states[:, :context_len]}")
                    _, act_dist_preds = model.forward(
                        timesteps[:, :context_len],
                        states[:, :context_len],
                        actions[:, :context_len],
                        returns_to_go[:, :context_len],
                    )
                    act = act_dist_preds.mean.reshape(eval_batch_size, -1, act_dim)[0, t].detach()
                else:
                    #print(f"FW2>:{states[:, :context_len]}")
                    _, act_dist_preds = model.forward(
                        timesteps[:, t - context_len + 1 : t + 1],
                        states[:, t - context_len + 1 : t + 1],
                        actions[:, t - context_len + 1 : t + 1],
                        returns_to_go[:, t - context_len + 1 : t + 1],
                    )
                    act = act_dist_preds.mean.reshape(eval_batch_size, -1, act_dim)[0, -1].detach()
                # env step
                running_state, running_reward, done, _, _ = env.step(
                    act.cpu().numpy()
                )
                # add action in placeholder
                actions[0, t] = act
                #print(f"Action {t} : {actions[0,t]}")
                #print(f"Reward rec. {t} : {running_reward}")
                # calculate return and episode length
                episode_return += running_reward
                episode_length += 1
                # terminate
                if done or t >= max_test_ep_len -1:
                    returns.append(episode_return)
                    lengths.append(episode_length)
                    print(f"Episode {i} returns : {episode_return}")
                    break
    
    return np.array(returns).mean(), np.array(returns).std(), np.array(lengths).mean(), np.array(lengths).mean()