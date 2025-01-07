import torch
import numpy as np


def evaluate(
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
            model.reset_h_prev()
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
                rtg_pred = model.get_rtg(
                    timesteps[:, t].unsqueeze(0),
                    states[:, t].unsqueeze(0),
                )
                rtg = rtg_pred[0, -1].detach()
                #print(f"RTG {t} : {rtg}")
                # add rtg in placeholder
                returns_to_go[0, t] = rtg
                # take action by model
                act_dist_preds = model.get_action(
                    timesteps[:, t].unsqueeze(0),
                    returns_to_go[:, t].unsqueeze(0),
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
