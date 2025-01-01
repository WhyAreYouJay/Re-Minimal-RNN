import warnings

warnings.filterwarnings("ignore")


def warn(*args, **kwargs):
    pass


import argparse
import gymnasium as gym
import numpy as np
import torch
import wandb
from eval import Reinformer_eval
from deep_learning_rl_sm.utils import *
from deep_learning_rl_sm.trainer.trainer import Trainer
from deep_learning_rl_sm.neuralnets.minGRU_Reinformer import minGRU_Reinformer
from deep_learning_rl_sm.neuralnets.lamb import Lamb
from deep_learning_rl_sm.environments import connect_four
from torch.utils.data import Dataset, DataLoader
import random


#os.environ["WANDB_MODE"]=offline

class D4RLDataset(Dataset):
    def __init__(self, s, a, rtg, seq_len, scale):
        self.s = s
        self.s_shape = list(s[0].shape)
        self.a = a
        self.a_shape = list(a[0].shape)
        self.rtg = [r / scale for r in rtg]
        self.rtg_shape = list(rtg[0].shape)
        self.seq_len = seq_len
        self.rng = random.randint

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        if self.s[idx].shape[0] > self.seq_len:
            si = self.rng(0, self.s[idx].shape[0] - self.seq_len)
            s, a, rtg = torch.from_numpy(self.s[idx][si:si + self.seq_len]), torch.from_numpy(self.a[idx][si:si + self.seq_len]), torch.from_numpy(self.rtg[idx][si:si + self.seq_len])
            t = torch.arange(si, si+self.seq_len,1)
            mask = torch.ones(self.seq_len)
        else:
            pad_len = self.seq_len - self.s[idx].shape[0]
            t = torch.arange(start=0, end=self.seq_len, step=1)
            s = torch.cat([torch.from_numpy(self.s[idx]), torch.zeros([pad_len] + self.s_shape[1:])], dim=0)
            a = torch.cat([torch.from_numpy(self.a[idx]), torch.zeros([pad_len] + self.a_shape[1:])], dim=0)
            rtg = torch.cat([torch.from_numpy(self.rtg[idx]), torch.zeros([pad_len] + self.rtg_shape[1:])], dim=0)
            mask = torch.cat([torch.ones(self.seq_len - pad_len), torch.zeros(pad_len)], dim=0)
        return (t, s, a, rtg, mask)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["reinformer"], default="reinformer")
    parser.add_argument("--env", choices=["halfcheetah", "walker2d", "hopper"], default="hopper")
    parser.add_argument("--env_discrete", type=bool, default=False)
    parser.add_argument("--dataset", choices=["medium", "medium_expert", "medium_replay"], type=str, default="medium_expert")
    parser.add_argument("--num_eval_ep", type=int, default=10)
    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str,
                        default="deep_learning_rl_sm/benchmarks/data/halfcheetah_medium_expert-v2.hdf5")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--num_steps_per_iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--conv", type=bool, default=False)
    parser.add_argument("--block_type", type=str, default="mingru")
    parser.add_argument("--std_cond_on_input", type=bool, default=False)
    parser.add_argument("--stacked", type=bool, default=False)
    parser.add_argument("--expansion_factor", type=float, default=2.0)
    parser.add_argument("--mult", type=float, default=4.0)
    parser.add_argument("--acc_grad", type=int, default=1)
    parser.add_argument("--graph_name", type=str, default="")
    parser.add_argument("--reuse_emb", type=bool, default=False)
    

    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=True)
    return parser.parse_args()


if __name__=="__main__":
    for seed in [0,42,2024]:
        args = parse_args()
        settings = f"{args.K}-{args.batch_size}-{args.lr}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = str(device) == "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True if gpu else False
        if gpu:
            torch.set_float32_matmul_precision("highest")
        #seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if args.use_wandb:
            wandb.login()
            wandb.init(
                name=args.env + "-" + args.dataset+"-"+settings,
                project="Reinformer",
                config=vars(args)
            )
        env = args.env + "_" + args.dataset + "-v2"
        fp = download_dataset_from_url(env)
        max_ep_len = 1000  # Same for all 3 envs (Hopper, Walker, HalfCheetah)
        scale = 1000  # Normalization for rewards/returns
        if args.env in ["walker2d"]:
            env_name = "Walker2d"
        if args.env in ["hopper"]:
            env_name = "Hopper"
        if args.env in ["halfcheetah"]:
            scale = 5000
            env_name = "HalfCheetah"
        observations, acts, rew_to_gos, state_mean, state_std = benchmark_data(fp)
        environment = gym.make(env_name + "-v4")


        def get_normalized_score(score, env=env):
            return (score - REF_MIN_SCORE[env]) / (REF_MAX_SCORE[env] - REF_MIN_SCORE[env])


        def evaluator(model):
            return_mean, _, _, _ = Reinformer_eval(
                seed=seed,
                model=model,
                device=device,
                context_len=args["K"],
                env=environment,
                state_mean=state_mean,
                state_std=state_std,
                num_eval_ep=args["num_eval_ep"],
                max_test_ep_len=args["max_eval_ep_len"]
            )
            return get_normalized_score(
                return_mean
            ) * 100


        state_dim, act_dim = observations[0].shape[1], acts[0].shape[1]
        # entropy to encourage exploration in RL typically -action_dim for continuous actions and -log(action_dim) when discrete
        args = vars(args)
        """args["tau"] = taus[args["env"]][args["dataset"]]
        args["K"] = K
        args["batch_size"] = batch_size
        args["embed_dim"] = embed_dim
        args["lr"] = lr
        args["seed"] = seed"""
        target_entropy = -np.log(np.prod(act_dim)) if args["env_discrete"] else -np.prod(act_dim)
        model = minGRU_Reinformer(state_dim=state_dim, act_dim=act_dim, expansion_factor = args["expansion_factor"], mult = args["mult"],
                                h_dim=args["embed_dim"], n_layers=args["n_layers"], stacked = args["stacked"],
                                drop_p=args["dropout_p"], init_tmp=args["init_temperature"],reuse_emb=args["reuse_emb"],
                                target_entropy=target_entropy, discrete=args["env_discrete"],
                                batch_size=args["batch_size"], device=device, max_timestep=max_ep_len, conv=args["conv"],
                                std_cond_on_input=args["std_cond_on_input"], block_type=args["block_type"])
        model = model.to(device)
        """if gpu:
            torch.compile(model=model, mode="max-autotune")"""
        optimizer = Lamb(
            model.parameters(),
            lr=args["lr"],
            weight_decay=args["wd"],
            eps=args["eps"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / args["warmup_steps"], 1)
        )
        # perhaps dynamically incease K
        dataset = D4RLDataset(observations, acts, rew_to_gos, args["K"], scale)
        traj_data_loader = DataLoader(
            dataset,
            batch_size=args["batch_size"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed)
        )
        trainer = Trainer(model=model, data_loader=traj_data_loader, optimizer=optimizer, scheduler=scheduler,
                        parsed_args=args, batch_size=args["batch_size"], device=device, acc_grad=args["acc_grad"])
        if gpu:
            torch.backends.cudnn.benchmark = True
        d4rl_norm_scores = []
        for it in range(args["max_iters"]):
            outputs = trainer.train_iteration_benchmark(num_steps=args['num_steps_per_iter'], iter_num=it + 1,
                                                        print_logs=True)
            # Eval
            with torch.no_grad():
                for b in trainer.model.blocks:
                    if not args["stacked"]:
                        b.cell.eval_mode()
                    else:
                        for cell in b.cells:
                            cell.eval_mode()
                d4rl_norm_scores.append(evaluator(trainer.model))
                print(60 * "=")
                if args["use_wandb"]:
                    wandb.log({f"Normalized_Score_{env}_{settings}{args['graph_name']}": d4rl_norm_scores[-1]})
                print(f"Normalized Score for {env} : {d4rl_norm_scores[-1]}")
                print(60 * "=")
                trainer.model.train()
                for b in trainer.model.blocks:
                    if not args["stacked"]:
                        b.cell.train_mode()
                    else:
                        for cell in b.cells:
                            cell.train_mode()
        wandb.finish()
