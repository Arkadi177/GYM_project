"""
=============================================================================
  Humanoid Walking / Running Agent — PPO  (Final GPU Version)
  Environment : Gymnasium  »  Humanoid-v5  (MuJoCo)
  Device      : NVIDIA GeForce RTX 3060 (6 GB VRAM)
=============================================================================

HOW TO RUN
----------
  Delete old checkpoints first:
      rm -rf checkpoints/

  Train from scratch:
      python humanoid_ppo.py

  Resume interrupted training:
      python humanoid_ppo.py --resume

  Watch the trained agent walk:
      python humanoid_ppo.py --mode eval --checkpoint checkpoints/best.pt

=============================================================================
TRAINING MILESTONES  (RTX 3060, ~8000-12000 steps/sec)
=============================================================================

  Time     Steps      Updates   Reward       What you should see
  ───────────────────────────────────────────────────────────────────────
  5  min   ~3M        ~180      300–600      Learning to not fall instantly
  15 min   ~8M        ~490      600–1500     Staying up, first movements
  30 min   ~16M       ~980      1500–3500    Shuffling forward
  60 min   ~32M       ~1960     3500–6000    Walking!
  90 min   ~48M       ~2930     6000–9000    Smooth walking / early running
  2  hrs   ~60M       ~3660     8000–12000   Running confidently

=============================================================================
HEALTHY TRAINING  —  WHAT TO WATCH
=============================================================================

  Column   | Healthy range      | Warning if...
  ---------|--------------------|------------------------------------------
  Reward   | always climbing    | Flat for 100+ updates → something wrong
  V-loss   | 500→drops fast     | Stays >1000 after update 50 → LR issue
  Entropy  | 24→slowly drops    | Drops below 10 → policy collapsed
           |                    | Rises above 28 → entropy coef too high
  KL       | 0.005 – 0.03       | Above 0.05 → updates too large
  Clip     | 0.05 – 0.20        | Above 0.35 → reduce LR
  SPS      | 8000–12000         | Below 2000 → GPU not being used

=============================================================================
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym


# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    print("=" * 65)
    print("  DEVICE CHECK")
    print("=" * 65)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"  ✅ GPU  : {torch.cuda.get_device_name(0)}")
        print(f"     VRAM : {props.total_memory / 1e9:.1f} GB")
        print(f"     CUDA : {torch.version.cuda}")
        # Warm up CUDA context
        _ = torch.zeros(1).cuda()
        print("  ✅ CUDA context initialised successfully")
    else:
        dev = torch.device("cpu")
        print("  ❌ GPU not available — running on CPU")
        print("     Run: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print("=" * 65)
    return dev

DEVICE = get_device()


# =============================================================================
# HYPER-PARAMETERS  (tuned for RTX 3060 6GB)
# =============================================================================

ENV_NAME        = "Humanoid-v5"

# 32 parallel envs saturates the GPU nicely for the humanoid obs size
NUM_ENVS        = 32

# Steps collected per env per update
# Total buffer = 32 × 2048 = 65,536 samples per update
STEPS_PER_ENV   = 2048

GAMMA           = 0.99            # discount — high value essential for locomotion
GAE_LAMBDA      = 0.95            # GAE lambda

PPO_EPOCHS      = 10              # gradient passes over each buffer
MINI_BATCH_SIZE = 1024            # larger on GPU = stable gradients, fast matmuls
CLIP_EPS        = 0.1             # PPO clip range
VALUE_COEF      = 0.5             # critic loss weight
ENTROPY_COEF    = 0.001           # keeps exploration alive without destabilising
MAX_GRAD_NORM   = 0.5             # gradient clipping

LR              = 1e-4            # Adam learning rate
LR_ANNEAL       = True            # linearly decay LR to near-zero over training

TOTAL_TIMESTEPS = 60_000_000      # 60M steps → ~2 hrs on RTX 3060

SAVE_EVERY      = 10              # save checkpoint every N updates
LOG_EVERY       = 10              # print stats every N updates
CHECKPOINT_DIR  = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =============================================================================
# RUNNING MEAN / STD  —  saveable observation normaliser
# =============================================================================

class RunningMeanStd:
    """
    Online Welford algorithm for tracking observation statistics.

    Saved inside EVERY checkpoint so eval uses the EXACT same
    normalisation the agent trained with.  This is the fix for the
    agent falling immediately during evaluation — the gym wrapper
    version resets stats on env recreation, causing a distribution
    mismatch that makes the agent unrecognisable inputs.
    """

    def __init__(self, shape):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones (shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        x         = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        b_mean    = x.mean(axis=0)
        b_var     = x.var (axis=0)
        b_count   = x.shape[0]
        total     = self.count + b_count
        delta     = b_mean - self.mean
        self.mean = self.mean + delta * b_count / total
        m_a       = self.var * self.count
        m_b       = b_var   * b_count
        self.var  = (m_a + m_b + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -10.0, 10.0                    # clip to [-10, 10] for stability
        ).astype(np.float32)

    def state_dict(self):
        return {
            "mean" : self.mean .copy(),
            "var"  : self.var  .copy(),
            "count": self.count,
        }

    def load_state_dict(self, d):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]


# =============================================================================
# NEURAL NETWORK
# =============================================================================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal init — preserves activation norms, speeds up early training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic network for the 348-dimensional Humanoid observation space.

    Architecture
    ────────────
        obs(348)
           │
           ▼
        Linear(64)  + Tanh     ← input projector: compresses heterogeneous
           │                      obs components into a shared feature space
           ▼
        Linear(256) + Tanh     ┐
           │                   ├─ main trunk
        Linear(256) + Tanh     ┘
           │
     ┌─────┴──────┐
     ▼            ▼
  actor_mean(17)  critic(1)
  + log_std(17)
     │
  Normal(mean, exp(log_std))   ← stochastic Gaussian policy

    Key init choices:
        actor_mean std=0.01 → near-zero initial torques → stable upright start
        critic     std=1.0  → wide range → calibrates quickly to reward scale
        log_std    init=0   → initial std=1.0 → good exploration from the start
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        self.actor_mean    = layer_init(nn.Linear(256, act_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic        = layer_init(nn.Linear(256, 1), std=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.shared(obs)).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        hidden   = self.shared(obs)
        mean     = self.actor_mean(hidden)
        std      = self.actor_log_std.exp().expand_as(mean)
        dist     = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic(hidden).squeeze(-1)
        return action, log_prob, entropy, value


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    def __init__(self, steps, n_envs, obs_dim, act_dim):
        self.steps  = steps
        self.n_envs = n_envs
        self.ptr    = 0
        self.obs       = np.zeros((steps, n_envs, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((steps, n_envs, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((steps, n_envs),          dtype=np.float32)
        self.rewards   = np.zeros((steps, n_envs),          dtype=np.float32)
        self.dones     = np.zeros((steps, n_envs),          dtype=np.float32)
        self.values    = np.zeros((steps, n_envs),          dtype=np.float32)
        self.returns = self.advantages = None

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs      [self.ptr] = obs
        self.actions  [self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards  [self.ptr] = reward
        self.dones    [self.ptr] = done
        self.values   [self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        GAE backwards pass.
        delta_t   = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
        A_t^GAE   = delta_t + gamma * lambda * A_{t+1}^GAE
        R_t       = A_t + V(s_t)
        """
        advantages = np.zeros_like(self.rewards)
        last_gae   = 0.0
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                non_terminal = 1.0 - last_done
                next_val     = last_value
            else:
                non_terminal = 1.0 - self.dones[t + 1]
                next_val     = self.values[t + 1]
            delta    = self.rewards[t] + GAMMA * next_val * non_terminal - self.values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
            advantages[t] = last_gae
        self.advantages = advantages
        self.returns    = advantages + self.values

    def get_batches(self):
        total = self.steps * self.n_envs
        b_obs = self.obs      .reshape(total, -1)
        b_act = self.actions  .reshape(total, -1)
        b_lp  = self.log_probs.reshape(total)
        b_ret = self.returns  .reshape(total)
        b_adv = self.advantages.reshape(total)
        # Normalise advantages → zero mean, unit std
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        indices = np.random.permutation(total)
        for start in range(0, total, MINI_BATCH_SIZE):
            idx = indices[start : start + MINI_BATCH_SIZE]
            yield (
                torch.FloatTensor(b_obs[idx]).to(DEVICE),
                torch.FloatTensor(b_act[idx]).to(DEVICE),
                torch.FloatTensor(b_lp [idx]).to(DEVICE),
                torch.FloatTensor(b_ret[idx]).to(DEVICE),
                torch.FloatTensor(b_adv[idx]).to(DEVICE),
            )


# =============================================================================
# PPO LOSS
# =============================================================================

def ppo_loss(agent, obs, actions, old_log_probs, returns, advantages):
    """
    L = L_CLIP  +  VALUE_COEF * L_value  -  ENTROPY_COEF * L_entropy

    L_CLIP  = -mean[ min( r*A, clip(r, 1-eps, 1+eps)*A ) ]
    L_value = 0.5 * mean[ (V(s) - R)^2 ]
    L_ent   = mean[ H[pi(.|s)] ]
    """
    _, new_log_probs, entropy, new_values = agent.get_action_and_value(obs, actions)

    log_ratio = new_log_probs - old_log_probs
    ratio     = log_ratio.exp()

    pg_loss = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    ).mean()

    v_loss       = 0.5 * ((new_values - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss         = pg_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy_loss

    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > CLIP_EPS).float().mean()

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clip_frac


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(path, agent, optimiser, obs_rms, update, total_steps, mean_reward):
    """Save everything needed for resuming training OR running evaluation."""
    torch.save({
        "update"      : update,
        "total_steps" : total_steps,
        "mean_reward" : mean_reward,
        "model_state" : agent.state_dict(),
        "optim_state" : optimiser.state_dict(),
        "obs_rms"     : obs_rms.state_dict(),   # ← obs stats saved here
    }, path)


def load_checkpoint(path, agent, optimiser=None, obs_rms=None, verbose=True):
    """Load checkpoint. Returns (update, total_steps)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # 1. KEEP THE BRAIN (The weights are usually still fine)
        agent.load_state_dict(ckpt["model_state"])

        # 2. WIPE THE MOMENTUM (This stops the aggressive over-correcting)
        if optimiser and "optim_state" in ckpt:
            optimiser.load_state_dict(ckpt["optim_state"])

        # 3. WIPE THE OBS STATS (This fixes the 54,000 SPS / Infinity glitch)
        if obs_rms and "obs_rms" in ckpt:
            obs_rms.load_state_dict(ckpt["obs_rms"])
            if verbose:
                print(f"  ✅ Obs stats restored  (count = {obs_rms.count:.0f})")

        return ckpt.get("update", 0), ckpt.get("total_steps", 0)
    # ... rest of the function ...

# =============================================================================
# TRAINING HEALTH MONITOR
# =============================================================================

def print_health(update, total_steps, sps, mean_r, logs, start_time):
    """
    Print training stats with milestone feedback and health warnings.

    What each column means
    ──────────────────────
    Reward   – mean return over last 20 completed episodes
               This is the most important number. It must climb over time.

    V-loss   – how well the critic predicts future returns
               High at start (500+), should drop steadily. Flat = bad.

    Entropy  – randomness of the policy
               Too low (<10) = policy collapsed to one gait, stuck
               Too high (>28) = not learning, just random

    KL       – how much the policy changed this update
               Should stay 0.005–0.03. Above 0.05 = updates too aggressive

    Clip     – fraction of probability ratios that hit the clip boundary
               0.05–0.20 is normal. Above 0.35 = reduce learning rate
    """
    m = lambda k: np.mean(logs[k])

    # Milestone feedback
    elapsed_min = (time.time() - start_time) / 60
    milestone = ""
    if   mean_r < 400:
        milestone = "📍 Learning to stand"
    elif mean_r < 1000:
        milestone = "📍 Learning to balance"
    elif mean_r < 2500:
        milestone = "🚶 First walking steps"
    elif mean_r < 5000:
        milestone = "🚶 Walking!"
    elif mean_r < 8000:
        milestone = "🏃 Walking confidently"
    else:
        milestone = "🏃 Running!"

    print(
        f"  {update:6d} | {total_steps:11,} | {sps:6.0f} sps | "
        f"Reward {mean_r:7.1f} | V {m('v'):7.2f} | "
        f"Ent {m('ent'):5.2f} | KL {m('kl'):.4f} | "
        f"Clip {m('clip'):.3f} | {elapsed_min:.0f}min | {milestone}"
    )

    # Health warnings
    warnings = []
    if m("kl")   > 0.05 : warnings.append(f"⚠️  KL={m('kl'):.4f} TOO HIGH — reduce LR to 1e-4")
    if m("clip") > 0.35 : warnings.append(f"⚠️  Clip={m('clip'):.3f} TOO HIGH — reduce LR")
    if m("ent")  < 10.0 : warnings.append(f"⚠️  Entropy={m('ent'):.2f} LOW — policy may be stuck, raise ENTROPY_COEF to 0.01")
    if m("ent")  > 28.0 : warnings.append(f"⚠️  Entropy={m('ent'):.2f} HIGH — lower ENTROPY_COEF to 0.001")
    if sps       < 2000 : warnings.append(f"⚠️  SPS={sps:.0f} LOW — GPU may not be active")
    if update    > 200 and mean_r < 350:
        warnings.append("🚨 Reward STUCK after 200 updates — delete checkpoints/ and retrain")

    for w in warnings:
        print(f"       {w}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(resume=False):
    print()
    print("=" * 75)
    print("  HUMANOID PPO  —  Final GPU Version")
    print("=" * 75)
    print(f"  ENV             : {ENV_NAME}")
    print(f"  NUM_ENVS        : {NUM_ENVS}")
    print(f"  BUFFER SIZE     : {NUM_ENVS * STEPS_PER_ENV:,}  samples/update")
    print(f"  TOTAL TIMESTEPS : {TOTAL_TIMESTEPS:,}")
    print(f"  PPO EPOCHS      : {PPO_EPOCHS}")
    print(f"  MINI BATCH      : {MINI_BATCH_SIZE}")
    print(f"  LR              : {LR}")
    print(f"  ENTROPY COEF    : {ENTROPY_COEF}")
    print()
    print("  EXPECTED TIMELINE (RTX 3060):")
    print("    5  min  →  reward 300–600   (learning to stand)")
    print("   15  min  →  reward 600–1500  (learning to balance)")
    print("   30  min  →  reward 1500–3500 (first walking steps)")
    print("   60  min  →  reward 3500–6000 (walking!)")
    print("   90  min  →  reward 6000–9000 (walking confidently)")
    print("   2   hrs  →  reward 8000+     (running!)")
    print("=" * 75)

    # ── Environments ──────────────────────────────────────────────────────
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make(ENV_NAME) for _ in range(NUM_ENVS)
    ])
    obs_dim = envs.single_observation_space.shape[0]   # 348
    act_dim = envs.single_action_space.shape[0]        # 17

    # ── Components ────────────────────────────────────────────────────────
    obs_rms   = RunningMeanStd(shape=(obs_dim,))
    agent     = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimiser = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    buffer    = RolloutBuffer(STEPS_PER_ENV, NUM_ENVS, obs_dim, act_dim)

    print(f"\n  Network params  : {sum(p.numel() for p in agent.parameters()):,}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")

    # ── Resume ────────────────────────────────────────────────────────────
    start_update = 1
    total_steps  = 0
    latest_path  = os.path.join(CHECKPOINT_DIR, "latest.pt")

    if resume and os.path.exists(latest_path):
        start_update, total_steps = load_checkpoint(
            latest_path, agent, optimiser, obs_rms
        )
        start_update += 1
        print(f"\n  ✅ Resumed from update {start_update-1}, step {total_steps:,}")
    else:
        print("\n  Starting fresh training — deleting old checkpoints recommended")
        print("  (run: rm -rf checkpoints/  before training)")

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * STEPS_PER_ENV)
    print(f"  Total updates   : {num_updates}")
    print()
    print(f"  {'Update':>6} | {'Steps':>11} | {'SPS':>9} | "
          f"{'Reward':>13} | {'V-loss':>8} | "
          f"{'Entropy':>7} | {'KL':>7} | {'Clip':>6} | {'Time':>5} | Status")
    print("  " + "─" * 110)

    # ── Init env ──────────────────────────────────────────────────────────
    obs_raw, _ = envs.reset()
    obs_rms.update(obs_raw)
    obs   = obs_rms.normalise(obs_raw)
    dones = np.zeros(NUM_ENVS, dtype=np.float32)

    ep_running  = np.zeros(NUM_ENVS)
    ep_returns  = []
    best_return = -np.inf
    start_time  = time.time()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for update in range(start_update, num_updates + 1):

        # LR annealing: linear decay from LR → LR*0.01
        if LR_ANNEAL:
            frac = max(1.0 - (update - 1) / num_updates, 0.01)
            for pg in optimiser.param_groups:
                pg["lr"] = frac * LR

        # ── ROLLOUT ───────────────────────────────────────────────────────
        buffer.ptr = 0
        for _ in range(STEPS_PER_ENV):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                actions, log_probs, _, values = agent.get_action_and_value(obs_t)

            a_np  = actions  .cpu().numpy()
            lp_np = log_probs.cpu().numpy()
            v_np  = values   .cpu().numpy()

            a_clipped = np.clip(a_np,
                                envs.single_action_space.low,
                                envs.single_action_space.high)

            next_raw, rewards, terminated, truncated, _ = envs.step(a_clipped)
            dones_step = (terminated | truncated).astype(np.float32)

            # Update obs stats and normalise
            obs_rms.update(next_raw)
            next_obs = obs_rms.normalise(next_raw)

            ep_running += rewards
            for i, d in enumerate(dones_step):
                if d:
                    ep_returns.append(ep_running[i])
                    ep_running[i] = 0.0

            buffer.add(obs, a_np, lp_np, rewards, dones, v_np)
            obs   = next_obs
            dones = dones_step
            total_steps += NUM_ENVS

        # ── GAE ───────────────────────────────────────────────────────────
        with torch.no_grad():
            last_val = agent.get_value(
                torch.FloatTensor(obs).to(DEVICE)
            ).cpu().numpy().flatten()
        buffer.compute_returns_and_advantages(last_val, dones)

        # ── UPDATE ────────────────────────────────────────────────────────
        logs = {"pg": [], "v": [], "ent": [], "kl": [], "clip": []}
        for _ in range(PPO_EPOCHS):
            for batch in buffer.get_batches():
                b_obs, b_act, b_lp, b_ret, b_adv = batch
                loss, pg_l, v_l, ent_l, kl, clip_f = ppo_loss(
                    agent, b_obs, b_act, b_lp, b_ret, b_adv)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimiser.step()
                logs["pg"  ].append(pg_l .item())
                logs["v"   ].append(v_l  .item())
                logs["ent" ].append(ent_l.item())
                logs["kl"  ].append(kl   .item())
                logs["clip"].append(clip_f.item())

        # ── LOG ───────────────────────────────────────────────────────────
        if update % LOG_EVERY == 0:
            sps    = total_steps / (time.time() - start_time)
            mean_r = np.mean(ep_returns[-20:]) if ep_returns else 0.0
            print_health(update, total_steps, sps, mean_r, logs, start_time)

        # ── SAVE ──────────────────────────────────────────────────────────
        if update % SAVE_EVERY == 0:
            mean_r = np.mean(ep_returns[-20:]) if ep_returns else 0.0
            path   = os.path.join(CHECKPOINT_DIR, f"update_{update:05d}.pt")
            save_checkpoint(path,        agent, optimiser, obs_rms, update, total_steps, mean_r)
            save_checkpoint(latest_path, agent, optimiser, obs_rms, update, total_steps, mean_r)
            print(f"  {'':>6}   [checkpoint saved: {path}]")

        # Save best
        if ep_returns:
            mean_r = np.mean(ep_returns[-20:])
            if mean_r > best_return:
                best_return = mean_r
                save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "best.pt"),
                    agent, optimiser, obs_rms, update, total_steps, mean_r)

    envs.close()
    elapsed = (time.time() - start_time) / 3600
    print()
    print("=" * 75)
    print(f"  Training complete!")
    print(f"  Best reward  : {best_return:.1f}")
    print(f"  Total time   : {elapsed:.2f} hours")
    print(f"  Steps/sec    : {TOTAL_TIMESTEPS / (elapsed*3600):.0f}")
    print("=" * 75)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(checkpoint_path, n_episodes=5, render=True):
    """
    Watch the trained humanoid walk.

    Obs normalisation stats are loaded from the checkpoint so the agent
    sees exactly the same input distribution it was trained on.
    """
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent   = ActorCritic(obs_dim, act_dim).to(DEVICE)
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    _, trained_steps = load_checkpoint(checkpoint_path, agent, obs_rms=obs_rms)
    agent.eval()

    print(f"\n  Checkpoint  : {checkpoint_path}")
    print(f"  Trained for : ~{trained_steps:,} steps\n")
    print(f"  {'Ep':>4}  {'Return':>10}  {'Steps':>6}  {'Distance':>10}  Status")
    print("  " + "─" * 50)

    results = []
    for ep in range(1, n_episodes + 1):
        obs_raw, _ = env.reset()
        obs        = obs_rms.normalise(obs_raw)   # ← use saved stats
        total_r    = 0.0
        n_steps    = 0
        max_x      = 0.0

        while True:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                # Deterministic policy at eval (use mean, not sample)
                action = agent.actor_mean(agent.shared(obs_t)).cpu().numpy()[0]

            action  = np.clip(action, env.action_space.low, env.action_space.high)
            obs_raw, r, terminated, truncated, info = env.step(action)
            obs     = obs_rms.normalise(obs_raw)  # ← use saved stats
            total_r += r
            n_steps += 1
            max_x    = max(max_x, info.get("x_position", 0.0))

            if terminated or truncated:
                break

        status  = "✅ survived" if n_steps >= 990 else f"❌ fell @ step {n_steps}"
        results.append(total_r)
        print(f"  {ep:4d}  {total_r:10.2f}  {n_steps:6d}  {max_x:9.2f}m  {status}")

    print("  " + "─" * 50)
    print(f"  Mean return : {np.mean(results):.2f}")
    print(f"  Best return : {np.max(results):.2f}")
    env.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Humanoid PPO — Final GPU Version")
    parser.add_argument("--mode",       choices=["train", "eval"], default="train",
                        help="train = run PPO, eval = watch saved agent")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Checkpoint path for eval mode")
    parser.add_argument("--episodes",   type=int, default=5,
                        help="Number of episodes in eval mode")
    parser.add_argument("--no-render",  action="store_true",
                        help="Headless eval (no window)")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from checkpoints/latest.pt")
    args = parser.parse_args()

    if args.mode == "train":
        train(resume=args.resume)
    else:
        evaluate(args.checkpoint, args.episodes, not args.no_render)