"""
evaluate_pusher.py
==================
Carga el modelo SAC entrenado (sac_pusher_combined.zip) y:
  1. Ejecuta N episodios guardando GIFs de cada uno.
  2. Genera saliency maps frame a frame (gradient-based sobre la CNN)
     y los guarda como GIFs separados + un panel resumen por episodio.
 
Uso:
    python evaluate_pusher.py \
        --model sac_pusher_combined.zip \
        --vecnorm sac_pusher_vecnormalize.pkl \   # opcional
        --n_episodes 5 \
        --out_dir ./eval_results \
        --saliency_every 10          # guarda saliency cada N pasos
"""
 
import os
import argparse
from collections import deque
 
import cv2
import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
 
import gymnasium as gym
import torch
import torch.nn as nn
 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
 
# ─────────────────────────────────────────────
#  Reutilizamos exactamente los wrappers del
#  script de entrenamiento (deben estar en el
#  mismo directorio o importarse desde allí).
# ─────────────────────────────────────────────
 
MAX_STEPS = 1000
 
 
# -------- Wrappers (copia exacta del training) --------
 
class StepLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=1000):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
 
    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)
 
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, done, truncated, info
 
 
class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k=3):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        img_space = env.observation_space["image"]
        shp = img_space.shape
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0, high=255,
            shape=(shp[0] * k, shp[1], shp[2]),
            dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs["image"])
        return self._get_obs(obs), info
 
    def observation(self, obs):
        self.frames.append(obs["image"])
        return self._get_obs(obs)
 
    def _get_obs(self, obs):
        return {
            "image": np.concatenate(list(self.frames), axis=0),
            "features": obs["features"]
        }
 
 
class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=(128, 128)):
        super().__init__(env)
        self.shape = shape
        new_obs_space = dict(env.observation_space.spaces)
        new_obs_space["image"] = gym.spaces.Box(
            low=0, high=255,
            shape=(3, shape[0], shape[1]),
            dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(new_obs_space)
 
    def observation(self, obs):
        img = obs["image"]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        resized = cv2.resize(img, self.shape)
        return {
            "image": resized.transpose(2, 0, 1),
            "features": obs["features"]
        }
 
 
class DictObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space
        raw_env = env.unwrapped
        feat_dim = raw_env.observation_space.shape[0]
        self.observation_space = gym.spaces.Dict({
            "image": img_space,
            "features": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(feat_dim,),
                dtype=np.float32
            )
        })
 
    def observation(self, obs):
        return {
            "image": obs,
            "features": self._last_features
        }
 
    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_features = info.get(
            "proprio_obs",
            self.env.unwrapped._get_obs()
        ).astype(np.float32)
        obs = self.observation(raw_obs)
        return obs, reward, terminated, truncated, info
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_features = self.env.unwrapped._get_obs().astype(np.float32)
        return self.observation(obs), info
 
 
# -------- Combined Extractor (idéntico al entrenamiento) --------
 
class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 cnn_output_dim: int = 256,
                 mlp_output_dim: int = 64):
        super().__init__(observation_space, features_dim=cnn_output_dim + mlp_output_dim)
 
        img_space  = observation_space["image"]
        feat_space = observation_space["features"]
        n_input_channels = img_space.shape[0]
 
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
 
        with torch.no_grad():
            sample = torch.as_tensor(img_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
 
        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )
 
        feat_dim = feat_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim),
            nn.ReLU()
        )
 
    def forward(self, observations):
        img     = observations["image"].float() / 255.0
        cnn_out = self.cnn_linear(self.cnn(img))
        feat    = observations["features"].float()
        mlp_out = self.mlp(feat)
        return torch.cat([cnn_out, mlp_out], dim=1)
 
 
# ─────────────────────────────────────────────
#  Saliency Map  (Gradient w.r.t. image input)
# ─────────────────────────────────────────────
 
def compute_saliency(model, obs_dict: dict, device: str = "cuda") -> np.ndarray:
    """
    Calcula el saliency map de la imagen de entrada respecto al
    valor Q de la acción determinista del actor.
 
    Devuelve un array (H, W) en [0, 1] normalizado.
    """
    policy = model.policy
 
    # Convertir obs a tensores con grad
    img_np  = obs_dict["image"].astype(np.float32)          # (C, H, W)
    feat_np = obs_dict["features"].astype(np.float32)        # (F,)
 
    img_t  = torch.tensor(img_np[None], device=device, requires_grad=True)
    feat_t = torch.tensor(feat_np[None], device=device)
 
    obs_t = {"image": img_t, "features": feat_t}
 
    # Forward: actor → acción determinista → critic Q
    policy.set_training_mode(False)
 
    # Extraer features combinadas
    features = policy.extract_features(obs_t, policy.actor.features_extractor)
 
    # Actor: mean action (determinista)
    latent_pi = policy.actor.latent_pi(features)
    mean_action = policy.actor.mu(latent_pi)
 
    # Critic Q1 sobre esa acción
    features_c = policy.extract_features(obs_t, policy.critic.features_extractor)
    q_input = torch.cat([features_c, mean_action], dim=1)
    q_value = policy.critic.qf0(q_input).sum()
 
    # Gradiente respecto a la imagen
    q_value.backward()
 
    grad = img_t.grad.detach().cpu().numpy()[0]  # (C, H, W)
 
    # Agregamos canales (valor absoluto máximo por pixel)
    saliency = np.max(np.abs(grad), axis=0)      # (H, W)
 
    # Normalizar a [0, 1]
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
 
    return saliency
 
 
def overlay_saliency(raw_frame: np.ndarray, saliency: np.ndarray,
                     alpha: float = 0.55) -> np.ndarray:
    """
    Superpone el saliency map (colormap 'hot') sobre el frame RGB.
 
    raw_frame : (H, W, 3) uint8
    saliency  : (Hs, Ws) float [0,1]  – se redimensiona a raw_frame
    Devuelve  : (H, W, 3) uint8
    """
    h, w = raw_frame.shape[:2]
 
    # Redimensionar saliency al tamaño del frame real
    sal_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
 
    # Colormap → RGB
    colormap = cm.hot(sal_resized)[:, :, :3]           # (H, W, 3) float [0,1]
    colormap = (colormap * 255).astype(np.uint8)
 
    # Blend
    overlay = cv2.addWeighted(raw_frame, 1 - alpha, colormap, alpha, 0)
    return overlay
 
 
# ─────────────────────────────────────────────
#  Función para crear el env de evaluación
# ─────────────────────────────────────────────
 
def make_eval_env():
    env = gym.make("Pusher-v5", render_mode="rgb_array")
    env = gym.wrappers.AddRenderObservation(env)
    env = DictObservationWrapper(env)
    env = ResizeObservation(env, (128, 128))
    env = FrameStack(env, k=3)
    env = StepLimitWrapper(env, max_steps=MAX_STEPS)
    return env
 
 
# ─────────────────────────────────────────────
#  Panel de resumen de saliency (figura PNG)
# ─────────────────────────────────────────────
 
def save_saliency_summary(frames_rgb, saliency_maps, episode_idx: int, out_dir: str):
    """
    Guarda una figura con N columnas mostrando frame | saliency overlay.
    """
    n = len(frames_rgb)
    if n == 0:
        return
 
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)
 
    for i, (frame, sal) in enumerate(zip(frames_rgb, saliency_maps)):
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"Frame t={i}", fontsize=9)
        axes[0, i].axis("off")
 
        overlay = overlay_saliency(frame, sal, alpha=0.6)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Saliency overlay", fontsize=9)
        axes[1, i].axis("off")
 
    fig.suptitle(f"Episode {episode_idx} — Saliency Summary", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"ep{episode_idx:02d}_saliency_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saliency] Summary saved → {path}")
 
 
# ─────────────────────────────────────────────
#  Episodio completo
# ─────────────────────────────────────────────
 
def run_episode(model, env, episode_idx: int, out_dir: str,
                saliency_every: int = 10, device: str = "cuda"):
    """
    Ejecuta un episodio, guarda:
      - GIF del episodio (frames reales a resolución completa)
      - GIF de saliency overlay (misma cadencia)
      - PNG de resumen con panel de saliency
    """
    obs, _ = env.reset()
    raw_env = env.unwrapped
 
    render_frames = []       # frames crudos del entorno (alta res)
    saliency_gifs = []       # frames con overlay saliency (alta res)
 
    # Para el panel resumen: tomamos un subconjunto espaciado
    summary_frames_rgb = []
    summary_saliency   = []
 
    done   = False
    step   = 0
    total_reward = 0.0
 
    while not done and step < MAX_STEPS:
        # Frame real (alta resolución)
        raw_frame = raw_env.render()   # (H, W, 3) uint8
 
        # Saliency en cada paso o cada N pasos
        compute_sal = (step % saliency_every == 0)
 
        if compute_sal:
            sal = compute_saliency(model, obs, device=device)
            overlay_frame = overlay_saliency(raw_frame, sal, alpha=0.55)
        else:
            # Reutilizamos el último saliency calculado para no romper la cadencia del GIF
            if saliency_gifs:
                overlay_frame = saliency_gifs[-1]
            else:
                overlay_frame = raw_frame.copy()
            sal = None
 
        render_frames.append(raw_frame)
        saliency_gifs.append(overlay_frame)
 
        # Subconjunto para el panel resumen (1 cada 50 pasos, máx 8)
        if compute_sal and sal is not None and len(summary_frames_rgb) < 8:
            summary_frames_rgb.append(raw_frame)
            summary_saliency.append(sal)
 
        # Acción
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done  = terminated or truncated
        step += 1
 
    print(f"  Episode {episode_idx}: {step} steps | reward = {total_reward:.2f}")
 
    # ── GIF episodio ──────────────────────────────────────
    TARGET_H = 240
    h0, w0 = render_frames[0].shape[:2]
    scale  = TARGET_H / h0
    new_w  = int(w0 * scale)
 
    gif_frames   = [cv2.resize(f, (new_w, TARGET_H)) for f in render_frames[::2]]
    sal_frames   = [cv2.resize(f, (new_w, TARGET_H)) for f in saliency_gifs[::2]]
 
    gif_path = os.path.join(out_dir, f"ep{episode_idx:02d}_rollout.gif")
    imageio.mimsave(gif_path, gif_frames, fps=15)
    print(f"  [video]   Rollout GIF → {gif_path}")
 
    sal_gif_path = os.path.join(out_dir, f"ep{episode_idx:02d}_saliency.gif")
    imageio.mimsave(sal_gif_path, sal_frames, fps=15)
    print(f"  [video]   Saliency GIF → {sal_gif_path}")
 
    # ── Panel resumen ─────────────────────────────────────
    if summary_frames_rgb:
        save_saliency_summary(summary_frames_rgb, summary_saliency,
                              episode_idx, out_dir)
 
    return total_reward
 
 
# ─────────────────────────────────────────────
#  Estadísticas finales
# ─────────────────────────────────────────────
 
def save_reward_plot(rewards: list, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(rewards)), rewards, color="#2196F3", edgecolor="white")
    ax.axhline(np.mean(rewards), color="#FF5722", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(rewards):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Evaluation — Episode Rewards")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "eval_rewards.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n[stats] Reward plot → {path}")
 
 
# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
 
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC Pusher model")
    parser.add_argument("--model",          type=str,   default="sac_pusher_combined.zip",
                        help="Path to the saved model (.zip)")
    parser.add_argument("--vecnorm",        type=str,   default=None,
                        help="Path to VecNormalize stats (.pkl) — opcional")
    parser.add_argument("--n_episodes",     type=int,   default=5,
                        help="Número de episodios a evaluar")
    parser.add_argument("--out_dir",        type=str,   default="./eval_results",
                        help="Directorio de salida para GIFs y figuras")
    parser.add_argument("--saliency_every", type=int,   default=10,
                        help="Calcular saliency cada N pasos (reduce coste)")
    parser.add_argument("--device",         type=str,   default="cuda",
                        help="Dispositivo PyTorch: cuda | cpu")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    os.makedirs(args.out_dir, exist_ok=True)
 
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    # ── Cargar modelo ──────────────────────────────────────
    print(f"\nLoading model from: {args.model}")
 
    # Necesitamos pasar custom_objects para que SB3 reconstruya el extractor
    custom_objects = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": dict(cnn_output_dim=256, mlp_output_dim=64),
    }
 
    # Creamos un env dummy solo para que SAC.load no se queje
    dummy_env = DummyVecEnv([make_eval_env])
 
    if args.vecnorm and os.path.exists(args.vecnorm):
        dummy_env = VecNormalize.load(args.vecnorm, dummy_env)
        dummy_env.training = False
        dummy_env.norm_reward = False
        print(f"  VecNormalize loaded from: {args.vecnorm}")
 
    model = SAC.load(
        args.model,
        env=dummy_env,
        custom_objects=custom_objects,
        device=device
    )
    model.policy.set_training_mode(False)
    print("  Model loaded OK\n")
 
    # ── Evaluar episodios ─────────────────────────────────
    rewards = []
 
    for ep in range(args.n_episodes):
        print(f"── Episode {ep} ──────────────────────────────")
        env = make_eval_env()
        r = run_episode(
            model, env,
            episode_idx=ep,
            out_dir=args.out_dir,
            saliency_every=args.saliency_every,
            device=device
        )
        rewards.append(r)
        env.close()
 
    # ── Resumen ───────────────────────────────────────────
    print("\n══════════════════════════════════")
    print(f"  Episodes     : {args.n_episodes}")
    print(f"  Mean reward  : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Best episode : {np.argmax(rewards)}  ({max(rewards):.2f})")
    print(f"  Worst episode: {np.argmin(rewards)}  ({min(rewards):.2f})")
    print("══════════════════════════════════\n")
 
    save_reward_plot(rewards, args.out_dir)
 
    print(f"All results saved in: {args.out_dir}")
 
 
if __name__ == "__main__":
    main()
 