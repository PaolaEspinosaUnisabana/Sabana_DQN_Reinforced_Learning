"""Evalúa el agente entrenado (sin exploración). Ver con --render."""
import argparse

import gymnasium as gym
import numpy as np
import torch

from train import ENV_ID, QNetwork # Traemos la clase QNetwork del archivo train.py para poder usarla aquí y cargar el modelo entrenado.

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--render", action="store_true", help="mostrar la ventana")
args = parser.parse_args()

env = gym.make(ENV_ID, render_mode="human" if args.render else None)
q_net = QNetwork(env.observation_space.shape[0], env.action_space.n) # Enviamos la cantidad de observaciones y acciones al constructor de QNetwork para crear la red con la misma arquitectura que la entrenada.
q_net.load_state_dict(torch.load("sabana_dqn_rl.pth")) # Cargamos los pesos del modelo entrenado desde el archivo sabana_dqn_rl.pth. Esto nos permite usar la red entrenada para evaluar el agente sin necesidad de volver a entrenarlo.
q_net.eval()

rewards = [] # Iniciamos una lista para almacenar las recompensas obtenidas en cada episodio de evaluación.
for ep in range(args.episodes):
    state, _ = env.reset(seed=1000 + ep)
    total, done = 0.0, False
    while not done:
        with torch.no_grad():
            action = q_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action) # Avanzamos un paso en el entorno con la acción elegida por la red Q entrenada.
        total += reward
        done = terminated or truncated
    rewards.append(total)
    print(f"Episodio {ep+1}: {total:.0f}")

print(f"\nMedia: {np.mean(rewards):.1f} ± {np.std(rewards):.1f} "
      f"(min {np.min(rewards):.0f}, max {np.max(rewards):.0f})")
