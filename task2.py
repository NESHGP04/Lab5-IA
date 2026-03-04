import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import time

# ==========================================================
# TASK 2.1 – Inicialización del entorno
# ==========================================================

# Para entrenamiento NO usar render_mode="human"
# porque hace el entrenamiento extremadamente lento.
env = gym.make(
    "FrozenLake-v1",
    is_slippery=True
)

# Verificación de espacios discretos
assert isinstance(env.observation_space, Discrete)
assert isinstance(env.action_space, Discrete)

num_states = env.observation_space.n
num_actions = env.action_space.n

print("Número de Estados:", num_states)
print("Número de Acciones:", num_actions)


# ==========================================================
# TASK 2.2 – Q-Learning
# ==========================================================

# Inicialización de la tabla Q (16 estados x 4 acciones)
Q = np.zeros((num_states, num_actions))

# ---------------------------
# Hiperparámetros
# ---------------------------
alpha = 0.1          # Learning Rate
gamma = 0.99         # Factor de descuento
epsilon = 1.0        # Exploración inicial
epsilon_min = 0.01   # Exploración mínima
epsilon_decay = 0.9995
episodes = 10000

# ---------------------------
# Entrenamiento
# ---------------------------
for episode in range(episodes):
    
    state, _ = env.reset()
    done = False
    
    while not done:
        
        # Política Epsilon-Greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(Q[state])        # Explotación
        
        # Ejecutar acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        reward = float(reward)  # Conversión explícita
        
        # Fórmula de Q-Learning
        best_next_value = np.max(Q[next_state])
        
        Q[state, action] += alpha * (
            reward + gamma * best_next_value - Q[state, action]
        )
        
        state = next_state
    
    # Decaimiento de epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Entrenamiento finalizado.")


# ==========================================================
# TASK 2.3 – Evaluación
# ==========================================================

test_episodes = 10
wins = 0

for episode in range(test_episodes):
    
    state, _ = env.reset()
    done = False
    
    trajectory = [state]
    
    while not done:
        
        # Política puramente greedy (ε = 0)
        action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        trajectory.append(next_state)
        state = next_state
    
    if reward == 1:
        wins += 1
    
    print(f"Episodio {episode+1}:")
    print("Trayectoria:", trajectory)
    print("Recompensa final:", reward)
    print("---------------------------")

win_rate = wins / test_episodes
print("Win Rate:", win_rate)


# ==========================================================
# Visualizar un episodio exitoso
# ==========================================================

env_render = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    render_mode="human"
)

success = False

while not success:
    state, _ = env_render.reset()
    done = False
    
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        state = next_state
        time.sleep(0.3)
    
    if reward == 1:
        success = True
        print("Episodio exitoso visualizado.")