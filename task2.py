import gymnasium as gym
from gymnasium.spaces import Discrete

'''Task 2.1 '''
env = gym.make(
    "FrozenLake-v1",
    is_slippery=True,
    render_mode="rgb_array"
)

assert isinstance(env.observation_space, Discrete)
assert isinstance(env.action_space, Discrete)

num_states = env.observation_space.n
num_actions = env.action_space.n

print("Número de Estados:", num_states)
print("Número de Acciones:", num_actions)