from parking_env_image import ParkingImage

env = ParkingImage(render_mode="human")
obs, info_ = env.reset(seed=15)
print(obs)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    if terminated or truncated:
        observation, info = env.reset()
