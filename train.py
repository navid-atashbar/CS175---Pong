import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from src import DQNAgent, PongStateEncoder

EPISODES = 5
WEIGHTS_PATH = "weights/dqn_pong.pt"


def main() -> None:
    env = gym.make("ALE/Pong-v5")
    encoder = PongStateEncoder()
    agent = DQNAgent(state_dim=encoder.state_dim)

    for ep in range(EPISODES):
        print(f"Episode {ep + 1}/{EPISODES} started.")

        encoder.reset()
        env.reset()
        state = encoder.encode_from_env(env)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, training=True)
            _, reward, terminated, truncated, _ = env.step(agent.to_env_action(action))
            done = terminated or truncated
            next_state = state if done else encoder.encode_from_env(env)

            agent.remember(state, action, reward, next_state, done)

            # train less often than every step to speed up training
            if steps % 4 == 0:
                agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

            if steps % 1000 == 0:
                print(f"Episode {ep + 1}: {steps} steps taken, total reward so far: {total_reward}")    

        agent.decay_epsilon()
        print(f"Episode {ep + 1} finished | steps: {steps} | total reward: {total_reward}")

    agent.save(WEIGHTS_PATH)
    env.close()
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()

