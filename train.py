import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from src import DQNAgent, PongStateEncoder

EPISODES = 5000
WEIGHTS_PATH = "weights/dqn_pong.pt"


def main() -> None:
    env = gym.make("ALE/Pong-v5")
    encoder = PongStateEncoder()
    agent = DQNAgent(state_dim=encoder.state_dim)

    for _ in range(EPISODES):
        encoder.reset()
        env.reset()
        state = encoder.encode_from_env(env)
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            _, reward, terminated, truncated, _ = env.step(agent.to_env_action(action))
            done = terminated or truncated
            next_state = state if done else encoder.encode_from_env(env)

            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

        agent.decay_epsilon()

    agent.save(WEIGHTS_PATH)
    env.close()
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()

