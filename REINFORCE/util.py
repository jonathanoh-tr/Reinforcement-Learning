
import numpy as np

def render_text_envq(env, agent):
    env.seed(np.random.randint(0, 10000))
    state = env.reset()

    while True:
        env.render()
        max_action = agent.act(state)
        state, reward, done, info = env.step(max_action[0])
        if (done):
            print("Environment Terminated")
            break
