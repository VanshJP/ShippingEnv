import cv2
import sarsa
from shipping import Environment

if __name__ == "__main__":
    env = Environment("mapa_mundi_binario.jpg")

    env.add_port([41, 40])
    env.add_port([60, 22])
    env.add_port([78, 29])
    env.add_port([49, 72])
    env.add_port([62, 72])

    state = env.reset()

    try:
        '''
        for _ in range(1000):
            action = env.sample_action()
            state, reward, done, meta = env.step(action)
            env.render_real_time()
            if done: break
        '''
        sarsa.train_agent(env)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()