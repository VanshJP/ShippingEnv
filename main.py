import cv2

from shipping import Environment

if __name__ == "__main__":
    env = Environment()
    state = env.reset()

    try:
        for _ in range(1000):
            state, reward, done = env.step()
            env.update_storms()
            env.render_real_time()
            if done:
                print(f"Completed the journey with reward: {reward}")
                state = env.reset()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()