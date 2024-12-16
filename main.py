import cv2

from shipping import Environment

if __name__ == "__main__":
    env = Environment("mapa_mundi_binario.jpg")

    env.add_port([41, 40])
    env.add_port([60, 22])
    env.add_port([78, 29])
    env.add_port([49, 72])
    env.add_port([62, 72])

    env.add_storm([[20, 30], [30, 40]])
    env.add_storm([[55, 65], [10, 20]])
    env.add_storm([[80, 90], [60, 70]])
    env.add_storm([[10, 20], [70, 80]])
    env.add_storm([[40, 50], [85, 95]])

    state = env.reset()

    try:
        for _ in range(1000):
            action = env.sample_action()
            state, reward, done, meta = env.step(action)
            env.update_storms()
            env.render_real_time()
            if done: break
    except KeyboardInterrupt:
        cv2.destroyAllWindows()