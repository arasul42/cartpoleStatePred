from stable_baselines3.common.callbacks import BaseCallback
import cv2

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=500, window_name="Training Render", verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.window_name = window_name

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            if hasattr(self.training_env, "envs"):
                env = self.training_env.envs[0]
            else:
                env = self.training_env
            if hasattr(env, "render"):
                frame = env.render()
                if frame is not None:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(self.window_name, bgr)
                    cv2.waitKey(1)
        return True
