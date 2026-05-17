from typing import Optional, Union

import numpy as np


class PongStateEncoder:
    """
    Features (normalized to [-1, 1]):
      - player paddle y
      - opponent paddle y
      - ball x, ball y
      - ball x velocity, ball y velocity (finite differences)
    """

    # ALE RAM indices used for Pong feature agents.
    _BALL_X = 49
    _OPP_PADDLE_Y = 50
    _PLAYER_PADDLE_Y = 51
    _BALL_Y = 54

    def __init__(self) -> None:
        self._prev_ball: Optional[np.ndarray] = None

    @property
    def state_dim(self) -> int:
        return 6

    def reset(self) -> None:
        self._prev_ball = None

    def encode(self, ram: Union[np.ndarray, list, memoryview]) -> np.ndarray:
        ram = np.asarray(ram, dtype=np.float32)
        ball = np.array([ram[self._BALL_X], ram[self._BALL_Y]], dtype=np.float32)

        if self._prev_ball is None:
            ball_vel = np.zeros(2, dtype=np.float32)
        else:
            ball_vel = ball - self._prev_ball
        self._prev_ball = ball.copy()

        state = np.array(
            [
                ram[self._PLAYER_PADDLE_Y] / 116.0,
                ram[self._OPP_PADDLE_Y] / 116.0,
                ball[0] / 160.0,
                ball[1] / 194.0,
                np.clip(ball_vel[0] / 20.0, -1.0, 1.0),
                np.clip(ball_vel[1] / 20.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return state

    def encode_from_env(self, env) -> np.ndarray:
        """Read RAM from a Gymnasium ALE Pong environment."""
        ram = env.unwrapped.ale.getRAM()
        return self.encode(ram)
