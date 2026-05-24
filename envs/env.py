from __future__ import annotations

import gfootball.env as football_env
from gym import Env


def create_football_env(scenario_name: str = "academy_corner") -> Env:
    """
    Tạo môi trường GRF chuẩn cho training.

    Args:
        scenario_name: Tên kịch bản GRF (mặc định: 'academy_corner').

    Returns:
        env: Môi trường gym với raw observation.
    """
    env: Env = football_env.create_environment(
        env_name=scenario_name,
        number_of_left_players_agent_controls=11,
        stacked=False,
        representation="raw",
        rewards="scoring",
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=True,
    )
    return env