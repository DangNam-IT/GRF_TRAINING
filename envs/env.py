import gfootball.env as football_env

def create_football_env(scenario_name="academy_corner"):
    env = football_env.create_environment(
        env_name=scenario_name,
        number_of_left_players_agent_controls=11,
        stacked=False,
        representation='raw',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=True
    )
    return env