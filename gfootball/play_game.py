# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Script allowing to play the game by multiple players."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import threading
import time

from gfootball.env import config
from gfootball.env import football_env

try:
  from pynput import keyboard
  HAS_PYNPUT = True
except ImportError:
  HAS_PYNPUT = False

FLAGS = flags.FLAGS

flags.DEFINE_string('players', 'keyboard:left_players=1',
                    'Semicolon separated list of players, single keyboard '
                    'player on the left by default. '
                    'E.g. keyboard:left_players=2;bot:right_players=3')
flags.DEFINE_string('level', '', 'Level to play')
flags.DEFINE_enum('action_set', 'default', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
                  'If true, environment will slow down so humans can play.')
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')
flags.DEFINE_bool('multi_agent_mode', False,
                  'If true, use TAB to switch between keyboard players (requires pynput)')


def main(_):
  if FLAGS.multi_agent_mode and not HAS_PYNPUT:
    logging.error('multi_agent_mode requires pynput. Install it with: pip install pynput')
    exit(1)
  
  players = FLAGS.players.split(';') if FLAGS.players else ''
  assert not (any(['agent' in player for player in players])
             ), ('Player type \'agent\' can not be used with play_game.')
  
  # Count keyboard players
  keyboard_players = sum(1 for p in players if 'keyboard' in p.split(':')[0])
  
  cfg_values = {
      'action_set': FLAGS.action_set,
      'dump_full_episodes': True,
      'players': players,
      'real_time': FLAGS.real_time,
  }
  if FLAGS.level:
    cfg_values['level'] = FLAGS.level
  cfg = config.Config(cfg_values)
  env = football_env.FootballEnv(cfg)
  if FLAGS.render:
    env.render()
  env.reset()
  
  current_player = [0]  # Use list to modify in nested function
  switch_player_requested = [False]
  
  if FLAGS.multi_agent_mode and keyboard_players > 1:
    logging.warning(f'Multi-agent mode: {keyboard_players} keyboard players')
    logging.warning('Press TAB to switch between keyboard players')
    logging.warning('Press Q to quit')
    
    def on_press(key):
      try:
        if key == keyboard.Key.tab:
          switch_player_requested[0] = True
        elif key == keyboard.Key.esc or key.char == 'q':
          logging.warning('Exiting...')
          exit(0)
      except AttributeError:
        pass
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
  
  try:
    while True:
      if FLAGS.multi_agent_mode and keyboard_players > 1 and switch_player_requested[0]:
        current_player[0] = (current_player[0] + 1) % keyboard_players
        switch_player_requested[0] = False
        logging.warning(f'Switched to keyboard player {current_player[0] + 1}')
      
      _, _, done, _ = env.step([])
      if done:
        env.reset()
        current_player[0] = 0
  except KeyboardInterrupt:
    logging.warning('Game stopped, writing dump...')
    env.write_dump('shutdown')
    exit(1)


if __name__ == '__main__':
  app.run(main)
