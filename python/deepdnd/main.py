"""
Copyright 2019 Xusen Yin
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

from deepdnd.hparams import load_hparams_for_evaluation
from deepdnd.hparams import load_hparams_for_training
from deepdnd.utils import setup_logging


parser = argparse.ArgumentParser(argument_default=None)
parser.add_argument('-m', '--model_dir', type=str)
parser.add_argument('-d', '--data_dir', type=str)
parser.add_argument('-c', '--config_file', type=str)
parser.add_argument('--game_dir', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--tgt_vocab_file', type=str)
parser.add_argument('--action_file', type=str)
parser.add_argument('--mode', default='training', help='[training|evaluation]')
parser.add_argument('--eval_episode', type=int)
parser.add_argument('--init_eps', type=float)
parser.add_argument('--final_eps', type=float)
parser.add_argument('--annealing_eps_t', type=int)
parser.add_argument('--init_gamma', type=float)
parser.add_argument('--final_gamma', type=float)
parser.add_argument('--annealing_gamma_t', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--save_gap_t', type=int)
parser.add_argument('--replay_mem', type=int)
parser.add_argument('--observation_t', type=int)
parser.add_argument('--lstm_num_units', type=int)
parser.add_argument('--lstm_num_layers', type=int)
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--total_t', default=sys.maxsize, type=int)
parser.add_argument('--game_episode_terminal_t', type=int)
parser.add_argument('--model_creator', type=str)
parser.add_argument('--tjs_creator', type=str)
parser.add_argument('--game_clazz', type=str)
parser.add_argument('--delay_target_network', type=int)
parser.add_argument('--max_action_len', type=int)
parser.add_argument('--max_games_used', type=int)
parser.add_argument('--eval_randomness', type=float)


if __name__ == '__main__':
    args = parser.parse_args()
    config_file = args.config_file
    game_path = args.data_dir
    model_dir = args.model_dir

    if args.mode == "evaluation":
        from deepdnd.agent import Agent
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging-eval.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=None)
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        agent = Agent(hp, game_path, model_dir)
        trained_step, eval_res, _ = agent.evaluate()
        print("after train step {}, evaluation results: {}".format(
            trained_step, eval_res))
    elif args.mode == "training":
        from deepdnd import agent
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=os.path.join(model_dir, 'game_script.log'))
        hp = load_hparams_for_training(config_file, args)
        agent_clazz = getattr(agent, hp.agent_clazz)
        agent = agent_clazz(hp, game_path, model_dir)
        agent.train()
    elif args.mode == "human":
        from deepdnd.agent import Agent
        current_dir = os.path.dirname(os.path.realpath(__file__))
        log_config_file = '{}/../../conf/logging-eval.yaml'.format(current_dir)
        setup_logging(
            default_path=log_config_file,
            local_log_filename=None)
        pre_config_file = os.path.join(model_dir, 'hparams.json')
        hp = load_hparams_for_evaluation(pre_config_file, args)
        agent = Agent(hp, game_path, model_dir)
        scores, steps, train_step = agent.human_check()
        print("after train step {}, evaluation (scores/steps): ({}/{})".format(
            train_step, scores, steps))
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))
