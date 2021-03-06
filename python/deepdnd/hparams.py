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

import json
import logging
import os
import sys
from shutil import copyfile

import tensorflow as tf


def find_default_model_hparams(model_creator=''):
    if model_creator == 'LSTMEncoderDQN':
        model_hparams = default_hparams_LSTMEncoderDQN()
    elif model_creator == 'LSTMEncoderDecoderDQN':
        model_hparams = default_hparams_LSTMEncoderDecoderDQN()
    elif model_creator == 'CNNEncoderDQN':
        model_hparams = default_hparams_CNNEncoderDQN()
    elif model_creator == 'MultiChannelCNNEncoderDQN':
        model_hparams = default_hparams_MultiChannelCNNEncoderDQN()
    elif model_creator == 'CNNEncoderDecoderDQN':
        model_hparams = default_hparams_CNNEncoderDecoderDQN()
    elif model_creator == 'CNNEncoderDRRN':
        model_hparams = default_hparams_CNNEncoderDRRN()
    elif model_creator == 'CNNEDMultiLayerDQN':
        model_hparams = default_hparams_CNNEDMultiLayerDQN()
    elif model_creator == 'CNNEncoderMultiLayerDQN':
        model_hparams = default_hparams_CNNEncoderMultiLayerDQN()
    else:
        raise ValueError('unknown model creator: {}'.format(model_creator))
    return model_hparams


def default_hparams_agent():
    return tf.contrib.training.HParams(
        model_dir='',
        data_dir='',
        vocab_file='',
        tgt_vocab_file='',
        action_file='',
        eval_episode=0,
        init_eps=1.,
        final_eps=1e-4,
        annealing_eps_t=5000,
        init_gamma=0.,
        final_gamma=0.7,
        annealing_gamma_t=5000,
        replay_mem=100000,
        observation_t=2000,
        total_t=sys.maxsize,
        game_episode_terminal_t=5000,
        vocab_size=0,
        tgt_vocab_size=0,
        n_actions=512,
        n_tokens_per_action=10,
        sos='<S>',
        eos='</S>',
        padding_val='<PAD>',
        unk_val='<UNK>',
        sos_id=0,
        eos_id=0,
        padding_val_id=0,
        unk_val_id=0,
        tgt_sos_id=0,
        tgt_eos_id=1,
        model_creator='',
        max_snapshot_to_keep=5,
        game_clazz='ZGameZork',
        delay_target_network=1
    )


def default_hparams_LSTMEncoderDQN():
    return tf.contrib.training.HParams(
        agent_clazz='Agent',
        tjs_creator='VarSizeTrajectory',
        batch_size=32,
        save_gap_t=1000,
        lstm_num_units=32,
        lstm_num_layers=3,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
    )


def default_hparams_LSTMEncoderDecoderDQN():
    return tf.contrib.training.HParams(
        agent_clazz='GenAgentFromAction',
        tjs_creator='VarSizeTrajectory',
        batch_size=32,
        save_gap_t=1000,
        lstm_num_units=32,
        lstm_num_layers=3,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
        max_action_len=6
    )


def default_hparams_CNNEncoderDQN():
    return tf.contrib.training.HParams(
        agent_clazz='Agent',
        tjs_creator='SingleChannelTrajectory',
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
        num_conv_filters=32
    )


def default_hparams_CNNEncoderDecoderDQN():
    return tf.contrib.training.HParams(
        agent_clazz='GenAgentFromAction',
        tjs_creator='SingleChannelTrajectory',
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
        num_conv_filters=32,
        max_action_len=6
    )


def default_hparams_CNNEncoderMultiLayerDQN():
    return tf.contrib.training.HParams(
        agent_clazz='Agent',
        tjs_creator='SingleChannelTrajectory',
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
        max_action_len=6,
        num_layers=4
    )


def default_hparams_CNNEDMultiLayerDQN():
    return tf.contrib.training.HParams(
        agent_clazz='GenAgentFromAction',
        tjs_creator='SingleChannelTrajectory',
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
        max_action_len=6,
        num_layers=4
    )


def default_hparams_MultiChannelCNNEncoderDQN():
    return tf.contrib.training.HParams(
        agent_clazz='Agent',
        tjs_creator='MultiChannelTrajectory',
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=300,
        num_conv_filters=32
    )


def default_hparams_CNNEncoderDRRN():
    return tf.contrib.training.HParams(
        agent_clazz='Agent',
        tjs_creator='SingleChannelTrajectory',
        batch_size=32,
        save_gap_t=1000,
        embedding_size=64,
        learning_rate=1e-5,
        num_turns=21,
        num_tokens=1000,
        num_conv_filters=32
    )


def output_hparams(hp):
    out_str = ['------------hparams---------------']
    hp_dict = hp.values()
    keys = sorted(hp_dict.keys())
    for k in keys:
        out_str.append('{} -> {}'.format(k, hp_dict[k]))
    out_str.append('-----------------------------------')
    return "\n".join(out_str)


def update_hparams_from_cmd(hp, cmd_args):
    dict_cmd_args = vars(cmd_args)
    for hp_key in dict_cmd_args:
        if hp_key in hp and dict_cmd_args[hp_key] is not None:
            hp.set_hparam(hp_key, dict_cmd_args[hp_key])
    return hp


def update_hparams_from_hparams(hp, hp2):
    """hp should not have same keys with hp2"""
    dict_hp2 = hp2.values()
    for k in dict_hp2:
        hp.add_hparam(k, dict_hp2.get(k))
    return hp


def update_hparams_from_file(hp, file_args):
    with open(file_args, 'r') as f:
        json_val = json.load(f)
        for k in json_val:
            if k in hp:
                hp.set_hparam(k, json_val.get(k))
            else:
                pass
    return hp


def copy_hparams(hp):
    hp2 = tf.contrib.training.HParams()
    dict_hp = hp.values()
    for k in dict_hp:
        hp2.add_hparam(k, dict_hp.get(k))
    return hp2


def load_hparams_for_training(file_args=None, cmd_args=None):
    """
    load hparams for training.
    priority(cmd_args) > priority(file_args)
    """
    hp = default_hparams_agent()
    model_hp = find_default_model_hparams(cmd_args.model_creator)
    hp = update_hparams_from_hparams(hp, model_hp)
    if file_args is not None:
        hp = update_hparams_from_file(hp, file_args)
    if cmd_args is not None:
        hp = update_hparams_from_cmd(hp, cmd_args)
    create_dependency(hp)
    return hp


def create_dependency(hp):
    logger = logging.getLogger('hparams')
    deps_to_change = ['data_dir', 'vocab_file', 'tgt_vocab_file', 'action_file']
    if not os.path.exists(hp.model_dir):
        raise ValueError(
            "bad path: {}, please create before using.".format(hp.model_dir))
    deps_dir = os.path.join(hp.model_dir, 'deps')
    if os.path.exists(deps_dir):
        if not os.path.isdir(deps_dir):
            raise ValueError('bad path: {}, it is not a dir.'.format(deps_dir))
        else:
            pass
    else:
        logger.info('create dependecy dir: {}'.format(deps_dir))
        os.mkdir(deps_dir)
    for hp_key in deps_to_change:
        try:
            new_file_path = os.path.join(
                deps_dir, os.path.basename(hp.get(hp_key)))
            copyfile(hp.get(hp_key), new_file_path)
            logger.info('copy dependency file {} to {}'.format(
                hp.get(hp_key), new_file_path))
            hp.set_hparam(hp_key, new_file_path)
        except Exception as e:
            logger.warning("copy dependency file {} -> {} failed: {}".format(
                hp_key, hp.get(hp_key), e))


def load_hparams_for_evaluation(pre_config_file, cmd_args=None):
    """
    load hparams for evaluation.
    priority(file_args) > priority(cmd_args)
     unless arg in allowed_to_change set.
    """
    allowed_to_change = ['model_dir', 'eval_episode', 'game_episode_terminal_t']
    deps_to_change = ['data_dir', 'vocab_file', 'tgt_vocab_file', 'action_file']
    hp = default_hparams_agent()
    # first load hp from file for choosing model_hp
    # notice that only hparams in hp can be updated.
    hp = update_hparams_from_file(hp, pre_config_file)
    model_hp = find_default_model_hparams(hp.model_creator)
    hp = update_hparams_from_hparams(hp, model_hp)
    # second load hp from file to change params back
    hp = update_hparams_from_file(hp, pre_config_file)

    if cmd_args is not None:
        dict_cmd_args = vars(cmd_args)
        for hp_key in dict_cmd_args:
            if (dict_cmd_args[hp_key] is not None
                    and hp_key in hp and hp_key in allowed_to_change):
                hp.set_hparam(hp_key, dict_cmd_args[hp_key])
            else:
                pass

    for hp_key in deps_to_change:
        hp.set_hparam(hp_key, os.path.join(hp.model_dir, hp.get(hp_key)))
    return hp


def save_hparams(hp, file_path, use_relative_path=False):
    logger = logging.getLogger('hparams')
    deps_to_change = ['data_dir', 'vocab_file', 'tgt_vocab_file', 'action_file']
    with open(file_path, 'w') as f:
        if not use_relative_path:
            f.write(hp.to_json())
        else:
            new_hp = copy_hparams(hp)
            for hp_key in deps_to_change:
                try:
                    new_path = os.path.relpath(hp.get(hp_key), hp.model_dir)
                    new_hp.set_hparam(hp_key, new_path)
                except Exception as e:
                    logger.warning(
                        "get relative path for {} -> {} failed: {}".format(
                            hp_key, hp.get(hp_key), e))
            f.write(new_hp.to_json())
