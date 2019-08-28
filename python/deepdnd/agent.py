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

import collections
import os
from threading import Thread, Condition

import numpy as np
import tensorflow as tf
from scipy.special import logsumexp

from deepdnd import dqn_model
from deepdnd import trajectory
from deepdnd.dqn_func import get_best_1Daction, get_random_1Daction
from deepdnd.game import game
from deepdnd.hparams import save_hparams, output_hparams
from deepdnd.log import Logging
from deepdnd.tree_memory import TreeMemory
from deepdnd.utils import load_uniq_lines, get_token2idx, ctime


class EvalResults(
    collections.namedtuple(
        "EvalResults",
        ("total_score", "total_step", "max_score", "max_score_step",
         "min_score", "min_score_step"))):

    def __repr__(self):
        return "EvalResult: {}, {}, {}, {}, {}, {}".format(
            self.total_score, self.total_step,
            self.max_score, self.max_score_step,
            self.min_score, self.min_score_step)


class Agent(Logging):
    def __init__(self, hp, game_path, model_dir):
        super(Agent, self).__init__()

        self.padding_val = 'O'
        self.unk_val = '<unk>'

        self.hp = hp
        self.actions = self.load_actions(self.hp.action_file)
        self.action2idx = get_token2idx(self.actions)
        self.game_path = game_path
        game_clazz = getattr(game, self.hp.game_clazz)
        self.info("run game class: {}".format(game_clazz))
        self.game = game_clazz(
            self.game_path, max_step=self.hp.game_episode_terminal_t)
        self.tokens = (
                [self.hp.sos, self.hp.eos, self.padding_val, self.unk_val] +
                self.load_vocab(hp.vocab_file))
        self.token2idx = get_token2idx(self.tokens)
        self.tgt_tokens = ([self.hp.sos, self.hp.eos] +
                           self.load_vocab(hp.tgt_vocab_file))
        self.tgt_token2idx = get_token2idx(self.tgt_tokens)
        self.hp.set_hparam('vocab_size', len(self.tokens))
        self.hp.set_hparam('n_actions', len(self.actions))
        self.hp.set_hparam('tgt_vocab_size', len(self.tgt_tokens))
        self.hp.set_hparam('tgt_sos_id', self.tgt_token2idx[self.hp.sos])
        self.hp.set_hparam('tgt_eos_id', self.tgt_token2idx[self.hp.eos])
        self.hp.set_hparam(
            'total_t',
            min(self.hp.annealing_eps_t + self.hp.observation_t, hp.total_t))
        self.model_dir = model_dir
        self.tjs_creator = getattr(trajectory, hp.tjs_creator)
        self.tjs = self.tjs_creator(
            self.hp,
            padding_val=self.token2idx[self.padding_val])
        self.tjs_path = os.path.join(self.model_dir, 'trajectories.npz')
        self.visited_path = os.path.join(self.model_dir, 'visited_rooms.npz')
        try:
            self.tjs.load_tjs(self.tjs_path)
        except Exception as e:
            self.info("load trajectory failed: \n{}".format(e))
        self.tjs.add_new_tj()
        self.memo = TreeMemory(capacity=self.hp.replay_mem)
        self.memo_path = os.path.join(self.model_dir, 'memo.npz')
        try:
            self.memo.load_memo(self.memo_path)
        except Exception as e:
            self.info("load memo failed: \n{}".format(e))
        try:
            visited = np.load(self.visited_path)
            self.visited_rooms = dict(zip(
                list(visited['visited_rooms']),
                list(visited['visited_times'])))
            pass
        except Exception as e:
            self.visited_rooms = None
            self.info("load visited rooms failed: \n{}".format(e))
        self.model = None
        self.target_model = None
        self.model_creator = getattr(dqn_model, hp.model_creator)
        self.info(output_hparams(self.hp))
        save_hparams(self.hp,
                     os.path.join(self.model_dir, 'hparams.json'),
                     use_relative_path=True)
        self.prev_player_t = None
        self.prev_master_t = None
        self.prev_cumulative_penalty = 0
        self.create_train_model = dqn_model.create_train_model
        self.create_eval_model = dqn_model.create_eval_model

    @classmethod
    def report_status(cls, lst_of_status):
        return ', '.join(
            map(lambda k_v: '{}: {}'.format(k_v[0], k_v[1]), lst_of_status))

    @classmethod
    def reverse_annealing_gamma(cls, init_gamma, final_gamma, t, total_t):
        gamma_t = init_gamma + ((final_gamma - init_gamma) * 1. / total_t) * t
        return min(gamma_t, final_gamma)

    @classmethod
    def annealing_eps(cls, init_eps, final_eps, t, total_t):
        eps_t = init_eps - ((init_eps - final_eps) * 1. / total_t) * t
        return max(eps_t, final_eps)

    @classmethod
    def load_vocab(cls, vocab_file):
        return list(map(lambda t: t.lower(), load_uniq_lines(vocab_file)))

    @classmethod
    def load_actions(cls, action_file):
        return load_uniq_lines(action_file)

    def index_string(self, sentence):
        return list(map(lambda t:
                        self.token2idx.get(t, self.token2idx.get(self.unk_val)),
                        sentence))

    @classmethod
    def q_vec_entropy(cls, q_vec):
        normalized_q_vec = q_vec - logsumexp(q_vec)
        normalized_p_q_vec = np.exp(normalized_q_vec)
        ent = np.sum(normalized_q_vec * normalized_p_q_vec) * -1.0
        return ent

    def _get_an_eps_action(self, sess, eps, tjs, model, cnt_action):
        """
        get either an random action index with action string
        or the best predicted action index with action string.
        :param sess: tf.session
        :param eps: probability of choosing random actions
        :param tjs: trajectories of playing history
        :param model: encoder
        :return: index of action (int), string of action (string)
        """
        reports = []
        if np.random.random() < eps:
            action_idx, player_t = get_random_1Daction(self.actions)
            reports += [('random_action', action_idx),
                        ('action', player_t)]
        else:
            indexed_state_t, lens_t = tjs.fetch_last_state()
            readout_t = sess.run(model.q_actions, feed_dict={
                model.src_: [indexed_state_t],
                model.src_len_: [lens_t]})
            readout_t = readout_t[0]
            action_idx, q_max, player_t = get_best_1Daction(
                readout_t - cnt_action, self.actions)
            reports += [('action', player_t), ('q_max', q_max),
                        ('q_argmax', action_idx),
                        ('entropy', self.q_vec_entropy(readout_t))]
            cnt_action[action_idx] += 0.1
        return action_idx, player_t, reports, cnt_action

    def take_an_eps_step(
            self, t, sess, eps, tjs, model, game_instance,
            cnt_action, play_through_action=None, is_training=True):
        reports = (
            [('t', t), ('in_game_t', game_instance.total_step), ('eps', eps)])

        if play_through_action is not None:
            action_idx = self.action2idx[play_through_action]
            player_t = play_through_action
            report_action = [("play_through", action_idx), ("action", player_t)]
            cnt_action = cnt_action
        else:
            (action_idx, player_t, report_action,
             cnt_action) = self._get_an_eps_action(
                sess, eps, tjs, model, cnt_action)
        reports += report_action
        master, room_name, reward = game_instance.take_action(player_t)

        reports += (
            [('master', master), ('reward', reward),
             ('total_reward', game_instance.total_reward),
             ('total_shaped_reward', game_instance.total_shaped_reward)])
        # self.debug(self.report_status(reports))
        # TODO: the reward in the reports could be changed later
        return (player_t.split(), master.split(), room_name,
                action_idx, reward, cnt_action, reports)

    @classmethod
    def count_trainable(cls, trainable_vars, mask=None):
        total_parameters = 0
        if mask is not None:
            if type(mask) is list:
                trainable_vars = filter(lambda v: v.op.name not in mask,
                                        trainable_vars)
            elif type(mask) is str:
                trainable_vars = filter(lambda v: v.op.name != mask,
                                        trainable_vars)
            else:
                pass
        else:
            pass
        for variable in trainable_vars:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def create_n_load_model(self):
        start_t = 0
        with tf.device('/device:GPU:0'):
            model = self.create_train_model(self.model_creator, self.hp)
        train_conf = tf.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True)
        train_sess = tf.Session(graph=model.graph, config=train_conf)
        with model.graph.as_default():
            train_sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
            restore_from = tf.train.latest_checkpoint(
                os.path.join(self.model_dir, 'last_weights'))
            if restore_from is not None:
                # Reload weights from directory if specified
                self.info("Try to restore parameters")
                saver.restore(train_sess, restore_from)
                global_step = tf.train.get_or_create_global_step()
                trained_steps = train_sess.run(global_step)
                start_t = trained_steps + self.hp.observation_t
            else:
                self.info('No checkpoint to load, training from scratch')
            trainable_vars = tf.trainable_variables()
            self.info('trainable variables: {}'.format(trainable_vars))
            self.info('count of trainable vars w/o src_embeddings: {}'.format(
                self.count_trainable(trainable_vars, mask='src_embeddings')))
        return train_sess, start_t, saver, model

    def best_trajectory(self):
        best_score = np.max(
            list(map(lambda x: x[-1], list(self.memo.tree.data))))
        self.debug("best score so far: {}".format(best_score))
        tids = []
        sids = []
        for mem in self.memo.tree.data:
            if mem[-1] == best_score:
                tids.append(mem[0])
                sids.append(mem[1])
            else:
                pass
        if len(tids) == 0:
            return []
        sampled_id = np.random.choice(range(len(tids)))
        s_tid = tids[sampled_id]
        s_sid = sids[sampled_id]
        if s_tid == self.tjs.get_current_tid():
            best_tj = self.tjs.curr_tj
        else:
            best_tj = self.tjs.trajectories[s_tid]
        # add a look action in the end to request observation for training
        action_list = list(map(
            lambda token_ids: " ".join(
                map(lambda ti: self.tokens[ti], token_ids)),
            best_tj[1:s_sid:2]))
        if np.random.random() < 0.5:
            action_list = action_list[:len(action_list) // 2 + 1]
        self.debug("sampled replay trajectory actions: {}".format(action_list))
        return action_list

    def train(self):
        cond_of_eval = Condition()
        eval_worker = Thread(name='eval_worker',
                             target=self.eval_nn, args=(cond_of_eval,))
        eval_worker.daemon = True
        eval_worker.start()

        chkp_prefix = os.path.join(self.model_dir,
                                   'last_weights', 'after-epoch')
        eps = self.hp.init_eps

        train_sess, start_t, saver, self.model = self.create_n_load_model()
        if start_t == 0:  # save model for loading of target net
            saver.save(
                train_sess, chkp_prefix,
                global_step=tf.train.get_or_create_global_step(
                    graph=self.model.graph))

        if self.hp.delay_target_network != 0:
            (target_sess, _, target_saver, self.target_model
             ) = self.create_n_load_model()
        else:
            self.target_model = self.model
            target_sess = train_sess
            target_saver = saver

        train_summary_dir = os.path.join(self.model_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     train_sess.graph)

        master, room_name = self.game.reset_game()
        state_t = master.split()
        indexed_state_t = self.index_string(state_t)
        # self.debug('master: {}'.format(' '.join(state_t)))

        self.tjs.append(indexed_state_t)

        epoch_start_t = ctime()

        # start training
        t = start_t
        cnt_action = np.zeros(self.hp.n_actions)
        # count visited room during whole training process
        self.debug("master room name: {}".format(room_name))
        assert room_name != "", "initial room name is empty"
        if self.visited_rooms is not None:
            visited_rooms = self.visited_rooms
            if room_name != "":
                if room_name not in visited_rooms:
                    visited_rooms[room_name] = 0
                visited_rooms[room_name] += 1
            else:
                pass
        else:
            visited_rooms = {room_name: 1}
        play_through_actions = []
        while t < self.hp.total_t:
            if self.game.terminal:
                play_through_actions = []
                cnt_action = np.zeros(self.hp.n_actions)
                self.prev_master_t = None
                self.prev_player_t = None
                self.tjs.add_new_tj()
                if t >= self.hp.observation_t and np.random.random() < 0.5:
                    try:
                        play_through_actions = self.best_trajectory()
                    except Exception as e:
                        self.debug(
                            "fetch best trajectory failed:\n{}".format(e))

                master, room_name = self.game.reset_game(
                    max_step=self.hp.game_episode_terminal_t + len(
                        play_through_actions))

                if room_name != "":
                    if room_name not in visited_rooms:
                        visited_rooms[room_name] = 0
                    visited_rooms[room_name] += 1
                else:
                    pass
                state_t = master.split()
                indexed_state_t = self.index_string(state_t)
                self.tjs.append(indexed_state_t)
                # self.debug(
                #     'game reset with master: {}'.format(' '.join(state_t)))
                # don't incur t here
                continue

            if len(play_through_actions) == 0:
                pt_action = None
            else:
                if len(play_through_actions) == 1:
                    self.debug("last play through action: {}".format(
                        play_through_actions[0]))
                pt_action = play_through_actions[0]
                play_through_actions = play_through_actions[1:]
            (p_str, m_str, room_name, action_idx, reward, cnt_action, reports
             ) = self.take_an_eps_step(
                t, train_sess, eps, self.tjs, self.model, self.game,
                cnt_action, play_through_action=pt_action, is_training=True)

            if (master == self.prev_master_t and
                    " ".join(p_str) == self.prev_player_t and reward < 0):
                reward = self.game.clip_reward(
                    reward + self.prev_cumulative_penalty)
                self.prev_cumulative_penalty -= 0.1
                self.debug("t: {}, repeated bad try, - {} -> {}".format(
                    t, self.prev_cumulative_penalty, reward))
            else:
                self.prev_player_t = " ".join(p_str)
                self.prev_master_t = master
                self.prev_cumulative_penalty = 0

            if room_name == "":
                discovery_bonus = 0
            else:
                if room_name not in visited_rooms:
                    visited_rooms[room_name] = 0
                visited_rooms[room_name] += 1
                discovery_bonus = 1.0 * visited_rooms[room_name] ** (-0.3)
            if discovery_bonus > 0.9:
                self.debug("visited room: {}".format(room_name))
                self.debug("bonus: {}".format(discovery_bonus))
                self.debug("new room: {}".format(
                    sorted(list(visited_rooms.items()), key=lambda x: x[1])))
            reward = self.game.clip_reward(reward + discovery_bonus)
            reports.append(
                ("training reward", reward))

            if reward > 0:
                self.debug(self.report_status(reports))

            self.tjs.append(self.index_string(p_str))
            self.tjs.append(self.index_string(m_str))

            # remove 10x entries of reward > 0.9
            self.memo.append(
                (self.tjs.get_current_tid(),
                 self.tjs.get_last_sid(),
                 action_idx,
                 reward,
                 self.game.terminal,
                 self.game.total_reward))

            if t >= self.hp.observation_t:
                eps = self.annealing_eps(
                    self.hp.init_eps, self.hp.final_eps,
                    t - self.hp.observation_t, self.hp.annealing_eps_t)
                if (t - self.hp.observation_t) % self.hp.save_gap_t == 0:
                    if not (t - self.hp.observation_t) == 0:
                        epoch_end_t = ctime()
                        delta_time = epoch_end_t - epoch_start_t
                        self.info('current epoch end')
                        reports_time = [
                            ('epoch time', delta_time),
                            ('#batches per epoch', self.hp.save_gap_t),
                            ('avg step time',
                             delta_time * 1.0 / self.hp.save_gap_t)]
                        self.info(self.report_status(reports_time))
                        saver.save(
                            train_sess, chkp_prefix,
                            global_step=tf.train.get_or_create_global_step(
                                graph=self.model.graph))
                        # restore target net from last saved train net
                        restore_from = tf.train.latest_checkpoint(
                            os.path.join(self.model_dir, 'last_weights'))
                        target_saver.restore(target_sess, restore_from)
                        self.info(
                            "target net load from: {}".format(restore_from))
                        self.memo.save_memo(self.memo_path)
                        self.tjs.save_tjs(self.tjs_path)
                        np.savez(
                            self.visited_path,
                            visited_rooms=list(visited_rooms.keys()),
                            visited_times=list(visited_rooms.values()))
                        self.info('save snapshot of agent')
                        with cond_of_eval:
                            cond_of_eval.notifyAll()
                    self.info('start next epoch with t: {}'.format(t))
                    epoch_start_t = ctime()

                self.train_impl(
                    train_sess, t, train_summary_writer, target_sess)
            # make a step forward
            t += 1

    def train_impl(self, sess, t, summary_writer, target_sess):
        gamma = self.reverse_annealing_gamma(
            self.hp.init_gamma, self.hp.final_gamma,
            t - self.hp.observation_t, self.hp.annealing_gamma_t)
        # self.debug('training on sample with gamma: {}'.format(gamma))

        t1 = ctime()
        b_idx, b_memory, b_weight = self.memo.sample_batch(self.hp.batch_size)
        trajectory_id = [m[0][0] for m in b_memory]
        state_id = [m[0][1] for m in b_memory]
        action_id = [m[0][2] for m in b_memory]
        reward = [m[0][3] for m in b_memory]
        is_terminal = [m[0][4] for m in b_memory]

        p_states, s_states, p_len, s_len = self.tjs.fetch_batch_states_pair(
            trajectory_id, state_id)
        t1_end = ctime()

        t2 = ctime()
        s_q_actions_target = target_sess.run(
            self.target_model.q_actions,
            feed_dict={self.target_model.src_: s_states,
                       self.target_model.src_len_: s_len})
        s_q_actions_dqn = sess.run(
            self.model.q_actions,
            feed_dict={self.model.src_: s_states,
                       self.model.src_len_: s_len}
        )
        t2_end = ctime()

        s_argmax_q = np.argmax(s_q_actions_dqn, axis=1)
        expected_q = np.zeros_like(reward)
        for i in range(len(expected_q)):
            expected_q[i] = reward[i]
            if not is_terminal[i]:
                expected_q[i] += gamma * s_q_actions_target[i, s_argmax_q[i]]

        t3 = ctime()
        _, summaries, loss_eval, abs_loss = sess.run(
            [self.model.train_op, self.model.train_summary_op, self.model.loss,
             self.model.abs_loss],
            feed_dict={self.model.src_: p_states,
                       self.model.src_len_: p_len,
                       self.model.action_idx_: action_id,
                       self.model.expected_q_: expected_q,
                       self.model.b_weight_: b_weight})
        t3_end = ctime()

        self.memo.batch_update(b_idx, abs_loss)

        self.info('loss: {}'.format(loss_eval))
        self.debug('t1: {}, t2: {}, t3: {}'.format(
            t1_end-t1, t2_end-t2, t3_end-t3))
        summary_writer.add_summary(summaries, t - self.hp.observation_t)

    def eval_nn(self, cv):
        self.info('evaluation worker started ...')
        prev_best = None
        while True:
            with cv:
                cv.wait()
                self.info('start evaluation ...')
                trained_steps, eval_res, prev_best = self.evaluate(
                    save_better=True, prev_best=prev_best)
                self.info('after train step: {}, evaluation results: {}'.format(
                    trained_steps, eval_res))

    @classmethod
    def agg_eval_results(cls, eval_results):
        eval_results = np.asarray(eval_results)
        max_score_idx = np.argmax(eval_results[:, 0])
        min_score_idx = np.argmin(eval_results[:, 0])
        total_score = np.sum(eval_results[:, 0])
        total_step = np.sum(eval_results[:, 1])
        return EvalResults(
            total_score=total_score, total_step=total_step,
            max_score=eval_results[max_score_idx, 0],
            max_score_step=eval_results[max_score_idx, 1],
            min_score=eval_results[min_score_idx, 0],
            min_score_step=eval_results[min_score_idx, 1])

    def evaluate(self, save_better=False, prev_best=None):
        best_chkp_prefix = os.path.join(
            self.model_dir, 'best_weights', 'after-epoch')
        with tf.device('/device:GPU:1'):
            eval_model = self.create_eval_model(self.model_creator, self.hp)
        eval_conf = tf.ConfigProto(log_device_placement=False,
                                   allow_soft_placement=True)
        eval_sess = tf.Session(graph=eval_model.graph, config=eval_conf)

        game_clazz = getattr(game, self.hp.game_clazz)
        eval_game = game_clazz(
            self.game_path, max_step=self.hp.game_episode_terminal_t)

        trained_steps = 0
        with eval_model.graph.as_default():
            saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
            restore_from = tf.train.latest_checkpoint(
                os.path.join(self.model_dir, 'last_weights'))
            if restore_from is not None:
                # Reload weights from directory if specified
                self.info("Try to restore parameters")
                saver.restore(eval_sess, restore_from)
                global_step = tf.train.get_or_create_global_step()
                trained_steps = eval_sess.run(global_step)
                self.info('evaluating after {} steps'.format(trained_steps))
            else:
                raise ValueError('cannot load parameters from {}'.format(
                    self.model_dir))

        # start evaluation
        eval_results = []
        eval_logs = []
        for episode in range(self.hp.eval_episode):
            eval_log_in_episode = []
            self.info("evaluating episode: {}".format(episode))
            tjs = self.tjs_creator(
                self.hp,
                padding_val=self.token2idx[self.padding_val])
            tjs.add_new_tj()
            master, room_name = eval_game.reset_game()
            state_t = master.split()
            indexed_state_t = self.index_string(state_t)
            tjs.append(indexed_state_t)
            # self.info('master: {}'.format(' '.join(state_t)))
            cnt_action = np.zeros(self.hp.n_actions)

            t = 0
            while not eval_game.terminal:
                (p_str, m_str, _, _, reward, cnt_action, reports
                 ) = self.take_an_eps_step(
                    t, eval_sess, 0.05, tjs, eval_model, eval_game, cnt_action,
                    play_through_action=None,
                    is_training=False)
                eval_log_in_episode.append(reports)
                tjs.append(self.index_string(p_str))
                tjs.append(self.index_string(m_str))
                if reward > 0:
                    self.debug("Eval: " + self.report_status(reports))
                # self.info("\nplayer: {}\nmaster: {}".format(
                #     " ".join(p_str), " ".join(m_str)))
                t += 1

            eval_results.append(
                [eval_game.total_positive_reward,
                 eval_game.total_step_to_positive_reward])
            eval_logs.append(eval_log_in_episode)

        eval_res = self.agg_eval_results(eval_results)
        # collect and save evaluation results
        np.savez(
            "{}/eval_res-after-epoch-{}.npz".format(
                self.model_dir, trained_steps),
            eval_results=eval_results,
            eval_logs=eval_logs)
        if save_better:
            if prev_best is None:
                total_steps = (self.hp.game_episode_terminal_t *
                               self.hp.eval_episode)
                prev_best = EvalResults(
                    0, total_steps, 0, total_steps, 0, total_steps)
            if ((eval_res.max_score > prev_best.max_score)
                    or (eval_res.total_score > prev_best.total_score)):
                self.info("better evaluation scores found")
                saver.save(
                    eval_sess, best_chkp_prefix, global_step=trained_steps)
                prev_best = eval_res
            elif ((eval_res.max_score >= prev_best.max_score)
                  and (eval_res.max_score_step < prev_best.max_score_step)):
                self.info("better evaluation steps found")
                saver.save(
                    eval_sess, best_chkp_prefix, global_step=trained_steps)
                prev_best = eval_res
            else:
                self.info("no better evaluation result found")
        return trained_steps, eval_res, prev_best

    def human_check(self):
        eval_model = self.create_eval_model(self.model_creator, self.hp)
        eval_sess = tf.Session(graph=eval_model.graph)

        game_clazz = getattr(game, self.hp.game_clazz)
        eval_game = game_clazz(self.game_path,
                               max_step=self.hp.game_episode_terminal_t)

        trained_steps = 0
        with eval_model.graph.as_default():
            saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
            restore_from = tf.train.latest_checkpoint(
                os.path.join(self.model_dir, 'last_weights'))
            if restore_from is not None:
                # Reload weights from directory if specified
                self.info("Try to restore parameters")
                saver.restore(eval_sess, restore_from)
                global_step = tf.train.get_or_create_global_step()
                trained_steps = eval_sess.run(global_step)
                self.info('evaluating after {} steps'.format(trained_steps))
            else:
                raise ValueError('cannot load parameters from {}'.format(
                    self.model_dir))

        # start evaluation
        game_scores = 0
        game_steps = 0
        for episode in range(self.hp.eval_episode):
            self.info("evaluating episode: {}".format(episode))
            tjs = self.tjs_creator(
                self.hp,
                padding_val=self.token2idx[self.padding_val])
            tjs.add_new_tj()
            master, room_name = eval_game.reset_game().split()
            state_t = master.split()
            indexed_state_t = self.index_string(state_t)
            tjs.append(indexed_state_t)
            self.info('master: {}'.format(' '.join(state_t)))

            t = 0
            while not eval_game.terminal:
                (action_idx, player_t, report_action, _
                 ) = self._get_an_eps_action(
                    eval_sess, 0.05, tjs, eval_model, None)
                self.info(report_action)
                human_action = input("action --> ")
                master, room_name, reward = eval_game.take_action(human_action)
                p_str = human_action.split()
                m_str = master.split()
                tjs.append(self.index_string(p_str))
                tjs.append(self.index_string(m_str))
                self.info("player: {}\nmaster: {}\n=========".format(
                    " ".join(p_str), " ".join(m_str)))
                t += 1

            game_scores += eval_game.total_reward
            game_steps += eval_game.total_step

        return game_scores, game_steps, trained_steps
