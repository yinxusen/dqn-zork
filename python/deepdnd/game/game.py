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

import os
import string

from nltk import word_tokenize, sent_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser

from deepdnd.utils import eprint
from deepdnd.log import Logging
from deepdnd.game import textplayer as tp

negative_masters_whole =\
"""
you dont
its too dark to see
those things arent here
its not clear
you cant
youre not
you already have that
there is no
you can not
it is already
have your eyes checked
i cant see
"""

negative_masters = list(filter(
    lambda nm: nm != "",
    map(lambda nm: nm.strip(), negative_masters_whole.split("\n"))))

# to remove punctuation
empty_trans_table = str.maketrans("", "", string.punctuation)

# be sure of starting CoreNLP server first
parser = CoreNLPDependencyParser()
# use dict to avoid parse the same sentences.
parsed_sentences = dict()


def clean_master(master):
    return ' '.join(map(lambda t: t.lower(),
               filter(lambda t: t.isalpha(), word_tokenize(
                   master.translate(empty_trans_table)))))


def dependency_parser_reordering(sent):
    tree = next(parser.raw_parse(sent)).tree()
    t_labels = ([
        [head.label()] +
        [child if type(child) is str else child.label() for child in head]
        for head in tree.subtrees()])
    t_str = [" ".join(labels) for labels in t_labels]
    return " O O O O ".join(t_str)


def dependency_blocks(master):
    """
    Notice that four padding letters " O O O O " can only work up to 5-gram
    :param master:
    :return:
    """
    sent_list = list(filter(
        lambda sent: sent != "",
        [clean_master(m) for m in sent_tokenize(master)]))
    tree_strs = []
    for s in sent_list:
        if not s in parsed_sentences:
            t_str = dependency_parser_reordering(s)
            parsed_sentences[s] = t_str
            eprint("parse {} into {}".format(s, t_str))
        else:
            eprint("found parsed {}".format(s))
        tree_strs.append(parsed_sentences[s])
    return " O O O O " + " O O O O ".join(tree_strs) + " O O O O "


def preprocess_master_output(master):
    if master == "":
        return master
    # return dependency_blocks(master)
    return clean_master(master)


class ZGameZork(Logging):
    def __init__(self, game_path, max_step=None):
        super(ZGameZork, self).__init__()
        self.game_name = os.path.basename(game_path)
        self.total_reward = 0
        self.total_shaped_reward = 0
        self.total_positive_reward = 0
        self.total_step_to_positive_reward = 0
        self.total_step = 0
        self.terminal = False
        self.game = tp.TextPlayer(game_path)
        self.max_step = max_step
        self.prev_room_name = None

    def reset_game(self, max_step=None):
        self.total_reward = 0
        self.total_step = 0
        self.total_shaped_reward = 0
        self.total_positive_reward = 0
        self.total_step_to_positive_reward = 0
        self.terminal = False
        if max_step is not None:
            self.debug("max step changed {} -> {}".format(
                self.max_step, max_step))
            self.max_step = max_step
        master, room_name, score, moves = self.game.reset()
        master = preprocess_master_output(master)
        self.prev_room_name = room_name
        verbose_response, _, _, _ = self.game.execute_command("verbose")
        eprint(verbose_response)
        return master, room_name

    def get_instant_reward(self, score, master):
        raw_reward = score - self.total_reward
        if raw_reward > 0:
            self.total_positive_reward += raw_reward
            self.total_step_to_positive_reward = self.total_step
        instant_reward = self.clip_reward(
            raw_reward + self.language_reward(master) - 0.1)
        return instant_reward

    def take_action(self, action):
        self.total_step += 1
        master, room_name, score, moves = self.game.execute_command(action)
        self.terminal = self.is_terminal(master)
        master = preprocess_master_output(master)
        if score is None:
            score = self.total_reward
        if room_name is None:
            room_name = self.prev_room_name
        instant_reward = self.get_instant_reward(score, master)
        self.total_reward = score
        self.total_shaped_reward += instant_reward
        self.prev_room_name = room_name
        return master, room_name, instant_reward

    @classmethod
    def clip_reward(cls, reward):
        """clip reward into [-1, 1]"""
        return max(min(reward, 1), -1)

    def is_terminal(self, master):
        terminal = False
        if self.max_step is not None and self.total_step >= self.max_step:
            terminal = True
        if master == "game terminated":
            terminal = True
        return terminal

    @classmethod
    def language_reward(cls, master):
        reward = 0
        if any(map(lambda nm: nm in master, negative_masters)):
            reward += (-1)
        return reward
