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
import re
import sys
from queue import Queue, Empty
from subprocess import PIPE, Popen
from threading import Thread


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TextPlayer(object):
    """
    This repo is adapted from
    [textplayer](https://github.com/danielricks/textplayer),
    thanks the original author.
    A few things are changed for my project, especially adding
    some try .. except .. to avoid crash when re-init the game
    """
    def __init__(self, game_path):
        self.lib_dir = os.path.dirname(os.path.realpath(__file__))
        self.dfrotz_exec_path = '{}/../../frotz/dfrotz'.format(self.lib_dir)
        self.game_path = game_path
        self.game_filename = os.path.basename(self.game_path)

        # Verify that specified game file exists, else limit functionality
        if self.game_path is None or not os.path.exists(self.game_path):
            raise ValueError("zplayer: unrecognized game file or bad path")

        self.game_process = None
        self.output_queue = None

    def _init_process(self):
        self.game_process = Popen([self.dfrotz_exec_path, self.game_path],
                                  stdin=PIPE, stdout=PIPE, bufsize=1)
        # Create Queue object
        self.output_queue = Queue()
        t = Thread(target=self.enqueue_pipe_output,
                   args=(self.game_process.stdout, self.output_queue),
                   daemon=True)
        t.start()

    # async pipe for game output buffer
    @staticmethod
    def enqueue_pipe_output(output, queue):
        for line in iter(output.readline, b''):
            queue.put(line)
        output.close()

    def _exec(self, command, timeout=None):
        raw_output = None
        try:
            self.game_process.stdin.write((command + '\n').encode("ascii"))
            self.game_process.stdin.flush()
            raw_output = (self._get_command_output() if timeout is None
                          else self._get_command_output(timeout))
        except Exception as e:
            try:
                self.game_process.terminate()
            except Exception as e_terminate:
                eprint("zplayer: terminate game process with error: {}".format(
                    e_terminate))
                eprint("error ignored!")
            self.game_process = None
            eprint("zplayer: command {} execution error: {}".format(command, e))
        return raw_output

    def _readout(self, raw_output):
        g_output = self.clean_command_output(raw_output)
        room_name, score, moves = self._get_status_bar(raw_output)
        return g_output, room_name, score, moves

    def execute_command(self, command):
        """
        There is not always having a room name.
        if there is no room name parsed out, we can treat it as the previous
        room.
        :param command:
        :return:
        """
        raw_output = self._exec(command)
        if raw_output is None:
            master = "game terminated"
            room_name = None
            score = None
            moves = None
        else:
            master, room_name, score, moves = self._readout(raw_output)
        return master, room_name, score, moves

    @staticmethod
    def _get_status_bar(text):
        regex = '^\s(.*)\s+Score:\s*(-*\d+)\s+Moves:\s*(\d+).*$'
        match_obj = re.search(regex, text, re.M | re.I)
        if match_obj is not None:
            room_name = match_obj.group(1).strip()
            score = float(match_obj.group(2))
            moves = int(match_obj.group(3))
            return room_name, score, moves
        else:
            return None, None, None

    @staticmethod
    def clean_command_output(text):
        # Clean up the output
        text = text.replace('\n', ' ')
        regex_list = ['[0-9]+/[0-9+]', 'Score:[ ]*[-]*[0-9]+',
                      'Moves:[ ]*[0-9]+', 'Turns:[ ]*[0-9]+',
                      '[0-9]+:[0-9]+ [AaPp][Mm]', ' [0-9]+ \.']
        for regex in regex_list:
            match_obj = re.search(regex, text, re.M|re.I)
            if match_obj is not None:
                text = text[match_obj.end() + 1:]
        return text

    def _get_command_output(self, timeout=0.001):
        command_output = ''
        output_continues = True

        # While there is still output in the queue
        while output_continues:
            try:
                line = self.output_queue.get(timeout=timeout)
            except Empty:
                output_continues = False
            else:
                command_output += line.decode("ascii")
            command_output = command_output.replace(">", " ").replace("<", " ")
        return command_output

    def quit(self):
        if self.game_process is not None:
            self.game_process.stdin.write('quit\n'.encode("ascii"))
            self.game_process.stdin.write('y\n'.encode("ascii"))
            self.game_process.terminate()
            eprint(self._get_command_output())
        else:
            eprint('zplayer: game not start')

    @staticmethod
    def remove_version_info(text):
        zork_info = [
            "ZORK I: The Great Underground Empire",
            "Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.",
            "ZORK is a registered trademark of Infocom, Inc.",
            "Revision 88 / Serial number 840726"]
        # don't strip texts, keep the raw response
        return "\n".join(
            filter(lambda s: s.strip() not in zork_info, text.split("\n")))

    def reset(self):
        """
        There are two ways to reset the game. (just for Zork)
         1. close the process and start a new one;
         2. use the command "restart" of the game.
        :return: initial master response
        """
        new_threading_retry = 3
        _ = self._exec('restart', timeout=0.1)
        raw_output = self._exec('y')
        n_retry = 0
        while raw_output is None and n_retry < new_threading_retry:
            eprint("restart game error, init new thread: {}".format(n_retry))
            self._init_process()
            raw_output = self._get_command_output(timeout=0.1)
            n_retry += 1

        master, room_name, score, moves = self._readout(
            self.remove_version_info(raw_output))
        # when resetting, there could be noisy tokens
        expected_room_name = "West of House"
        if room_name is None or expected_room_name in room_name:
            room_name = expected_room_name
        return master, room_name, score, moves