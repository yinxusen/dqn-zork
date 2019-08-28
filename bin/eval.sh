#!/bin/bash

set -e -x

FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PDIR="$FWDIR/.."
filename=$(basename "$0")
extension="${filename##*.}"
filename="${filename%.*}"

DATAHOME="$PDIR/../zplayer/resources/games/zork1.z5"
MODELHOME=$1

ACTION_FILE="$PDIR/resources/commands-zork1-egg-take.txt"
VOCAB_FILE="$PDIR/resources/vocab.50K.en.trimed"
TGT_VOCAB_FILE="$PDIR/resources/vocab.mini-zork.txt"

./bin/run.sh python/deepdnd/main.py \
    -d $DATAHOME -m $MODELHOME \
    --mode evaluation --eval_episode 1 \
    --game_episode_terminal_t 600
