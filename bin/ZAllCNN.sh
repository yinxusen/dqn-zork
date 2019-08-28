#!/bin/bash

set -e -x

FWDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PDIR="$FWDIR/.."
filename=$(basename "$0")
extension="${filename##*.}"
filename="${filename%.*}"

# prepare jdk for stanford parser
if [[ -f $HOME/local/etc/init_jdk.sh ]]; then
    source $HOME/local/etc/init_jdk.sh
fi

CORENLP_HOME="$HOME/local/opt/corenlp"
pushd $CORENLP_HOME
source ./start-server.sh
popd
sleep 120  # wait for stanford parser server ready

DATAHOME="$PDIR/../zplayer/resources/games/zork1.z5"
MODELHOME="$PDIR/../experiments/agent-zork-${filename}"

ACTION_FILE="$PDIR/resources/commands-zork1-minimum.txt"
VOCAB_FILE="$PDIR/resources/vocab.50K.en.trimed"
TGT_VOCAB_FILE="$PDIR/resources/vocab.mini-zork.txt"

if [[ ! -d $MODELHOME ]]; then
    mkdir $MODELHOME
fi

pushd $PDIR
./bin/run.sh python/deepdnd/main.py \
    -d $DATAHOME -m $MODELHOME \
    --action_file $ACTION_FILE --vocab_file $VOCAB_FILE --tgt_vocab_file $TGT_VOCAB_FILE \
    --annealing_eps_t 2000000  --annealing_gamma_t 2000 --observation_t 50000 --replay_mem 500000 \
    --eval_episode 10 --save_gap_t 5000 --game_episode_terminal_t 600 \
    --batch_size 32  --model_creator CNNEncoderDQN --game_clazz ZGameZork
popd
