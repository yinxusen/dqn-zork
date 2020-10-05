# DQN-Zork (deprecated)

Note: This repository is deprecated. Please refer to the new repository: https://github.com/yinxusen/deepword

A DQN framework to play Zork.

# Dependencies

Use Python3 for this repository.

First, install python dependencies from requirements.txt;

Second, install fortz, which is a Z-machine interpreter.

Third, install Stanford CoreNLP to CORENLP_HOME="$HOME/local/opt/corenlp"

```bash
pip3 install requirements.txt
git clone https://github.com/DavidGriffith/frotz.git fortz
cd fortz
make dumb
```

Thanks David Griffith for providing fortz, and Daniel Ricks for the text player.



cite our paper:

```
@article{DBLP:journals/corr/abs-1905-02265,
  author    = {Xusen Yin and
               Jonathan May},
  title     = {Comprehensible Context-driven Text Game Playing},
  journal   = {CoRR},
  volume    = {abs/1905.02265},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.02265},
  archivePrefix = {arXiv},
  eprint    = {1905.02265},
  timestamp = {Mon, 27 May 2019 13:15:00 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-02265},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
