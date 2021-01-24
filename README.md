# SpEx Multi-scale Time Domain Speaker Extraction Network

The codes here are speaker extraction, where only target speaker's voice will be extracted given this target speaker's characteristics. In paper 1) SpEx, we use MFCC features toghether with LSTMs to jointly learn target speaker's characteristics from a different utterance of target speaker. In paper 2) SpEx+, as the time-domian speech encoder is applied that results in thousands of frames, the speaker encoder adopts a shared speech encoder and ResBlock to jointly optimize a multi-task learning task. The rests of the network of 1) and 2) are same. You also can replace the network by using i-vector, or x-vector network.

If you are interested in speech separation to get all the speaker's voices in the mixture, please move to https://github.com/xuchenglin28/speech_separation

If you are interested in frequency-domain speaker extraction with tensorflow implementation, please move to https://github.com/xuchenglin28/speaker_extraction

## Papers

Please cite:

 1) Chenglin Xu, Wei Rao, Eng Siong Chng and Haizhou Li, "SpEx: Multi-Scale Time Domain Speaker Extraction Network," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1370-1384, 2020, doi: 10.1109/TASLP.2020.2987429. (Open Access at https://ieeexplore.ieee.org/document/9067003)
 2) Meng Ge, Chenglin Xu, Longbiao Wang, Eng Siong Chng, Jianwu Dang and Haizhou Li, "SpEx+: A Complete Time Domain Speaker Extraction Network", in Proc. of INTERSPEECH 2020, pp 1406-1410. (Open Access at https://isca-speech.org/archive/Interspeech_2020/pdfs/1397.pdf)

## Data Generation:

1) If you are using wsj0 to simulate data with maximum protocal for WSJ0-2mix_extr as reported in Table 1 and Table 2 in paper 2), please vist https://github.com/xuchenglin28/speaker_extraction,  and read the code in run_data_generation.sh for detials, and change the path accordingly.

The list of files and SNRs for {training, development and test sets} are in simulation/mix_2_spk_{tr,cv,tt}\_extr.txt. In the files, the first column is the utterance of the target speaker to generate mixture and also used as target clean to supervise the network learning. The seconde column is the interference speaker to generate the mixture. The third column is the taget speaker's another utterane to obtain speaker's characteristics.

After run the .sh script, there will be 3 folders {mix, aux, s1} for the three sets {tr, cv, tt}. The mix folder is the mixture speech, aux folder is the utterances to obtain speaker's characteristics, and s1 is the folder of target clean. In all three folders, the names are cosistent for each example. 

2) If you are using wsj0 to simulate data with mimumun protocal for WSJ0-2mix as reported in Table 4 in paper 2), please visit https://github.com/xuchenglin28/speech_separation to generate the database for speech separation first. Then you can use the lists as included in this repository as data/wsj0_2mix/{tr, cv, tt_avg, tt_aux60}. tt_avg adopts a reference of a utterance with random duration as repoted in Table 3 in paper 2). tt_aux60 daopts a reference of a utterance with 60s, which is combing several origial WSJ0 utterances of same speaker. The combintation files for each 60s reference are included in data/wsj0_2mix/tt_aux60/aux_generated_list.scp

## Speaker Extraction

This part includes model training, run-time inference, and evaluation. Please read the train.sh, decode.sh and evaluation.sh codes for detail and revise accordingly.

### The results reported in paper 1) is based on tensorflow implementation with sigmoid activation for mask estimation and discarding the utterances less than 4s. It is different from this pytorch implementation. And we use this pytorch implementation to reimplemented the SpEx, whose results are reported as in paper 2).

## Environments:

python: 3.8

Pytorch: 1.6

## Contact

e-mail: xuchenglin28@gmail.com

## Licence

The code and models in this repository are licensed under the GNU General Public License Version 3.

## Citation
If you would like to cite, use this :
```BibTex
@article{xu2020spex,
  title={SpEx: Multi-scale time domain speaker extraction network},
  author={Xu, Chenglin and Rao, Wei and Chng, Eng Siong and Li, Haizhou},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={1370--1384},
  year={2020},
  publisher={IEEE}
}
@inproceedings{ge2020spex+,
  title={Spex+: A complete time domain speaker extraction network},
  author={Ge, Meng and Xu, Chenglin and Wang, Longbiao and Chng, Eng Siong and Dang, Jianwu and Li, Haizhou},
  booktitle={Proc. of INTERSPEECH},
  pages={1406--1410},
  year={2020}
}
```
