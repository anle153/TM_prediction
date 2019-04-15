#!/usr/bin/env bash
#/usr/bin/bash

scp -P 4015 *.py deepbox:~/TM_estimation_RNN/
scp -P 4015 ./Utils/*.py deepbox:~/TM_estimation_RNN/Utils/
scp -P 4015 -r ./AttentionsLayer deepbox:~/TM_estimation_RNN
scp -P 4015 -r ./Models/ deepbox:~/TM_estimation_RNN/
#scp -P 4015 -r ./Model_Recorded/* deepbox:~/TM_estimation_RNN/Model_Recorded/

