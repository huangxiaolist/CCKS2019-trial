# CCKS2019-trial
IN CCKS2019 relation extraction, but I dont finished this task. Also,because I dont join in this game,so the test-dataset doesnt have label.
# Note!
This datasets just to test model that I designed  or the open-sourced. MEANINGLESS!!!
# data
-  [origin data](https://pan.baidu.com/s/1EGPYAQp90usvpzROdNilPw) d6vp
- [preprocessed data](https://pan.baidu.com/s/1lHD-IO7zI4EwyNfi_Fe1HQ) 4f93
Then put all files  into directory `data`
# train&eval
- python *.py -name pcnn -data data/person_preprocessed.pkl (the rest parameters)
## model
1. PCNN_SATT.py
- just use PCNN + Sentence_att  to build bag representation
- when use dropout=0.8 lr=0.1 filter_nums=200.the F1=0.168
2. BiLSTM_WATT_SATT.py
- just use BiLSTM + Word_att + Sentence_att to build bag representation

