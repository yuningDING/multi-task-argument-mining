echo 'Linear Regression'
python3.7 ./experiments_study_1/experiment_LR_effectiveness.py
echo 'BERT - Baseline'
python3.7 ./experiments_study_1/experiment_BERT_effectiveness.py
echo 'BERT - Baseline+Prompt'
python3.7 ./experiments_study_1/experiment_BERT_effectiveness.py --extra_feature prompt
echo 'BERT - Baseline+Argument Type'
python3.7 ./experiments_study_1/experiment_BERT_effectiveness.py --extra_feature argument
echo 'BERT - Baseline+Text'
python3.7 ./experiments_study_1/experiment_BERT_effectiveness.py --plus_text True
echo 'BERT - Baseline+Prompt+Argument Type+Text'
python3.7 ./experiments_study_1/experiment_BERT_effectiveness.py --extra_feature both --plus_text True