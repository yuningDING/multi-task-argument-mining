# Multi-Task Argument Mining

This is the code to 2023 ACL Finding paper for <br>Score It All Together: A Multi-Task Learning Study on Automatic Scoring of Argumentative Essays</br>. 

In Study 1, we show that the automatic scoring of argument quality benefits from additional information about context, writing prompt and argument type. 

In Study 2, we explore the different combinations of three tasks: automated span detection, type and quality prediction. Results show that a multi-task learning approach combining the three tasks outperforms sequential approaches that first learn to segment and then predict the quality/type of a segment.
The architecture of multi-task learning is shown below.

### How to use

1. Download data from: https://www.kaggle.com/c/feedback-prize-2021/ and save data to './data'

2. Install environment

    ```bash
    conda create --name env python=3.7
    conda activate env
    pip install -r experiment_requirements.txt
    ```
    
3. Split data into clusters and different experiment settings
    ```bash
    python ./data_split.py
    ```

4. To run all experiments in Study 1, execute `bash run_study_1.sh`.

5. To run study 2, refer to `run_study_2.sh`.

### How to cite

```
@InProceedings{ding2023,
  title     = {Score It All Together: A Multi-Task Learning Study on Automatic Scoring of Argumentative Essays},
  author    = {Ding, Yuning and Bexte, Marie and Horbach, Andrea},
  booktitle = {Proceedings of the Findings of the association for computational linguistics: ACL-Findings 2023},
  year      = {2023}
}
```
