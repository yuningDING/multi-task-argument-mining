import argparse
import gc
import pandas as pd
from numpy import mean
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../feedback-prize-effectiveness", required=False)
    parser.add_argument("--input_csv", type=str, default="train.csv", required=False)
    return parser.parse_args()


# TF-IDF Vectorization
def tf_idf_vectorize(train):
    tf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
    train_vec = tf.fit_transform(train).toarray()
    return train_vec


# top N n-grams Vectorization
def ngram_vectorize(train, N=10000, min_n=1, max_n=3):
    cv = CountVectorizer(ngram_range=(min_n, max_n), max_features=N)
    train_vec = cv.fit_transform(train).toarray()
    return train_vec


def logistic_regression_cv(x_train, y_train):
    lr = LogisticRegression(max_iter=500)
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    acc = cross_val_score(lr, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    loss = cross_val_score(lr, x_train, y_train, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    print("CV Accuracy:", mean(acc))
    print("CV Loss:", mean(loss))


args = parse_args()
dataset = pd.read_csv(args.input_dir + '/' + args.input_csv)
dataset['text'] = dataset['essay_id'].apply(lambda x: open(f'../feedback-prize-effectiveness/train/{x}.txt').read())
prompts = pd.read_csv("../clusters_effectivness.csv")

OUTPUT_EFFECT = {"Ineffective": 0, "Adequate": 1, "Effective": 2}
dataset["target"] = dataset["discourse_effectiveness"].map(OUTPUT_EFFECT)
dataset = dataset.reset_index(drop=True)

y_train = dataset["target"]

x_train = tf_idf_vectorize(dataset["discourse_text"])
x_train = sparse.coo_matrix(x_train)

x_text_train = tf_idf_vectorize(dataset["text"])
x_text_train = sparse.coo_matrix(x_text_train)

ohe = OneHotEncoder()
x_type_train = sparse.csr_matrix(ohe.fit_transform(dataset["discourse_type"].values.reshape(-1, 1)))
x_prompt_train = sparse.csr_matrix(ohe.fit_transform(dataset["prompt"].values.reshape(-1, 1)))

print('===baseline===')
logistic_regression_cv(x_train, y_train)
gc.collect()

print("===baseline + argument label===")
x_stack_train = sparse.hstack((x_train, x_type_train))
logistic_regression_cv(x_stack_train, y_train)
gc.collect()

print("===baseline + prompt===")
x_stack_train = sparse.hstack((x_train, x_prompt_train))
logistic_regression_cv(x_stack_train, y_train)
gc.collect()

print('===baseline + text===')
x_stack_train = sparse.hstack((x_train, x_text_train))
logistic_regression_cv(x_stack_train, y_train)
gc.collect()

print("===baseline + text + argument + prompt===")
x_stack_train = sparse.hstack((x_train, x_text_train, x_type_train, x_prompt_train))
logistic_regression_cv(x_stack_train, y_train)
