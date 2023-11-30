import numpy as np
import pandas as pd

train_pd = pd.read_csv("hw2_data/digits/usps/train.csv")
train_file = train_pd["image_name"].to_list()
train_label = train_pd["label"].to_list()
valid_pd = pd.read_csv("hw2_data/digits/usps/val.csv")
valid_file = valid_pd["image_name"].to_list()
valid_label = valid_pd["label"].to_list()

train_file.extend(valid_file)
train_label.extend(valid_label)


test_pd = pd.read_csv("DANN/test/test_pred.csv")
test_file = test_pd["image_name"].to_list()
test_label = test_pd["label"].to_list()

train_file = np.array(train_file)
train_label = np.array(train_label)
test_file = np.array(test_file)
test_label = np.array(test_label)

# print(len(train_file))
index = np.argsort(train_file)
train_label = train_label[index]
index = np.argsort(test_file)
test_label = test_label[index]

print("acc:", (train_label == test_label).mean())