import pandas as pd
import numpy as np
pred = pd.read_csv("output_p1/pred.csv")
filename = pred["filename"].to_list()
filename = np.array([int(name.split('_')[0]) for name in filename])
ans = np.array(pred["label"].to_list())

print(sum(filename == ans) / len(ans))