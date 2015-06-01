import numpy as np

def softmax(x):
        if len(x.shape)==1:
            x = x - np.max(x)
            exp_x = np.exp(x);
            normalize_factor = np.sum(exp_x)
            x = exp_x/normalize_factor
            return x
        m = x.shape[0]
        max_in_each_row = np.max(x, axis = 1).reshape([m,1]);
        x = x - max_in_each_row;   # removing maximum of each row from entries of that row
        exp_x = np.exp(x);
        normalize_factor = np.sum(exp_x, axis = 1).reshape([m,1])
        x = exp_x/normalize_factor
        return x

