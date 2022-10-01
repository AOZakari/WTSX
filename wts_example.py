import random
import wts
import numpy as np

# Dummy models
class Dummy_model:

    """
            dummy predictor with the t time step's n_th variable influencing only 
    """

    def __init__(self, t, n):
        self.t = t
        self.n = n

    def predict(self, input):
        return input[self.t][self.n]

class Dummy_model_feats:

    """
            dummy predictor with the t time step's n_th to m_th variables influencing only 
    """

    def __init__(self, t, n, m):
        self.t = t
        self.n = n
        self.m = m

    def predict(self, input):
        return np.sum(input[self.t][self.n:self.m])

class Dummy_model2:

    """
            dummy predictor with the t to k_th time steps' n_th variable influencing only 
    """

    def __init__(self, t, k, n):
        self.t = t
        self.n = n
        self.k = k

    def predict(self, input):
        return np.sum(input[self.t:self.k, self.n])

class Dummy_model2_feats:

    """
            dummy predictor with the t to k_th time steps' n_th to m_th variables influencing only 
    """

    def __init__(self, t, k, n, m):
        self.t = t
        self.n = n
        self.k = k
        self.m = m

    def predict(self, input):
        return np.sum(input[self.t:self.k, self.n:self.m])        


class Dummy_model3:

    """
            dummy predictor with the t to k_th time steps' n_th variable influencing only 
    """

    def __init__(self, t, k, n):
        self.t = t
        self.n = n
        self.k = k

    def predict(self, input):
        return np.sum(input[self.t, self.n]) + np.sum(input[self.k, self.n])
class Dummy_model3_feats:

    """
            dummy predictor with the t to k_th time steps' n_th to m_th variables influencing only 
    """

    def __init__(self, t, k, n, m):
        self.t = t
        self.n = n
        self.k = k
        self.m = m

    def predict(self, input):
        return np.sum(input[self.t, self.n:self.m]) + np.sum(input[self.k, self.n:self.m]) 

class Dummy_model4_feats:

    """
            dummy predictor with the t to k_th time steps' n_th to m_th variables influencing only 
    """

    def __init__(self, t, k, n, m):
        self.t = t
        self.n = n
        self.k = k
        self.m = m

    def predict(self, input):
        return np.sum(input[self.t, self.n:self.m]) * np.sum(input[self.k, self.n:self.m]) * (np.sum(input[self.m, self.n:self.m])) + (np.sum(input[self.n, self.n:self.m]))

class Dummy_model5_feats:
    """
            dummy predictor with the t to k_th time steps' n_th to m_th variables influencing only 
    """
    def __init__(self, t, k, n, m, r1, r2, random=False):
            self.t = t
            self.n = n
            self.k = k
            self.m = m
            self.r1 = r1
            self.r2 = r2
            self.random = random

    def predict(self, input):
        r = 0
        if self.random:
            r = 0.1*random.random()
        if np.sum(input[self.t, self.n:self.m]) > np.sum(input[self.k, self.n:self.m]):
            return np.sum(input[self.r1, self.n:self.m]) * (1+r)
        else:
            return np.sum(input[self.r2, self.n:self.m]) * (1+r)

# Initializing dummy series and models:

dummy_series_length = 22
dummy_series = np.array([[i+j*dummy_series_length for i in range(dummy_series_length)] for j in range(dummy_series_length)])
dummy_series_ones = np.ones(dummy_series.shape)
dummy_series_refs_num = 2
dummy_series_refs = np.random.rand(dummy_series_refs_num, dummy_series_length, dummy_series_length)
ref_dummy_series = np.array([dummy_series.mean(1) for a in range(dummy_series_length)])
dummy_ml = Dummy_model4_feats(5, 18, 2, 10)

print("REF INPUT", ref_dummy_series)
print("INPUT", dummy_series)
print("REF PRED", dummy_ml.predict(ref_dummy_series))
print("INPUT PRED", dummy_ml.predict(dummy_series_ones))

# Initializing explainer:


explainer = wts.WindowX(dummy_series, dummy_ml, time_windows=[1, 2, 3, 4, 5, 6, 7, 8, 9], time_shifts=[1], norm=False, reg="exp", excluded_feats=[0])

# Computing explanation:

explanation = explainer.explain_pf(dummy_series)
