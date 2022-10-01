
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap 
from sklearn.preprocessing import normalize
from matplotlib.ticker import AutoMinorLocator
from itertools import combinations
class WindowX:

    
    """Basic class for a window explainer for time series to explain lag importances in supplied model's decisions.


    Parameters
    ----------

    dataset: pandas dataframe
        The dataframe used compute a base reference (mean or median) when no specific reference series is given.
    
    ml_model: any class with a predict() function
        The time-series model we want to explain, the output of the model can be a univariate or multi-variate prediction,
        with either aa 1-step or multistep horizon, and can be either real values or labels, in which case the distance 
        used to measure differences in outputs will need to be provided.

    passed_ref: boolean
        True if a reference series is passed and false otherwise. If False, a reference series will be computed from
        the passed dataset.

    ref: numpy array
        the reference point used to compute the distance from, if not given one will be computed from the dataset.

    dataset_ref: "Mean" or "Median"
        Which function to use when computing a reference series from the dataset

    time_windows: iterable of ints
        the time windows used to compute the lag importances.

    time_shifts: iterable of ints
        the time shifts used to compute the lag importances.

    task: "regression" or "classification"
        the type of task the model tackles, this parameter will be used to determine an appropriate default distance used

    distance: string in ["MSE"] or a function
        the distance used between different outputs used to compute the importance of lags

    shap: boolean
        If true, computes shapley values from the dataset and uses multiple permutations instead of just 1 reference.

    

    """

    def __init__(self, dataset, ml_model, passed_ref = False, ref=None, dataset_ref="Mean", time_windows=[1], time_shifts=[1], task = "regression", distance="MSE", shap=False, shap_nsamples = 100, reg = None, impact = None, norm=True, xticks = None, independant = False, excluded_feats=[], excluded_times=[], columns = None):
         
        self.dataset = dataset
        self.ml_model = ml_model
        self.ref_series = passed_ref
        self.dataset_ref = dataset_ref

        if not self.ref_series:
        
            # Computing reference series from the passed dataset
            if self.dataset_ref == "Mean":
                # Using the Mean of the dataset as a reference
                self.ref = dataset.mean(0)
            elif self.dataset_ref == "Median":
                # Using the Median of the dataset as a reference
                self.ref = dataset.median(0)
            else: 
                self.ref = dataset.mean(0)
        else:
             self.ref = ref
        self.time_windows = time_windows                 
        self.time_shifts = time_shifts
        self.task = task
        self.distance = distance
        self.shap_explainer = shap.KernelExplainer(ml_model, dataset) if shap else None
        self.shap_nsamples = shap_nsamples
        self.impact = lambda pred, mod_pred: pred - mod_pred if not impact else impact
        self.reg = reg
        self.xticks = xticks
        self.norm = norm
        self.independet = independant
        self.excluded_feats = excluded_feats
        self.excluded_times = excluded_times
        self.columns = columns

    def get_subwindows(self, i_start, i_end):
        subwindows = []
        for i in range(i_start, i_end):
            subwindows.append((i, ))
            for j in range(i+1, i_end):
                subwindows.append((i, j))
        return subwindows

    def explain(self, input, verbose = True, print_preds = False):

        """
            input: T*D
            Returns lag importances for the specified input in comparison with the reference one (stored in the explainer)
        """

        if print_preds:
            print("REF", self.ref)
            print("INPUT", input)
            if self.ref_series:
                print("PRED REF", self.ml_model.predict(self.ref))
            else:
                print("PRED REF", self.ml_model.predict(np.array([self.ref for a in range(input.shape[1])])))
            print("PRED INPUT", self.ml_model.predict(input))
        T = input.shape[0]
        prediction = self.ml_model.predict(input)
        histo_impact = dict()
        effect = np.zeros(T) 
        for time_shift in self.time_shifts:
            for time_window in self.time_windows:
                for i in range(0, T, time_shift):

                    
                    mod_input = input.copy()
                    i_end = i+time_window if i+time_window <= T else T
                    
                    if self.ref_series:
                        mod_input[i:i_end] = self.ref[i:i_end]
                        for feat in self.excluded_feats:
                            mod_input[i:i_end][:, feat] = input[i:i_end][:, feat]
                        for time in self.excluded_times:
                            mod_input[time] = input[time]
                    else:
                        mod_input[i:i_end] = np.array([self.ref for a in range(i, i_end)])
                        for feat in self.excluded_feats:
                            mod_input[i:i_end][:, feat] = input[i:i_end][:, feat]
                        for time in self.excluded_times:
                            mod_input[time] = input[time]
                            
                    mod_prediction = self.ml_model.predict(mod_input)
                    impact = self.impact(prediction, mod_prediction).squeeze()
                    if not self.independet and time_window > 1:
                        list_combinations = self.get_subwindows(i, i_end)
                        for combination in list_combinations:
                            impact -= histo_impact.get(combination, 0)
                    
                    if not self.reg:
                        effect[i:i_end] += impact
                    elif self.reg == "linear":
                        effect[i:i_end] += impact/time_window
                    elif self.reg == "exp":
                        effect[i:i_end] += impact*np.exp(-time_window+1)
                    elif self.reg == "softmax":
                        effect[i:i_end] += impact/np.exp(time_window)*np.sum(np.exp(np.array(self.time_windows)-1))
                    histo_impact[tuple([k for k in range(i, i_end)])] = impact
        if self.norm and np.max(np.abs(effect)) > 0:
            effect /= np.max(np.abs(effect))
        self.last_impact = effect
        if verbose:
            plt.bar(range(T), effect)
            if self.xticks:
                plt.xticks(self.xticks)
                if self.columns.any():
                    plt.xticks(self.xticks, labels = self.columns, rotation = 90)
            else:
                plt.xticks(range(T))
                if self.columns.any():
                    plt.xticks(range(T), labels = self.columns, rotation = 90)
            plt.grid()
            plt.show()
        return effect
      
    def explain_pf(self, input, verbose = True):

        """
            input: T*D
            Returns lag importances for each feature for the specified input in comparison with the reference one (stored in the explainer)
        """

        if verbose:
            print("REF", self.ref)
            print("INPUT", input)
            if self.ref_series:
                print("PRED REF", self.ml_model.predict(self.ref))
            else:
                print("PRED REF", self.ml_model.predict(np.array([self.ref for a in range(input.shape[1])])))
            print("PRED INPUT", self.ml_model.predict(input))
        T = input.shape[0]
        NB_FEATS = input.shape[1]
        width = 1/NB_FEATS
        prediction = self.ml_model.predict(input)
        histo_impact = [dict() for i in range(NB_FEATS)]
        effect = np.zeros((NB_FEATS, T))
        for feature in range(NB_FEATS):
            if feature not in self.excluded_feats:
                for time_shift in self.time_shifts:
                    for time_window in self.time_windows:
                        for i in range(0, T, time_shift):
                            mod_input = input.copy()
                            i_end = i+time_window if i+time_window <= T else T

                            if self.ref_series:
                                mod_input[i:i_end][:, feature] = self.ref[i:i_end][:, feature]
                            else:
                                mod_input[i:i_end, feature] = np.array([self.ref[feature] for a in range(i, i_end)])
                                
                            mod_prediction = self.ml_model.predict(mod_input)
                            impact = self.impact(prediction, mod_prediction).squeeze()

                            if not self.independet and time_window > 1:
                                list_combinations = self.get_subwindows(i, i_end)
                                for combination in list_combinations:
                                    impact -= histo_impact[feature].get(combination, 0)
                            if not self.reg:
                                effect[feature, i:i_end] += impact
                            elif self.reg == "linear":
                                effect[feature, i:i_end] += impact/time_window
                            elif self.reg == "exp":
                                effect[feature, i:i_end] += impact*np.exp(-time_window+1)
                            elif self.reg == "softmax":
                                effect[feature, i:i_end] += impact/np.exp(time_window)*np.sum(np.exp(np.array(self.time_windows)-1))
                            histo_impact[feature][tuple([k for k in range(i, i_end)])] = impact
                if self.norm and np.max(np.abs(effect)) > 0:
                    effect[feature] /= np.max(np.abs(effect[feature]))

                if verbose:
                        plt.bar(np.array(range(T))+width*feature, effect[feature], width, label="feature {}".format(feature))
        if verbose:
            plt.legend()
            if self.xticks:
                plt.xticks(self.xticks)
            else:
                plt.xticks(np.array(range(T))+0.5, labels=np.array(range(T)))
            minor_locator = AutoMinorLocator(2)
            plt.gca().xaxis.set_minor_locator(minor_locator)
            plt.grid(which="minor")
            plt.grid(ls="--", axis='y')
            plt.show()
        self.last_impact = effect
        return effect


    def explain_pf_shaplike(self, input, refs):

        """
            input: T*D
        """
        
        print("REF", refs)
        input = input.astype(float)
        NB_FEATS = input.shape[1]
        width = 1/NB_FEATS
        T = input.shape[0]
        prediction = self.ml_model.predict(input)
        histo_pred = []
        effect = np.zeros((NB_FEATS, T))
        for feature in range(NB_FEATS):
            if feature not in self.excluded_feats:
                for time_shift in self.time_shifts:
                    for time_window in self.time_windows:
                        for ref in refs:
                            for i in range(0, T, time_shift):
                                mod_input = input.copy()
                                i_end = i+time_window if i+time_window <= T else T
                                mod_input[i:i_end][:, feature] = ref[i:i_end][:, feature]
                                mod_prediction = self.ml_model.predict(mod_input)
                                impact = self.impact(prediction, mod_prediction).squeeze()
                                histo_pred.append(impact)
                                if not self.reg:
                                    effect[feature, i:i_end] += impact
                                elif self.reg == "linear":
                                    effect[feature, i:i_end] += impact/time_window
                                elif self.reg == "exp":
                                    effect[feature, i:i_end] += impact*np.exp(-time_window+1)
                                elif self.reg == "softmax":
                                    effect[feature, i:i_end] += impact/np.exp(time_window)*np.sum(np.exp(np.array(self.time_windows)-1))
                if np.max(np.abs(effect[feature])):
                    effect[feature] /= np.max(np.abs(effect[feature]))
                plt.bar(np.array(range(T))+width*feature, effect[feature], width, label="feature {}".format(feature))
        plt.legend()
        if self.xticks:
            plt.xticks(self.xticks)
        else:
            plt.xticks(np.array(range(T))+0.5, labels=np.array(range(T)))
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        plt.grid(which="minor")
        plt.grid(ls="--", axis='y')
        plt.show()
        self.last_impact = effect
        return effect
    
    def explain_shap(self, input):

        if self.shap_explainer:

            T = input.shape[0]
            prediction = self.ml_model.predict(input)
            histo_pred = []
            effect = np.zeros(T) 
            for time_shift in self.time_shifts:
                for time_window in self.time_windows:
                    for i in range(0, T, time_shift):
                        mod_input = input.copy()
                        i_end = i+time_window if i+time_window <= T else T
                        if self.ref_series:
                            mod_input[i:i_end] = self.ref[i:i_end]
                        else:
                            mod_input[i:i_end] = np.array([self.ref for a in range(i, i_end)])
                        mod_prediction = self.ml_model.predict(mod_input)
                        impact = self.shap_explainer.shap_values(mod_input, nsamples=self.shap_nsamples)

                        histo_pred.append(impact)

                        if not self.reg:
                            effect[i:i_end] += impact
                        elif self.reg == "linear":
                            effect[i:i_end] += impact/time_window
                        elif self.reg == "exp":
                            effect[i:i_end] += impact*np.exp(-time_window+1)
                        elif self.reg == "softmax":
                            effect[i:i_end] += impact/np.exp(time_window)*np.sum(np.exp(np.array(self.time_windows)-1))

            norm_max = np.max(np.abs(effect))
            if norm_max > 0:
                effect /= np.max(np.abs(effect))
            self.last_impact = effect
            plt.bar(range(T), effect)
            if self.xticks:
                plt.xticks(self.xticks)
            else:
                plt.xticks(range(T))

            plt.grid()
            plt.show()
            return effect
        else:
            print("NO SHAP")


