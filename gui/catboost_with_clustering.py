import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pandastable import Table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from mrmr import mrmr_regression
from catboost import CatBoostRegressor
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster.dbscan import dbscan
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV

import os
import json
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from .helpers import *

class CatBoostWithClustering:
    def __init__(self):
        self.root = ttk.Frame()

        # Get Train Set
        get_train_set_frame = ttk.Labelframe(self.root, text="Get Train Set")
        get_train_set_frame.grid(column=0, row=0)

        file_path = tk.StringVar(value="")
        ttk.Label(get_train_set_frame, text="Train File Path").grid(column=0, row=0)
        ttk.Entry(get_train_set_frame, textvariable=file_path).grid(column=1, row=0)
        ttk.Button(get_train_set_frame, text="Read Data", command=lambda: self.readCsv(file_path)).grid(column=2, row=0)

        self.input_list = tk.Listbox(get_train_set_frame)
        self.input_list.grid(column=0, row=1)
        self.input_list.bind("<Double-Button-1>", self.addPredictor)
        self.input_list.bind("<Double-Button-3>", self.addTarget)

        self.predictor_list = tk.Listbox(get_train_set_frame)
        self.predictor_list.grid(column=1, row=1)
        self.predictor_list.bind("<Double-Button-1>", self.ejectPredictor)

        self.target_list = tk.Listbox(get_train_set_frame)
        self.target_list.grid(column=2, row=1)
        self.target_list.bind("<Double-Button-1>", self.ejectTarget)

        ttk.Button(get_train_set_frame, text="Add Predictor", command=self.addPredictor).grid(column=1, row=2)
        ttk.Button(get_train_set_frame, text="Eject Predictor", command=self.ejectPredictor).grid(column=1, row=3)

        ttk.Button(get_train_set_frame, text="Add Target", command=self.addTarget).grid(column=2, row=2)
        ttk.Button(get_train_set_frame, text="Eject Target", command=self.ejectTarget).grid(column=2, row=3)

        # Customize Train Set
        customize_train_set_frame = ttk.LabelFrame(self.root, text="Customize Train Set")
        customize_train_set_frame.grid(column=0, row=2)

        self.scale_var = tk.StringVar(value="None")
        ttk.Label(customize_train_set_frame, text="Scale Type").grid(column=0, row=0)
        ttk.OptionMenu(customize_train_set_frame, self.scale_var, "None", "None","StandardScaler", "MinMaxScaler").grid(column=1, row=0)
        
        self.anomaly_n_estimator = tk.IntVar(value=100)
        self.anomaly_option = tk.IntVar(value=0)  
        
        tk.Checkbutton(customize_train_set_frame, text="Anomaly Detection with IsolationForest", offvalue=0, onvalue=1, variable=self.anomaly_option, command=self.openAllEntries).grid(column=0, row=1, columnspan=3)

        ttk.Label(customize_train_set_frame, text="N_Estimators:").grid(column=0, row=2)
        self.anomaly_n_estimator_entry = ttk.Entry(customize_train_set_frame, textvariable=self.anomaly_n_estimator, width=8, state=tk.DISABLED)
        self.anomaly_n_estimator_entry.grid(column=1, row=2, pady=2)
        
        self.fs_option = tk.IntVar(value=0)
        self.mrmr_count = tk.IntVar(value=5)
        ttk.Checkbutton(customize_train_set_frame, text="Feature Selection:", offvalue=0, onvalue=1, variable=self.fs_option, command=self.openAllEntries).grid(column=0, row=3)
        self.fs_entry = ttk.Entry(customize_train_set_frame, textvariable=self.mrmr_count, width=8)
        self.fs_entry.grid(column=1, row=3)

        ## Cluster Frame
        cluster_frame = ttk.Labelframe(self.root, text="Clustering Frame")
        cluster_frame.grid(column=1, row=0)

        ### DBSCAN
        self.cluster_type = tk.IntVar(value=0)
        tk.Radiobutton(cluster_frame, text="Dbscan", value=0, variable=self.cluster_type, command=self.openAllEntries).grid(column=0, row=0, sticky=tk.W)

        self.dbscan_parameters = [tk.DoubleVar(value=0.5), tk.IntVar(value=3)]
        ttk.Label(cluster_frame, text="Eps:").grid(column=0, row=1)
        self.dbscan_eps_entry = ttk.Entry(cluster_frame, textvariable=self.dbscan_parameters[0], width=8, state=tk.DISABLED)
        self.dbscan_eps_entry.grid(column=1, row=1, pady=2)
        
        ttk.Label(cluster_frame, text="Neighbors:").grid(column=0, row=2)
        self.dbscan_neighbors_entry = ttk.Entry(cluster_frame, textvariable=self.dbscan_parameters[1], width=8, state=tk.DISABLED)
        self.dbscan_neighbors_entry.grid(column=1, row=2, pady=2)

        ### CLARA
        tk.Radiobutton(cluster_frame, text="Clara", value=1, variable=self.cluster_type, command=self.openAllEntries).grid(column=2, row=0, sticky=tk.W)
        self.clara_parameters = [tk.IntVar(value=3), tk.IntVar(value=2), tk.IntVar(value=5)]
        ttk.Label(cluster_frame, text="Cluster Amount:").grid(column=2, row=1)
        self.clara_eps_entry = ttk.Entry(cluster_frame, textvariable=self.clara_parameters[0], width=8, state=tk.DISABLED)
        self.clara_eps_entry.grid(column=3, row=1, pady=2)
        
        ttk.Label(cluster_frame, text="Iterations:").grid(column=2, row=2)
        self.clara_iterations_entry = ttk.Entry(cluster_frame, textvariable=self.clara_parameters[1], width=8, state=tk.DISABLED)
        self.clara_iterations_entry.grid(column=3, row=2, pady=2)
        
        ttk.Label(cluster_frame, text="Max Neighbors:").grid(column=2, row=3)
        self.clara_neighbors_entry = ttk.Entry(cluster_frame, textvariable=self.clara_parameters[2], width=8, state=tk.DISABLED)
        self.clara_neighbors_entry.grid(column=3, row=3, pady=2)

        # Model
        model_frame = ttk.Labelframe(self.root, text="Model Frame")
        model_frame.grid(column=1, row=1)

        ## Parameter Optimization
        parameter_optimization_frame = ttk.Labelframe(model_frame, text="Parameter Optimization")
        parameter_optimization_frame.grid(column=0, row=0)

        self.interval_var = tk.IntVar(value=3)
        ttk.Label(parameter_optimization_frame, text="Grid Search Interval:").grid(column=0, row=0)
        ttk.Entry(parameter_optimization_frame, textvariable=self.interval_var, width=8).grid(column=1, row=0, pady=2)

        self.gs_cross_val_option = tk.IntVar(value=0)
        self.gs_cross_val_var = tk.IntVar(value=5)
        tk.Checkbutton(parameter_optimization_frame, text="Cross validate; folds:", offvalue=0, onvalue=1, variable=self.gs_cross_val_option, command=self.openAllEntries).grid(column=0, row=1)
        self.gs_cross_val_entry = tk.Entry(parameter_optimization_frame, textvariable=self.gs_cross_val_var, state=tk.DISABLED, width=8)
        self.gs_cross_val_entry.grid(column=1, row=1)

        ## Model Parameters
        model_parameters_frame = ttk.LabelFrame(model_frame, text="Model Parameters")
        model_parameters_frame.grid(column=1, row=0, rowspan=3, columnspan=2)

        parameter_names = ["Max Depth", "Iterations", "Learning Rate"]
        self.optimization_parameters = [[tk.IntVar(value=5), tk.IntVar(value=25)], [tk.IntVar(value=1), tk.IntVar(value=4)], [tk.DoubleVar(value=0.1), tk.DoubleVar(value=1)]]
        ttk.Label(model_parameters_frame, text="----- Search Range -----").grid(column=1, row=0, columnspan=2)

        self.model_parameters_frame_options = [
            [
                ttk.Label(model_parameters_frame, text=j+":").grid(column=0, row=i+1),
                ttk.Entry(model_parameters_frame, textvariable=self.optimization_parameters[i][0], width=9).grid(column=1, row=i+1, padx=2, pady=2),
                ttk.Entry(model_parameters_frame, textvariable=self.optimization_parameters[i][1], width=9).grid(column=2, row=i+1, padx=2, pady=2)
            ] for i,j in enumerate(parameter_names)
        ]

        ttk.Button(model_frame, text="Create Model", command=self.createModel).grid(column=0, row=3)
        ttk.Button(model_frame, text="Save Model", command=self.saveModel).grid(column=1, row=3)
        ttk.Button(model_frame, text="Load Model", command=self.loadModel).grid(column=2, row=3)

        # Test Model Frame
        test_model_main_frame = ttk.LabelFrame(self.root, text="Test Model")
        test_model_main_frame.grid(column=0, row=1)

        self.forecast_num = tk.IntVar(value="") # type: ignore
        ttk.Label(test_model_main_frame, text="# of Forecast").grid(column=0, row=0)
        ttk.Entry(test_model_main_frame, textvariable=self.forecast_num).grid(column=1, row=0)

        test_file_path = tk.StringVar()
        ttk.Label(test_model_main_frame, text="Test File Path").grid(column=0, row=1)
        ttk.Entry(test_model_main_frame, textvariable=test_file_path).grid(column=1, row=1)
        ttk.Button(test_model_main_frame, text="Get Test Set", command=lambda: self.getTestSet(test_file_path)).grid(column=2, row=1)


        # Metrics Frame
        test_model_metrics_frame = ttk.LabelFrame(self.root, text="Test Metrics")
        test_model_metrics_frame.grid(column=1, row=2)
        
        ttk.Button(test_model_metrics_frame, text="Values", command=self.showPredicts).grid(column=0, row=1)
        ttk.Button(test_model_metrics_frame, text="Actual vs Forecast Graph", command=self.vsGraph).grid(column=0, row=2)

        test_metrics = ["NMSE", "RMSE", "MAE", "MAPE", "SMAPE"]
        self.test_metrics_vars = [tk.Variable() for _ in range(len(test_metrics))]
        for i, j in enumerate(test_metrics):
            ttk.Label(test_model_metrics_frame, text=j).grid(column=1, row=i)
            ttk.Entry(test_model_metrics_frame, textvariable=self.test_metrics_vars[i]).grid(column=2,row=i)

        self.openAllEntries()

    def readCsv(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Xlsx Files", "*.xlsx"), ("Xlrd Files", ".xls")])
        if not path:
            return
        file_path.set(path)
        if path.endswith(".csv"):
            self.df = pd.read_csv(path) # type: ignore
        else:
            try:
                self.df = pd.read_excel(path)
            except:
                self.df = pd.read_excel(path, engine="openpyxl")
        self.fillInputList()
        
    def fillInputList(self):
        self.input_list.delete(0, tk.END)

        self.df: pd.DataFrame
        for i in self.df.columns.to_list():
            self.input_list.insert(tk.END, i)

    def getTestSet(self, file_path):
        path = filedialog.askopenfilename(filetypes=[("Csv Files", "*.csv"), ("Xlsx Files", "*.xlsx"), ("Xlrd Files", ".xls")])
        if not path:
            return
        file_path.set(path)
        if path.endswith(".csv"):
            self.test_df = pd.read_csv(path) # type: ignore
        else:
            try:
                self.test_df = pd.read_excel(path)
            except:
                self.test_df = pd.read_excel(path, engine="openpyxl")
        
    def mrmrFeatureSelection(self, X, y):
        selected_features = mrmr_regression(X=X, y=y, K=self.mrmr_count.get())
        self.predictor_list.delete(0, tk.END)
        for i in selected_features:
            self.predictor_list.insert(tk.END, i)
        self.predictor_names = list(self.predictor_list.get(0, tk.END))
        return X[selected_features]

    def showPredicts(self):
        try:
            df = pd.DataFrame({"Test": self.y_test, "Predict": self.pred})
        except:
            return
        top = tk.Toplevel(self.root)
        pt = Table(top, dataframe=df, editable=False)
        pt.show()

    def addPredictor(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if a not in self.predictor_list.get(0,tk.END):
                self.predictor_list.insert(tk.END, a)
        except:
            pass

    def ejectPredictor(self, _=None):
        try:
            self.predictor_list.delete(self.predictor_list.curselection())
        except:
            pass
    
    def addTarget(self, _=None):
        try:
            a = self.input_list.get(self.input_list.curselection())
            if self.target_list.size() < 1:
                self.target_list.insert(tk.END, a)
        except:
            pass

    def ejectTarget(self, _=None):
        try:
            self.target_list.delete(self.target_list.curselection())
        except:
            pass

    def saveModel(self):
        path = filedialog.asksaveasfilename()
        if not path:
            return

        params = {}
        params["predictor_names"] = self.predictor_names
        params["label_name"] = self.label_name
        params["is_round"] = self.is_round
        params["is_negative"] = self.is_negative
        params["scale_type"] = self.scale_var.get()
        
        params["anomaly_option"] = self.anomaly_option.get()
        if self.anomaly_option.get():
            params["anomaly_n_estimator"] = self.anomaly_n_estimator.get()
        
        params["fs_option"] = self.fs_option.get()
        if self.fs_option.get():
            params["mrmr_count"] = self.mrmr_count.get()
        
        params["cluster_type"] = self.cluster_type.get()
        params["dbscan_eps"] = self.dbscan_parameters[0].get()
        params["dbscan_neighbors"] = self.dbscan_parameters[1].get()
        
        params["clara_cluster_amount"] = self.clara_parameters[0].get()
        params["clara_iterations"] = self.clara_parameters[1].get()
        params["clara_neighbors"] = self.clara_parameters[2].get()

        os.mkdir(path)
        with open(path+"/pred.npy", "wb") as outfile:
            np.save(outfile, self.pred)
        if self.scale_var.get() != "None":
            with open(path+"/feature_scaler.pkl", "wb") as f:
                pickle_dump(self.feature_scaler, f)
            with open(path+"/label_scaler.pkl", "wb") as f:
                pickle_dump(self.label_scaler, f)
        with open(path+"/model.json", 'w') as outfile:
            json.dump(params, outfile)

    def loadModel(self):
        path = filedialog.askdirectory()
        if not path:
            return
        try:
            with open(path+"/pred.npy", "rb") as f:
                self.pred = np.load(f)
        except Exception:
            popupmsg("There is no save file at the path")
            return
        

        infile = open(path+"/model.json")
        params = json.load(infile)

        self.predictor_names = params["predictor_names"]
        self.label_name = params["label_name"]
        self.is_round = params.get("is_round", True)
        self.is_negative = params.get("is_negative", False)
        self.scale_var.set(params["scale_type"])
        if params["scale_type"] != "None":
            try:
                with open(path+"/feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle_load(f)
                with open(path+"/label_scaler.pkl", "rb") as f:
                    self.label_scaler = pickle_load(f)
            except Exception:
                pass
        
        try:
            self.y_test = self.test_df[self.label_name][:len(self.pred)].to_numpy().reshape(-1) # type: ignore
            losses = loss(self.y_test, self.pred)
            for i in range(len(self.test_metrics_vars)):
                self.test_metrics_vars[i].set(losses[i])
        except Exception:
            popupmsg("Please read a test file first")
            return
        
        self.anomaly_option.set(params["anomaly_option"])
        if params["anomaly_option"]:
            self.anomaly_n_estimator.set(params["anomaly_n_estimator"])
        
        self.fs_option.set(params["fs_option"])
        if params["fs_option"]:
            self.mrmr_count.set(params["mrmr_count"])
        
        self.cluster_type.set(params["cluster_type"])
        self.dbscan_parameters[0].set(params["dbscan_eps"])
        self.dbscan_parameters[1].set(params["dbscan_neighbors"])
        
        self.clara_parameters[0].set(params["clara_cluster_amount"])
        self.clara_parameters[1].set(params["clara_iterations"])
        self.clara_parameters[2].set(params["clara_neighbors"])
       
        self.openEntries()
        self.openOtherEntries()
        msg = f"Predictor names are {self.predictor_names}\nLabel name is {self.label_name}"
        popupmsg(msg)

    def openEntries(self):
        self.gs_cross_val_entry["state"] = tk.DISABLED

        if self.gs_cross_val_option.get():
            self.gs_cross_val_entry["state"] = tk.NORMAL
    
    def openOtherEntries(self):
        cluster_elements = [self.dbscan_eps_entry, self.dbscan_neighbors_entry, self.clara_eps_entry, self.clara_iterations_entry, self.clara_neighbors_entry]
        for i in cluster_elements:
            i["state"] = tk.NORMAL
        
        self.anomaly_n_estimator_entry["state"] = tk.NORMAL if self.anomaly_option.get() else tk.DISABLED
        self.fs_entry["state"] = tk.NORMAL if self.fs_option.get() else tk.DISABLED

        cluster_elements = [self.dbscan_eps_entry, self.dbscan_neighbors_entry, self.clara_eps_entry, self.clara_iterations_entry, self.clara_neighbors_entry]
        if not self.cluster_type.get():
            for i in cluster_elements[:2]:
                i["state"] = tk.NORMAL
            for i in cluster_elements[2:]:
                i["state"] = tk.DISABLED
        if self.cluster_type.get() == 1:
            for i in cluster_elements[:2]:
                i["state"] = tk.DISABLED
            for i in cluster_elements[2:]:
                i["state"] = tk.NORMAL

    def openAllEntries(self):
        self.openOtherEntries()
        self.openEntries()

    def checkErrors(self):
        try:
            msg = "Read a data first"
            self.df.head(1)

            msg = "Select predictors"
            if not self.predictor_list.get(0):
                raise Exception
            
            msg = "Select a target"
            if not self.target_list.get(0):
                raise Exception

            msg = "Target and predictor have same variable"
            if self.target_list.get(0) in self.predictor_list.get(0, tk.END):
                raise Exception

            msg = "Enter a valid Interval for grid search"
            if self.interval_var.get() < 1:
                raise Exception
        
            msg = "Enter a valid Cross Validation fold for grid search (Above 2)"
            if self.gs_cross_val_option.get() and self.gs_cross_val_var.get() < 2:
                raise Exception

        except Exception:
            popupmsg(msg) # type: ignore
            return True

    def getNonAnomaly(self, X, y):
        if self.anomaly_option.get() == 1:
            self.anon_model = IsolationForest(n_estimators=self.anomaly_n_estimator.get(), random_state=0)
            ids = np.where(self.anon_model.fit_predict(X.values) != -1)[0] 
            X, y = X.iloc[ids], y.iloc[ids]
            return X, y
        return X, y

    def getData(self):
        self.is_round = False
        self.is_negative = False
        scale_choice = self.scale_var.get()

        self.predictor_names = list(self.predictor_list.get(0, tk.END))
        self.label_name = self.target_list.get(0)

        self.df: pd.DataFrame
        X = self.df[self.predictor_names].copy()
        y = self.df[self.label_name].copy()
        
        X, y = self.getNonAnomaly(X, y)
        if self.fs_option.get():
            X = self.mrmrFeatureSelection(X, y)
        
        if y.dtype == int or y.dtype == np.intc or y.dtype == np.int64:
            self.is_round = True
        if any(y < 0):
            self.is_negative = True

        if scale_choice == "StandardScaler":
            self.feature_scaler = StandardScaler()
            self.label_scaler = StandardScaler()
        
            X.iloc[:] = self.feature_scaler.fit_transform(X)
            y.iloc[:] = self.label_scaler.fit_transform(y.to_numpy().reshape(-1,1)).reshape(-1)
        
        elif scale_choice == "MinMaxScaler":
            self.feature_scaler = MinMaxScaler()
            self.label_scaler = MinMaxScaler()
            
            X.iloc[:] = self.feature_scaler.fit_transform(X)
            y.iloc[:] = self.label_scaler.fit_transform(y.to_numpy().reshape(-1,1)).reshape(-1)
        
        return X.reset_index(drop=True), y.reset_index(drop=True)
    
    def getClusterIds(self, X):
        if not self.cluster_type.get():
            ids = dbscan(X, self.dbscan_parameters[0].get(), self.dbscan_parameters[1].get()).process().get_clusters()
        else:
            ids = clarans(X, self.clara_parameters[0].get(), self.clara_parameters[1].get(), self.clara_parameters[2].get()).process().get_clusters()
        return ids

    def integrateTestSet(self, X_train):
        try:
            X_test = self.test_df[self.predictor_names].copy()
        except Exception:
            popupmsg("Read a test data")
            return
        if self.scale_var.get() != "None":
            X_test.iloc[:] = self.feature_scaler.transform(X_test)

        X_test["train"] = 0
        X_train["train"] = 1 
        XX = pd.concat([X_train, X_test], ignore_index=True).reset_index(drop=True)
        return XX

    def createModel(self):
        if self.checkErrors():
            return
        
        X, y = self.getData()
        XX = self.integrateTestSet(X.copy()) # type: pd.DataFrame
        cluster_ids = self.getClusterIds(XX.drop("train", axis=1).values)

        params = {}
        interval = self.interval_var.get()
        params["max_depth"] = np.unique(np.linspace(self.optimization_parameters[0][0].get(), self.optimization_parameters[0][1].get(), interval, dtype=int))
        params["iterations"] = np.unique(np.linspace(self.optimization_parameters[1][0].get(), self.optimization_parameters[1][1].get(), interval, dtype=int))
        params["learning_rate"] = np.unique(np.linspace(self.optimization_parameters[2][0].get(), self.optimization_parameters[2][1].get(), interval, dtype=float))
        cv = self.gs_cross_val_var.get() if self.gs_cross_val_option.get() == 1 else None

        indexes = []
        pred = []

        for ids in cluster_ids: # type: ignore
            slice = XX.iloc[ids]
            X_test = slice.loc[slice["train"] == 0].drop("train", axis=1)

            if X_test.empty:
                print("Continue")
                continue

            X_train = slice.loc[slice["train"] == 1].drop("train", axis=1)
            y_train = y.iloc[X_train.index]
            if len(X_train) < 5:
                popupmsg("Invalid clustering occured please, rerun with different clustering parameters")
                return

            regressor = GridSearchCV(CatBoostRegressor(allow_writing_files=False, logging_level="Silent"), params, cv=cv)
            # regressor = CatBoostRegressor(allow_writing_files=False, logging_level="Silent")
            regressor.fit(X_train, y_train)
            predict = regressor.predict(X_test)
            indexes.extend(X_test.index.tolist()) 
            pred.extend(predict)

        if len(pred) == 0:
            popupmsg("Invalid clustering occured please, rerun with different clustering parameters")
            return
    
        self.pred = pd.DataFrame(index=indexes, data=pred).sort_index(inplace=False).values.reshape(-1)
        try:
            num = self.forecast_num.get()
        except Exception:
            popupmsg("Enter a valid forecast value")
            return

        self.pred = self.pred[:num]
        
        try:
            y_test = self.test_df[self.label_name][:num].to_numpy().reshape(-1) # type: ignore
            self.y_test = y_test.copy()
        except Exception:
            popupmsg("Read a test data")
            return

        if self.scale_var.get() != "None":
            self.pred = self.label_scaler.inverse_transform(self.pred.reshape(-1,1)).reshape(-1) # type: ignore

        if not self.is_negative:
            self.pred = self.pred.clip(0, None)
        if self.is_round:
            self.pred = np.round(self.pred).astype(int)
        
        losses = loss(y_test, self.pred)
        for i in range(len(self.test_metrics_vars)):
            self.test_metrics_vars[i].set(losses[i])

    def vsGraph(self):
        y_test = self.y_test
        try:
            pred = self.pred
        except:
            return
        plt.plot(y_test)
        plt.plot(pred)
        plt.legend(["test", "pred"], loc="upper left")
        plt.show()
