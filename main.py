import tkinter as tk
from tkinter import ttk

import warnings
warnings.filterwarnings("ignore")

from gui.timeseries import TimeSeries
from gui.random_forest import RandomForest
from gui.catboost_arch import CatBoost
from gui.catboost_with_clustering import CatBoostWithClustering
from gui.rf_with_clustering import RFWithClustering
from gui.sarima import SARIMA


class GUI:
    def __init__(self):
        self.gui = tk.Tk()
        self.parent = ttk.Notebook(self.gui)

        time_series = TimeSeries()
        self.add(time_series, "Time Series")
        
        rf = RandomForest()
        self.add(rf, "Random Forest")

        catboost = CatBoost()
        self.add(catboost, "CatBoost")
        
        catboostclustering = CatBoostWithClustering()
        self.add(catboostclustering, "CatBoost Clustering")
        
        rfclustering = RFWithClustering()
        self.add(rfclustering, "RandomForest Clustering")
        
        arima = SARIMA()
        self.add(arima, "Arima")

        self.parent.pack(expand=1, fill='both')

    def add(self, frame, text):
        self.parent.add(frame.root, text=text)

    def start(self):
        self.gui.mainloop()

s = GUI()
s.start()
