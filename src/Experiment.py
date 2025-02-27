import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid #correct function name 
from SignalDetection import SignalDetection

class Experiment:
    def __init__(self):
        self.conditions = []
        self.labels = []
    
    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        if not isinstance(sdt_obj, SignalDetection):
            raise TypeError("sdt_obj must be an instance of SignalDetection")
        
        self.conditions.append(sdt_obj)
        self.labels.append(label if label else f"Condition {len(self.conditions)}")
    
    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        if not self.conditions:
            raise ValueError("No conditions have been added to the experiment.")
        
        fa_rates = [sdt.false_alarm_rate() for sdt in self.conditions]
        hit_rates = [sdt.hit_rate() for sdt in self.conditions]
        
        sorted_indices = np.argsort(fa_rates)
        sorted_fa = [fa_rates[i] for i in sorted_indices]
        sorted_hr = [hit_rates[i] for i in sorted_indices]
        
        return sorted_fa, sorted_hr
    
    def compute_auc(self) -> float:
        sorted_fa, sorted_hr = self.sorted_roc_points()
        return trapezoid(sorted_hr, sorted_fa) #correct function name 
    
    def plot_roc_curve(self, show_plot: bool = True):
        sorted_fa, sorted_hr = self.sorted_roc_points()
        
        plt.figure(figsize=(6, 6))
        plt.plot(sorted_fa, sorted_hr, marker='o', linestyle='-', label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance Level (AUC=0.5)')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        
        if show_plot:
            plt.show()
