import scipy.stats as stats
from scipy.stats import norm
import statistics as st
import unittest
import numpy as np
import matplotlib.pyplot as plt

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
       
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections
   
    def hit_rate(self): 
        total_signals = self.hits + self.misses
        H = self.hits / total_signals if total_signals > 0 else 0.0001
        return min(max(H, 0.0001), 0.9999)
   
    def false_alarm_rate(self):
        """Calculate flase alarm rate (FA)"""
        total_noises = self.false_alarms + self.correct_rejections
        FA = self.false_alarms / total_noises if total_noises > 0 else 0.0001
        return min(max(FA, 0.0001), 0.9999) 

    def d_prime(self):
        Hr = self.hit_rate()
        FA = self.false_alarm_rate()
        return norm.ppf(Hr) - norm.ppf(FA)
    
    def criterion(self):
        Hr = self.hit_rate()
        FA = self.false_alarm_rate()
        return -0.5 * (norm.ppf(Hr) + norm.ppf(FA))
