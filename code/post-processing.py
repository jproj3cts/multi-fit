import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import scipy.constants as c
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.signal import find_peaks
from scipy.signal import periodogram
from scipy.interpolate import interp1d
import scipy.signal as sig
import math
import string
import pandas as pd
import time
import json
import csv

# Gui stuff.
import tkinter
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
#plt.style.use("dark_background")

#%%
class post_processing():
    def __init__(self, channels = ['A','B','C','D'], params = [r'$\omega$',r'$\gamma$','T']):
        self.CHs = channels
        self.params = params
        self.labels = list(string.ascii_uppercase)
    
    def data_loader_Allan_var(self):
        a=1
    
    def Allan_var(self, directory):
        a=1
    
    def data_loader_fit_trends(self):
        self.files = []
        i = 0
        for f in listdir(self.dir): 
            if re.sub(r'[^a-zA-Z]', '', f) == self.il: # if dir matches iteration label
                temp = [re.sub(self.il, '', f), [[] for k in range(len(self.CHs))]]
                for g in listdir(self.dir + '\\' + f): # checking each file
                    i += 1
                    for j in range(len(self.CHs)):
                        if g == 'multifit'+self.CHs[j]+'.csv':
                            temp[1][j] = f + '\\' + g
                self.files.append(temp)
        self.data = self.files
        for i in range(len(self.files)):
            self.data[i][0] = float(self.data[i][0])
            for j in range(len(self.CHs)):
                if self.files[i][1][j] != []:
                    incoming_dataframe = pd.read_json(self.dir + '\\' + self.files[i][1][j]) 
                    incoming_list = incoming_dataframe.values.tolist()
                    self.data[i][1][j] = incoming_list
        self.data = sorted(self.data, key=lambda i: i[0])
        self.pressures = [i[0] for i in self.data]
    
    def peak_collector(self, tolerance = 3e3):
        self.pp = {} #peak pointer
        #self.pp = []
        diff = [0,0,0]
        for i in range(len(self.CHs)):
            self.pp[self.CHs[i]] = []
            for k in range(len(self.data)):
                if k != []:
                    a = k
                    for j in range(len(self.data[k][1][i][0][1])):
                        self.pp[self.CHs[i]].append((0,i,j))
                    break
            for j in range(a+1,len(self.data)):
                for k in range(len(self.data[i][1][j][0][1])): #peak loop [0][1] goes into the fit params and into the peak vals
                    for l in range(len(self.pp)): #checking if it matches other peaks
                        if (self.data[j][1][i][0][1][k][0] - tolerance <= self.data[self.pp[i][l][-1][0]][1][self.pp[i][l][-1][1]][0][1][self.pp[i][l][-1][2]][0] 
                            and self.data[j][1][i][0][1][k][0] +tolerance >= self.data[self.pp[i][l][-1][0]][1][self.pp[i][l][-1][1]][0][1][self.pp[i][l][-1][2]][0] and 
                            j!=self.pp[i][l][-1][0]):
                            
                            diff = [0,0,0]
                            for m in [-1,0,1]: #check to see if peak is closer to a different neighbour
                                diff[m+1] = abs(self.data[j][1][i][0][1][k][0] - self.data[self.pp[(l+m)%len(self.pp[i])][-1][0]][1][self.pp[i][(l+m)%len(self.pp[i])][-1][1]][0][1][self.pp[i][(l+m)%len(self.pp[i])][-1][2]][0])
                            idx = min(range(len(diff)), key=diff.__getitem__)
                            if diff[idx] !=0.0:
                                self.pp[l+idx-1].append([j,i,k])
                                break
                        elif self.data[j][1][i][0][1][k][0] > self.data[self.pp[i][l][-1][0]][1][self.pp[i][l][-1][1]][0][1][self.pp[i][l][-1][2]][0]:
                            self.pp[i].insert(l, [[j,i,k]])
                            break
        
        
        print(self.pp)
        
        #for i in range(len(self.data)): #sweep param
        #    for j in range(len(self.CHs)): #channel
        #        if self.data[i][1][j] != []:
        #            for k in range(len(self.data[i][1][j][0][1])): #peak loop [0][1] goes into the fit params and into the peak vals
        #                if self.pp == []:
        #                    self.pp.append([[i,j,k]])
        #                else:
        #                    for l in range(len(self.pp)): #checking if it matches other peaks
        #                        if (self.data[i][1][j][0][1][k][0] - tolerance <= self.data[self.pp[l][-1][0]][1][self.pp[l][-1][1]][0][1][self.pp[l][-1][2]][0] 
        #                            and self.data[i][1][j][0][1][k][0] +tolerance >= self.data[self.pp[l][-1][0]][1][self.pp[l][-1][1]][0][1][self.pp[l][-1][2]][0] and 
        #                            i!=self.pp[l][-1][0]):
        #                            
        #                            diff = [0,0,0]
        #                            for m in [-1,0,1]: #check to see if peak is closer to a different neighbour
        #                                diff[m+1] = abs(self.data[i][1][j][0][1][k][0] - self.data[self.pp[(l+m)%len(self.pp)][-1][0]][1][self.pp[(l+m)%len(self.pp)][-1][1]][0][1][self.pp[(l+m)%len(self.pp)][-1][2]][0])
        #                            idx = min(range(len(diff)), key=diff.__getitem__)
        #                            if diff[idx] !=0.0:
        #                                self.pp[l+idx-1].append([i,j,k])
        #                                break
        #                        elif self.data[i][1][j][0][1][k][0] > self.data[self.pp[l][-1][0]][1][self.pp[l][-1][1]][0][1][self.pp[l][-1][2]][0]:
        #                            self.pp.insert(l, [[i,j,k]])
        #                            break
        self.pp.reverse()
    
    def peak_identification(self, bounds = 1000):
        #Harmonic identification
        #bounds = 1000 #bound on identification for harmonics
        self.h_order = 1 #max order to identify, fundamental is order 0
        self.harmonic = np.zeros((len(self.pp),self.h_order))
        for i in range(len(self.pp)): #peak being identified
            for j in range(len(self.pp)): # peak being compared to
                for k in range(self.h_order):
                    if self.data[self.pp[i][-1][0]][1][self.pp[i][-1][1]][0][1][self.pp[i][-1][2]][0] >= (k+2)*self.data[self.pp[j][-1][0]][1][self.pp[j][-1][1]][0][1][self.pp[j][-1][2]][0] - bounds and self.data[self.pp[i][-1][0]][1][self.pp[i][-1][1]][0][1][self.pp[i][-1][2]][0] >= (k+2)*self.data[self.pp[j][-1][0]][1][self.pp[j][-1][1]][0][1][self.pp[j][-1][2]][0] + bounds:
                        self.harmonic[i,k] = j
        
        #Sideband identification
        bounds = 1500 #bound on identification for sidebands
        self.harm = np.sum(self.harmonic,1) # used to ignore harmonics
        self.s_order = 2 #order of sidebands
        s = [y for y in range(-1*self.s_order, self.s_order+1, 1) if y!=0]
        self.sideband = np.zeros((len(self.pp),2,2*self.s_order))
        for i in range(len(self.pp)):
            for j in range(len(self.pp)):
                for k in range(len(self.pp)):
                    for l in range(len(s)):
                        if self.data[self.pp[i][-1][0]][1][self.pp[i][-1][1]][0][1][self.pp[i][-1][2]][0] >= self.data[self.pp[j][-1][0]][1][self.pp[j][-1][1]][0][1][self.pp[j][-1][2]][0] + s[l]*self.data[self.pp[k][-1][0]][1][self.pp[k][-1][1]][0][1][self.pp[k][-1][2]][0] - bounds and self.data[self.pp[i][-1][0]][1][self.pp[i][-1][1]][0][1][self.pp[i][-1][2]][0] <= self.data[self.pp[j][-1][0]][1][self.pp[j][-1][1]][0][1][self.pp[j][-1][2]][0] + s[l]*self.data[self.pp[k][-1][0]][1][self.pp[k][-1][1]][0][1][self.pp[k][-1][2]][0] + bounds and (i!=j and i>k) and j!=k and not (i in self.harmonic) and self.harm[j] == 0 and self.harm[k] == 0: # and harm[i] == 0 
                            if self.data[self.pp[i][-1][0]][1][self.pp[i][-1][1]][0][1][self.pp[i][-1][2]][2] < self.data[self.pp[j][-1][0]][1][self.pp[j][-1][1]][0][1][self.pp[j][-1][2]][2]: #or amp_list[i] > amp_list[int(sideband[i,1,j])]:
                                self.sideband[i,0,l] = j
                                self.sideband[i,1,l] = k
        
        #removing fake sidebands/finding real peaks
        self.side = np.sum(self.sideband, 2)
        for i in range(len(self.pp)):
            for j in range(len(s)):
                if self.sideband[i,0,j] != 0:
                    if self.side[int(self.sideband[i,0,j]),0] != 0 or self.side[int(self.sideband[i,1,j]),0] != 0:
                        self.sideband[i,0,j] = 0
                        self.sideband[i,1,j] = 0
        self.side = np.sum(self.sideband, 2) #update after removals
        
        #labelling
        alph = list(string.ascii_uppercase)
        self.labels = list(alph[0:len(self.pp)])
        j = 0
        for i in range(len(self.pp)):
            if self.harm[i] == 0 and self.side[i,0] == 0:
                self.labels[i] = alph[j]
                j += 1
        for i in range(len(self.pp)):
            if self.harm[i] != 0:
                for k in range(self.h_order):
                    self.labels[i] = str(int((k+2))) + self.labels[int(self.harmonic[i,k])];
            elif self.side[i,0] != 0:
                for k in range(len(s)):
                    if self.sideband[i,0,k] != 0:
                        if s[k] < 0:
                            sign = '-' + str(int(abs(s[k])))
                        elif s[k] > 0:
                            sign = '+' + str(int(abs(s[k])))
                        if self.labels[i] != alph[i]:
                            self.labels[i] += '/' + '\n' + self.labels[int(self.sideband[i,0,k])] + sign + self.labels[int(self.sideband[i,1,k])]
                        else:
                            self.labels[i] = self.labels[int(self.sideband[i,0,k])] + sign + self.labels[int(self.sideband[i,1,k])]
    
    def repack_data(self):
        self.re_data = np.zeros((len(self.pressures),len(self.pp),len(self.CHs),3))
        self.re_err = np.zeros((len(self.pressures),len(self.pp),len(self.CHs),3))
        for i in range(len(self.pp)): #Sweep over peak groups
            for j in self.pp[i]: #Sweep over peaks in group
                for k in range(3): #Sweep over fitting parameters
                    self.re_data[j[0]][i][j[1]][k] = self.data[j[0]][1][j[1]][0][1][j[2]][k]
                    self.re_err[j[0]][i][j[1]][k] = self.data[j[0]][1][j[1]][1][1][j[2]][k]
        self.re_data[ self.re_data==0 ] = np.nan
        self.re_err[ self.re_data==0 ] = np.nan
    
    def canv(self): #To show figures without running gui
        self.fig = plt.figure()
        self.plot1 = self.fig.add_axes((0.1, 0.3, 1, 1))
        self.plot1.tick_params(axis = 'x', bottom = True, top = True, left = True, right = 'True', direction = 'in', which = 'both', colors = 'black')
        self.plot1.tick_params(axis = 'y', bottom = True, top = True, left = True, right = 'True', direction = 'in', which = 'both', colors = 'black')

    def plot_trend(self, ax, param, peak):
        for i in range(len(self.CHs)):
            ax.errorbar(self.pressures, self.re_data[:,peak,i,param], yerr = self.re_err[:,peak,i,param], marker='.', linestyle='',
                     color='black', alpha=0.7, ms = 10, linewidth = 0, elinewidth = 2,
                     capsize = 1, ecolor = 'black', zorder = 0) #plot psd
            #ax.plot(self.pressures, self.re_data[:,peak,i,param], linewidth = 0, ) #plot multifit
        ax.set_title('Peak' + str(peak))
        ax.set_xlabel('Pressure (mbar)')
        ax.set_ylabel(self.params[param])
    
    def fit_trends(self, directory, iteration_label):
        self.dir = directory
        self.il = iteration_label
        self.data_loader_fit_trends()
        self.peak_collector()
        #self.peak_identification()
        self.repack_data()
        self.canv()
        self.plot_trend(self.plot1, 0, 1)
        
#%%
post = post_processing()
directory = r'\\samba.nms.kcl.ac.uk\store\millen\OptoMech\880nm Nanorods Polarisation Feedback\20230711TempRun60mW'
post.fit_trends(directory,'mbar')
#%%
post.canv()
post.plot_trend(post.plot1,1,3)
#%%
print(post.pp)
