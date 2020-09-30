#!/usr/bin/env python

import numpy as np  # for matrix calculations
import math # for pi and sqrt
import glob # for sensible listdir()
import ceflib # for reading CEF files
from copy import deepcopy # for obtaining variables in CEF files
import matplotlib.pyplot as plt # for plotting
import datetime as dt # for dates
from matplotlib import dates # for formatting axes

# User-defined variables:

# Path to data:
path = r'../Data/FGM_5VPS__20010611_200500_20010611_202300/'

# Filename for output current density:
outfile = '/Users/Hmiddleton/Documents/Cluster/Projects/\
InProgress/Curlometer/Data/200106112005-2023_J.txt'

# Plot filenames:
BJQFileName = '20010611_2005-2023_BJQ.png'
GeomFileName = '20010611_2005-2023_Geom.png'

# X-axis labels:
XAxisLabel = 'Time on 11th June 2001'

# Desired resolution of data for the curlometer calculation
window = 0.2 # in seconds, 0.2 = minimum


'''The Curlometer Function'''
def delta(ref, i):
    delrefi = i - ref
    return delrefi

def curlometer(d1, d2, d3, d4):
    
    km2m = 1e3
    nT2T = 1e-9
    mu0 = (4*math.pi)*1e-7
    
    C1R = np.array([d1[3], d1[4], d1[5]])*km2m
    C1B = np.array([d1[0], d1[1], d1[2]])*nT2T
    C2R = np.array([d2[3], d2[4], d2[5]])*km2m
    C2B = np.array([d2[0], d2[1], d2[2]])*nT2T
    C3R = np.array([d3[3], d3[4], d3[5]])*km2m
    C3B = np.array([d3[0], d3[1], d3[2]])*nT2T
    C4R = np.array([d4[3], d4[4], d4[5]])*km2m
    C4B = np.array([d4[0], d4[1], d4[2]])*nT2T
    
    delB14 = delta(C4B, C1B)
    delB24 = delta(C4B, C2B)
    delB34 = delta(C4B, C3B)
    delR14 = delta(C4R, C1R)
    delR24 = delta(C4R, C2R)
    delR34 = delta(C4R, C3R)

# J

    # Have to 'convert' this to a matrix to be able to get the inverse.
    R = np.matrix(([np.cross(delR14, delR24), np.cross(delR24, delR34),
         np.cross(delR14, delR34)]))
    Rinv = R.I

    # I(average) matrix:
    Iave = ([np.dot(delB14, delR24) - np.dot(delB24, delR14)],
        [np.dot(delB24, delR34) - np.dot(delB34, delR24)],
        [np.dot(delB14, delR34) - np.dot(delB34, delR14)])

    JJ = (Rinv*Iave)/mu0
                  
# div B
    lhs = np.dot(delR14, np.cross(delR24, delR34))

    rhs = np.dot(delB14, np.cross(delR24, delR34)) + \
        np.dot(delB24, np.cross(delR34, delR14)) + \
        np.dot(delB34, np.cross(delR14, delR24))

    divB = abs(rhs)/abs(lhs)

# div B / curl B
    curlB = JJ*mu0
    magcurlB = math.sqrt(curlB[0]**2 + curlB[1]**2 + curlB[2]**2)
    divBbycurlB = divB/magcurlB

    return [JJ, divB, divBbycurlB]
# End of curlometer function


'''Read in all the data using CEFLIB.read '''

cluster = ['C'+str(x) for x in range(1,5)]

time = {}
B = {}
pos = {}

for c in cluster:
    folder = c+'_CP_FGM_5VPS/*.cef'
    filename = glob.glob(path+folder)
    print(filename)
    ceflib.read(filename[0])
    time[c] = deepcopy(ceflib.var('time_tags')) # in milli-seconds
    B[c] = deepcopy(ceflib.var('B_vec_xyz_gse'))
    pos[c] = deepcopy(ceflib.var('sc_pos_xyz_gse'))
    ceflib.close()

'''Align all the data with the time by using a dictionary with 
the time in milliseconds as the key'''
clean = {}
for c in cluster:
    for i,p in enumerate(time[c]):
        if p not in clean.keys():
            clean[int(p)] = {}
        clean[p][c] = [B[c][i][0], 
                       B[c][i][1],
                       B[c][i][2],
                       pos[c][i][0],
                       pos[c][i][1],
                       pos[c][i][2]]

mintime, maxtime = min(clean.keys()), max(clean.keys())

# Time array (min, max, step)
tarr = range(mintime, maxtime, int(window*1000))
nwin = len(tarr)

Jave = np.zeros(nwin, dtype = [('time', float),('Jx', float),
                                  ('Jy', float),('Jz', float),
                                  ('divB', float), 
                                  ('divBcurlB', float)])

for i,t in enumerate(tarr):

    if len(clean[t]) == 4:
        onej = curlometer(clean[t]['C1'],clean[t]['C2'],
                         clean[t]['C3'],clean[t]['C4'])

        Jave['time'][i] = t/1000
        Jave['Jx'][i] = onej[0][0]
        Jave['Jy'][i] = onej[0][1]
        Jave['Jz'][i] = onej[0][2]
        Jave['divB'][i] = onej[1]
        Jave['divBcurlB'][i] = onej[2]
    else:
        Jave['time'][i] = t/1000
        Jave['Jx'][i] = np.nan
        Jave['Jy'][i] = np.nan
        Jave['Jz'][i] = np.nan
        Jave['divB'][i] = np.nan
        Jave['divBcurlB'][i] = np.nan

'''Write all results out to file, tarr is already sorted'''

with open(outfile, 'w') as f:
    for j in Jave:
        outstring = str(dt.datetime.utcfromtimestamp(j['time'])) + \
        ', ' + str(j['Jx']) + ', ' + str(j['Jy']) + \
        ', ' + str(j['Jz']) +', '+ str(j['divBcurlB'])+'\n'
        f.write(outstring)

'''Pull out the mag field used for the calculation'''            
Magnpt = {}
for c in cluster:
    Bx, By, Bz, Bmag = [], [], [], []
    for p in tarr:
        if c in clean[p].keys():
            Bx.append(clean[p][c][0])
            By.append(clean[p][c][1])
            Bz.append(clean[p][c][2])
            Bmag.append(math.sqrt(clean[p][c][0]**2 + clean[p][c][1]**2 + clean[p][c][2]**2))
        else:
            Bx.append(np.nan)
            By.append(np.nan)
            Bz.append(np.nan)
            Bmag.append(np.nan)
    Magnpt[c] = [Bx, By, Bz, Bmag]

'''Take times and put as date into list'''
tdate = []
for t in tarr:
    tdate.append(dates.date2num(dt.datetime.utcfromtimestamp(t/1000)))

    
'''Plot the B field and the current density and divB/curlB'''
fig = plt.figure(figsize=(8.5, 12))

hfmt = dates.DateFormatter('%H:%M')
minutes = dates.MinuteLocator(interval=2)

sub1=fig.add_subplot(811)
plt.plot(tdate, Magnpt['C1'][2], color='black', linestyle='-', label = 'C1')
plt.plot(tdate, Magnpt['C2'][2], color='red', linestyle='-', label = 'C2')
plt.plot(tdate, Magnpt['C3'][2], color='green', linestyle='-', label = 'C3')
plt.plot(tdate, Magnpt['C4'][2], color='blue', linestyle='-', label = 'C4')
plt.ylim(-20, 30)
plt.yticks([-20,-10,0,10,20,30]) 
plt.ylabel('Bz (nT)')
sub1.xaxis.set_major_locator(minutes)
sub1.xaxis.set_major_formatter(hfmt)
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize='small')

sub2=fig.add_subplot(812)
plt.plot(tdate, Magnpt['C1'][1], color='black', linestyle='-' )
plt.plot(tdate, Magnpt['C2'][1], color='red', linestyle='-' )
plt.plot(tdate, Magnpt['C3'][1], color='green', linestyle='-' )
plt.plot(tdate, Magnpt['C4'][1], color='blue', linestyle='-' )
plt.ylim(-20, 20)
plt.yticks([-20, -10, 0, 10, 20])
plt.ylabel('By (nT)')
sub2.xaxis.set_major_locator(minutes)
sub2.xaxis.set_major_formatter(hfmt)

sub3=fig.add_subplot(813)
plt.plot(tdate, Magnpt['C1'][0], color='black', linestyle='-' )
plt.plot(tdate, Magnpt['C2'][0], color='red', linestyle='-' )
plt.plot(tdate, Magnpt['C3'][0], color='green', linestyle='-' )
plt.plot(tdate, Magnpt['C4'][0], color='blue', linestyle='-' )
plt.ylim(-20, 20)
plt.yticks([-20, -10, 0, 10, 20])
plt.ylabel('Bx (nT)')
sub3.xaxis.set_major_locator(minutes)
sub3.xaxis.set_major_formatter(hfmt)

sub4=fig.add_subplot(814)
plt.plot(tdate, Magnpt['C1'][3], color='black', linestyle='-' )
plt.plot(tdate, Magnpt['C2'][3], color='red', linestyle='-' )
plt.plot(tdate, Magnpt['C3'][3], color='green', linestyle='-' )
plt.plot(tdate, Magnpt['C4'][3], color='blue', linestyle='-' )
plt.ylim(0, 30)
plt.yticks([0, 10, 20, 30])
plt.ylabel('|B| (nT)')
sub4.xaxis.set_major_locator(minutes)
sub4.xaxis.set_major_formatter(hfmt)

sub5=fig.add_subplot(815)
plt.plot(tdate, Jave['Jz']*1e9, color='black', linestyle='-' )
plt.ylim(-15, 10)
plt.yticks([-15, -10, -5, 0, 5, 10])
plt.ylabel(r'$\mathbf{J}_Z (nA/m^2)$')
sub5.xaxis.set_major_locator(minutes)
sub5.xaxis.set_major_formatter(hfmt)

sub6=fig.add_subplot(816)
plt.plot(tdate, Jave['Jy']*1e9, color='red', linestyle='-' )
plt.ylim(-20, 20)
plt.yticks([-20, -10, 0, 10, 20])
plt.ylabel(r'$\mathbf{J}_Y (nA/m^2)$')
sub6.xaxis.set_major_locator(minutes)
sub6.xaxis.set_major_formatter(hfmt)

sub7=fig.add_subplot(817)
plt.plot(tdate, Jave['Jx']*1e9, color='green', linestyle='-' )
plt.ylim(-10, 20)
plt.yticks([-10, 0, 10, 20])
plt.ylabel(r'$\mathbf{J}_X (nA/m^2)$')
sub7.xaxis.set_major_locator(minutes)
sub7.xaxis.set_major_formatter(hfmt)

sub8=fig.add_subplot(818)
plt.plot(tdate, Jave['divBcurlB'], color='blue', linestyle='-' )
plt.ylim(0, 2)
plt.yticks([0, 1, 2])
plt.ylabel(r'$|{\rm {div}}\,\mathbf{B}|/|{\rm {curl}}\,\mathbf{B}|$')
sub8.xaxis.set_major_locator(minutes)
sub8.xaxis.set_major_formatter(hfmt)
plt.xlabel(XAxisLabel)
plt.savefig(BJQFileName, dpi=300)
#plt.show()
plt.close()

'''Read in the geometry data'''

all_data = {}
folder = 'CL_SP_AUX/*.cef'
filename = glob.glob(path+folder)
ceflib.read(filename[0])

time = deepcopy(ceflib.var('time_tags')) # in milli-seconds
QG = deepcopy(ceflib.var('sc_config_QG'))
QR = deepcopy(ceflib.var('sc_config_QR'))
E = deepcopy(ceflib.var('sc_geom_elong'))
P = deepcopy(ceflib.var('sc_geom_planarity'))
ceflib.close()


tgdate = []
for t in time:
    tgdate.append(dates.date2num(dt.datetime.utcfromtimestamp(t/1000)))

gminutes = dates.MinuteLocator(interval=2)

'''Plot the geometrical parameters'''
fig = plt.figure(figsize=(8, 5))

gsub1 = fig.add_subplot(211)
plt.plot(tgdate, P, label = 'Planarity')
plt.plot(tgdate, E, label = 'Elongation')
plt.legend(bbox_to_anchor=(0.4, 0.8), loc=2, borderaxespad=0., fontsize='small')
gsub1.xaxis.set_major_locator(gminutes)
gsub1.xaxis.set_major_formatter(hfmt)

gsub2 = fig.add_subplot(212)
plt.plot(tgdate, QR, label = r'$Q_R$')
plt.plot(tgdate, QG, label = r'$Q_G$')
plt.xlabel('Time on 4th February 2001')
plt.ylim(0, 3.5)
gsub2.xaxis.set_major_locator(gminutes)
gsub2.xaxis.set_major_formatter(hfmt)
plt.legend(bbox_to_anchor=(0.4, 0.75), loc=2, borderaxespad=0., fontsize='small', ncol=2)
plt.savefig(GeomFileName, dpi=300)

#plt.show()
plt.close()
