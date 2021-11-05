# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:29:22 2020

@author: Callum Pitceathly
"""
import numpy as np
import matplotlib.pyplot as plt
dr_deuts = [217, 280, 293, 266, 241, 198, 184, 158, 81]
dr_antis = [189, 279, 273, 264, 232, 172, 135, 103, 77]
coal_deuts = [303, 367, 340, 336, 234, 155, 93, 91, 100]
coal_antis = [232, 296, 297, 236, 193, 132, 98, 53, 72]

binned2 = [0.03343746, 0.3276133 , 0.53581644, 0.74293189, 0.96569554,
       1.22347304, 1.5275202 , 1.84633554, 2.21873567, 2.98763174]

bin_wid = []
for j in range(0,9):
    bin_wid.append(binned2[j+1]-binned2[j])

x = []

for i in range(0,9):
    x.append(0.5*(binned2[i+1]+binned2[i]))

for k in range(0,9):
    dr_deuts[k] = dr_deuts[k] * (2E-4/bin_wid[k])
    dr_antis[k] = dr_antis[k] * (2E-4/bin_wid[k])
    coal_deuts[k] = coal_deuts[k] * (2E-4/bin_wid[k])
    coal_antis[k] = coal_antis[k] * (2E-4/bin_wid[k])


ax = plt.gca()
ax.scatter(x, dr_deuts, color='red', marker='x', label = 'Dal-Raklev Deuterons')
ax.scatter(x, dr_antis, color='blue', marker='x', label = 'Dal-Raklev Antideuterons')
ax.scatter(x, coal_deuts, color='green', marker='x', label = 'Coalescence Deuterons')
ax.scatter(x, coal_antis, color='purple', marker='x', label = 'Coalescence Antideuterons')

ax.set_xlabel(r'$P_{T}$ (Gev)')
ax.set_ylabel(r'$d\sigma/dP_{T}$ (mb/GeV)')


ax.legend()
plt.show