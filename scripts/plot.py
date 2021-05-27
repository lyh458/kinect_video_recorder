#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt

with open('frames.json') as f:
    data=json.load(f)

object = 'right_gripper'
time = True

x = []
y = []
z = []
times = []

for row in data['transforms']:
    if row['objects'][object]['visible']:
        x.append(row['objects'][object]['pose'][0][0])
        y.append(row['objects'][object]['pose'][0][1])
        z.append(row['objects'][object]['pose'][0][2])
        times.append(row['time'])

plt.plot(times if time else len(x), x)
plt.plot(times if time else len(y), y)
plt.plot(times if time else len(z), z)

plt.show()

