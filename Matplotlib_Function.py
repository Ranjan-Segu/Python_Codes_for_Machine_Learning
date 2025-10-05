# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 20:56:27 2025

@author: Ranjan Segu
"""

import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
#### MATPLOTLIB_FUNCTION####


# Prepare_Data

import numpy as np
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)

data = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))
y, x = np.mgrid[-3:3:100j, -3:3:100j]
u = -1 - x**2 + y
v = 1 + x - y**2
img = get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)


# Create_Plot


x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)

fig = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))

fig.add_axes
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2, ncols=2)
fig4, axes2 = plt.subplots(ncols=3)


# Plotting_Routines

fig, ax = plt.subplots()
lines = ax.plot(x, y)
ax.scatter(x, y)
axes[0, 0].bar([1, 2, 3], [3, 4, 5])
axes[1, 0].barh([0.5, 1, 2.5], [0, 1, 2])
axes[1, 1].axhline(0.45)
axes[0, 1].axvline(0.65)
ax.fill(x, y, color="blue")
ax.fill_between(x, y, color="yellow")

fig, ax = plt.subplots()
img = np.random.randn(100, 100)
im = ax.imshow(img, cmap="gist_earth",
               interpolation="nearest", vmin=-2, vmax=2)

y, x = np.mgrid[-3:3:100j, -3:3:100j]
y = np.linspace(-3, 3, 100)
x = np.linspace(-3, 3, 100)
y, x = np.mgrid[-3:3:100j, -3:3:100j]
axes[0, 1].arrow(0, 0, 0.5, 0.5)
axes[1, 1].quiver(y, z)
u = -y
v = x
axes[0, 1].streamplot(x, y, u, v)

ax1.hist(y)
ax3.boxplot(y)
ax3.violinplot(z)

data = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))
axes2[0].pcolor(data)
axes2[0].pcolormesh(data)
CS = plt.contour(y, x, u)
axes2[2].contourf(data2)
axes2[2] = ax.clabel(CS)


# Plot_Anatomy

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(x, y, color = "lightblue", linewidth=3)
ax.scatter([2, 4, 6], [5, 15, 25], color = "darkgreen", marker="^")
ax.set_xlim(1, 6.5)
plt.savefig("foo.png")

plt.show()


plt.plot(x, x, x, [i**2 for i in x], x, [i**3 for i in x])
ax.plot(x, y, alpha = 0.4)
ax.plot(x, y, c="k")
fig.colorbar(im, orientation="horizontal")
im = ax.imshow(img, cmap = "seismic")

fig, ax = plt.subplots()
ax.scatter(x, y, marker=".")
ax.plot(x, y, marker = "o")
fig.savefig("foo1.png")

plt.plot(x, y, linewidth=4.0)
plt.plot(x, y, ls="solid")
plt.plot(x, y, ls="--")
plt.plot(x, y, "--", [i**2 for i in x], [j**2 for j in y], "-.")
plt.setp(lines, color='r', linewidth=4.0)

ax.text(1, -2.1, "Example Graph", style = "italic")
ax.annotate("Sine", xy = (8, 0), xycoords="data", arrowprops=dict(arrowstyle="->", connectionstyle = "arc3"))

plt.title(r"$sigma_i=15$", fontsize = 20)

ax.margins(x=0.0,y=0.1)
ax.axis("equal")
ax.set(xlim=[0, 10.5], ylim=[-1.5, 1.5])
ax.set_xlim(0, 10.5)

ax.set(title="Ranjan", ylabel="y-axis", xlabel="x-axis")
ax.legend(loc = "best")

ax.xaxis.set(ticks=range(1, 5), ticklabels=[3, 100, -12, "foo"])
ax.tick_params(axis="y", direction="inout", lenght = 10)

fig3.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
fig.tight_layout()

ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_position(("outward", 10))


plt.savefig("foo2.png")
plt.savefig("foo3.png", transparent=True)

plt.show()

plt.cla()
plt.clf()
plt.close()

