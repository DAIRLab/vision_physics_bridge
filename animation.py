import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.transform as transform

data_all = torch.load('data/tosses_processed/31.pt')
x = data_all.data[0,:,0].numpy()
y = data_all.data[0,:,1].numpy()
z = data_all.data[0,:,2].numpy()
q = data_all.data[0,:,3:7].numpy()
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x,y,z)

print(q.shape[0])

for i in range(0, q.shape[0], 2):

    # Mismatch between quaternion ordering in data and scipy
    R = transform.Rotation.from_quat((q[i,1], q[i,2], q[i,3], q[i,0])).as_matrix()

    # Arrow length to draw
    len = .3

    # constant transform so axes look more human interperable
    R = R @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # whether its columns/rows depends on format of input data
    delta_x = R[0,:]
    delta_y = R[1,:]
    delta_z = R[2,:]
    # delta_x = R[:,0]
    # delta_y = R[:,1]
    # delta_z = R[:,2]

    ax.quiver(x[i], y[i], z[i], delta_x[0], delta_x[1], delta_x[2], length=len, colors = [1, 0, 0])
    ax.quiver(x[i], y[i], z[i], delta_y[0], delta_y[1], delta_y[2], length=len, colors = [0, 1, 0])
    ax.quiver(x[i], y[i], z[i], delta_z[0], delta_z[1], delta_z[2], length=len, colors = [0, 0, 1])
    print(R)

ax.set_xlim3d([-4, 4])
ax.set_ylim3d([-4, 4])
ax.set_zlim3d([0, 4])

ax.view_init(15, 0)

plt.show()