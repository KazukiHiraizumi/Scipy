# Scipy

## 点群から平面を求める
### コード

~~~
# Calcurates the equation of plane as 
#    ax+by+cz+d=0
# from Point clouds

import open3d as o3d
import numpy as np
from scipy import optimize

def error(param,data):
  den=np.linalg.norm(param[:3])
  vars=np.vstack((data.T,np.ones(len(data))))
  return np.abs(param.dot(vars))/den

######Making input data
mesh=o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=0.001)
pcd=mesh.sample_points_uniformly(number_of_points=10000)
points=np.array(pcd.points)
noise=np.random.normal(0,1,(len(points),3))
points=points+noise
pcd.points=o3d.utility.Vector3dVector(points)

#o3d.visualization.draw_geometries([pcd])
param=np.array([1,1,1,1])
result=optimize.leastsq(error,param,points)

print(result[0])
arg=result[0]/np.linalg.norm(result[0][:3])
print(arg)
~~~
