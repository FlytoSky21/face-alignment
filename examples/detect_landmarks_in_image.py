# coding:utf-8
import numpy as np

import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import time

# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold": 0.8
}

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

try:
    input_img_real = io.imread('../test/imgs/256/0020_01.png')
    input_img_fake256 = io.imread('../test/imgs/256/0020_01_e.png')
    input_img_fake128 = io.imread('../test/imgs/128/0001_01_e.png')
except FileNotFoundError:
    input_img = io.imread('test/assets/imgs/n000129/0004_01.png')

start_time = time.time()
# predss = fa.get_landmarks_from_directory('../test/assets/imgs/n000009/')

preds_real = fa.get_landmarks(input_img_real)[-1]
preds_fake256 = fa.get_landmarks(input_img_fake256)[-1]
#preds_fake128 = fa.get_landmarks(input_img_fake128)[-1]

base = np.linalg.norm(preds_real[36] - preds_real[45])  # outter corners of eyes

def nme(pred_pts, gt_pts, base):
    return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1) / base)


out_nme256 = nme(preds_fake256, preds_real, base)
#out_nme128 = nme(preds_fake128, preds_real, base)
end_time = time.time()
cost_time = end_time - start_time
print('nme256为{:.4f},测试时间为{:.4f}'.format(out_nme256, cost_time))
#print('nme128为{:.4f},测试时间为{:.4f}'.format(out_nme128, cost_time))

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 3, 1)
ax.imshow(input_img_real)

for pred_type in pred_types.values():
    ax.plot(preds_real[pred_type.slice, 0],
            preds_real[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')

ax = fig.add_subplot(1, 3, 2)
ax.imshow(input_img_fake256)

for pred_type in pred_types.values():
    ax.plot(preds_fake256[pred_type.slice, 0],
            preds_fake256[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')

# ax = fig.add_subplot(1, 3, 3)
# ax.imshow(input_img_fake128)

# for pred_type in pred_types.values():
#     ax.plot(preds_fake128[pred_type.slice, 0],
#             preds_fake128[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)
#
# ax.axis('off')

# 3D-Plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(preds[:, 0] * 1.2,
#                   preds[:, 1],
#                   preds[:, 2],
#                   c='cyan',
#                   alpha=1.0,
#                   edgecolor='b')

# for pred_type in pred_types.values():
#     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
#               preds[pred_type.slice, 1],
#               preds[pred_type.slice, 2], color='blue')

# ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
