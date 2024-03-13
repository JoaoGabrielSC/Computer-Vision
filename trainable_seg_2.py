import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import cv2

# Carregar a imagem a partir do diretório
full_img_kr = cv2.imread("images/kr_02.png")
full_img_kr = cv2.cvtColor(full_img_kr, cv2.COLOR_BGR2RGB)
full_img_kr = cv2.GaussianBlur(full_img_kr, (3,3), 0)


training_labels = np.zeros(full_img_kr.shape[:2], dtype=np.uint8)
training_labels[10:150, 170:1250] = 1 # out
training_labels[507:691, 570:626] = 1 # out
training_labels[475:630, 360:550] = 2 # refratario
training_labels[575:650, 218:550] = 2 # refratario
#training_labels[513:535, 670:781] = 2 # refratario
training_labels[625:652, 634:664] = 2 # refratario
training_labels[730:845, 335:480] = 3 # slag
training_labels[942:992, 591:686] = 3 # slag
training_labels[738:754, 560:581] = 3 # slag
#training_labels[711:745, 586:673] = 3 # slag
training_labels[808:817, 555:590] = 4 # pir iron
training_labels[893:903, 491:498] = 4 # pir iron


sigma_min = 1
sigma_max = 20
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)

features = features_func(full_img_kr)

clf = RandomForestClassifier(n_estimators=40, n_jobs=-1,
                             max_depth=30, max_samples=0.05)

clf = future.fit_segmenter(training_labels, features, clf)

result = future.predict_segmenter(features, clf)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(full_img_kr, result, mode='thick'))
ax[0].contour(training_labels)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()


# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
l = len(clf.feature_importances_)
feature_importance = (
        clf.feature_importances_[:l//3],
        clf.feature_importances_[l//3:2*l//3],
        clf.feature_importances_[2*l//3:])
sigmas = np.logspace(
        np.log2(sigma_min), np.log2(sigma_max),
        num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
        base=2, endpoint=True)
for ch, color in zip(range(3), ['r', 'g', 'b']):
    ax[0].plot(sigmas, feature_importance[ch][::3], 'o', color=color)
    ax[0].set_title("Intensity features")
    ax[0].set_xlabel("$\\sigma$")
for ch, color in zip(range(3), ['r', 'g', 'b']):
    ax[1].plot(sigmas, feature_importance[ch][1::3], 'o', color=color)
    ax[1].plot(sigmas, feature_importance[ch][2::3], 's', color=color)
    ax[1].set_title("Texture features")
    ax[1].set_xlabel("$\\sigma$")

fig.tight_layout()


img_new = cv2.imread("images/kr_03.jpg")
img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)

features_new = features_func(img_new)
result_new = future.predict_segmenter(features_new, clf)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
ax[0].imshow(segmentation.mark_boundaries(img_new, result_new, mode='thick'))
ax[0].set_title('Image')
ax[1].imshow(result_new)
ax[1].set_title('Segmentation')
fig.tight_layout()

plt.show()