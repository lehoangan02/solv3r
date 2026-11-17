"""Pick N keyframes spaced evenly in cumulative image motion, the sharpest frame of each motion segment."""
import sys, os, shutil, cv2, numpy as np
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
prev, motion, sharp = None, [0.0], []
for f in files:
    im = cv2.imread(os.path.join(src, f), cv2.IMREAD_GRAYSCALE)
    sharp.append(cv2.Laplacian(im, cv2.CV_64F).var())
    s = cv2.resize(im, (192, int(192 * im.shape[0] / im.shape[1])))
    if prev is not None:
        fl = cv2.calcOpticalFlowFarneback(prev, s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion.append(motion[-1] + float(np.linalg.norm(fl, axis=2).mean()))
    prev = s
motion = np.array(motion); sharp = np.array(sharp)
n = min(n, len(files))
seg = np.minimum((motion / (motion[-1] + 1e-9) * n).astype(int), n - 1)
pick = [int(np.where(seg == k)[0][np.argmax(sharp[seg == k])]) for k in range(n) if (seg == k).any()]
pick[0] = 0                                           # first frame defines the world frame
os.makedirs(out, exist_ok=True)
for i, p in enumerate(sorted(set(pick))):
    shutil.copy(os.path.join(src, files[p]), os.path.join(out, f"{i:04d}.jpg"))
print(len(files), "->", len(set(pick)), "keyframes")
