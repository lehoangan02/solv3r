import sys, numpy as np
sys.path.insert(0, "/home/lehoangan/LHAN/solv3r_test/solv3r/examples")
from windowed import umeyama
ref = np.load(sys.argv[1]); 
for f in sys.argv[2:]:
    w = np.load(f)
    a, b = w["c2w"][:, :3, 3].astype(float), ref["c2w"][:, :3, 3].astype(float)
    s, R, t = umeyama(a, b)
    err = np.linalg.norm(a @ (s*R).T + t - b, axis=1)
    ext = np.linalg.norm(b - b.mean(0), axis=1).max()
    print(f"{f.split(chr(47))[-3]}: ATE rmse {np.sqrt((err**2).mean())/ext*100:.2f}% of path radius, max {err.max()/ext*100:.2f}%")
