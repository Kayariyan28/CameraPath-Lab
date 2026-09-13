"""Prototype: profile-likelihood test of focal observability on a COLMAP model.

    python benchmarks/probe_focal.py <sparse_model_dir> [true_fov_deg]
"""
import sys, copy, numpy as np, pycolmap

model_dir = sys.argv[1]
true_fov = float(sys.argv[2]) if len(sys.argv) > 2 else None
base = pycolmap.Reconstruction(model_dir)
cam0 = next(iter(base.cameras.values()))
W = cam0.width
fov = lambda f: float(np.degrees(2 * np.arctan((W / 2) / f)))

def cost_at(focal: float) -> float:
    rec = pycolmap.Reconstruction(base)          # copy
    for cam in rec.cameras.values():
        cam.focal_length = focal
    o = pycolmap.BundleAdjustmentOptions()
    o.refine_focal_length = False
    o.refine_extra_params = False
    o.refine_principal_point = False
    pycolmap.bundle_adjustment(rec, o)
    return float(rec.compute_mean_reprojection_error())

f_star = float(cam0.focal_length)
print(f"model focal {f_star:.1f}px = {fov(f_star):.2f} deg" + (f" (true {true_fov:.2f})" if true_fov else ""))
e0 = cost_at(f_star)
for k in (0.70, 0.85, 1.15, 1.30):
    e = cost_at(f_star * k)
    print(f"  focal x{k:.2f} ({fov(f_star*k):6.2f} deg): reproj {e:.4f} px   rise {100*(e-e0)/e0:+7.2f}%")
print(f"  baseline at solution: {e0:.4f} px")
