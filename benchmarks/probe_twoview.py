"""Prototype: compare a reconstruction's relative rotations with direct two-view
estimates from each matched pair, alongside per-pose error against ground truth.
    python benchmarks/probe_twoview.py <work_dir> <truth.json>"""
import sys, shutil, tempfile, os, json, numpy as np, pycolmap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from app.geometry.conventions import cv_rotation_to_camerapath, blender_quat_to_camerapath
from app.geometry.rotations import matrix_to_quat, quat_angular_distance
from app.geometry.alignment import align_poses

work = sys.argv[1]
tmp = tempfile.mkdtemp(prefix="cpl_tv_")
db_path = os.path.join(tmp, "database.db")
shutil.copy(os.path.join(work, "database.db"), db_path)
db = pycolmap.Database.open(db_path)
db.clear_matches(); db.clear_two_view_geometries(); db.close()

pairing = pycolmap.SequentialPairingOptions()
pairing.overlap = 5; pairing.quadratic_overlap = True; pairing.loop_detection = False
verify = pycolmap.TwoViewGeometryOptions(); verify.compute_relative_pose = True
pycolmap.match_sequential(database_path=db_path, pairing_options=pairing,
                          verification_options=verify, device=pycolmap.Device.cpu)

rec = pycolmap.Reconstruction(os.path.join(work, "sparse", "0"))
db = pycolmap.Database.open(db_path)
ids = sorted(rec.reg_image_ids(), key=lambda i: rec.image(i).name)
R = {i: np.asarray(rec.image(i).cam_from_world().rotation.matrix()) for i in ids}
ang = lambda M: float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))

per_image = {i: [] for i in ids}
missing = []
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        i, j = ids[a], ids[b]
        if not db.exists_two_view_geometry(i, j):
            continue
        tvg = db.read_two_view_geometry(i, j)
        n_in = len(tvg.inlier_matches)
        if tvg.cam2_from_cam1 is None:
            missing.append(str(tvg.config)); continue
        if n_in < 30:
            continue
        R_tv = np.asarray(tvg.cam2_from_cam1.rotation.matrix())
        R_rec = R[j] @ R[i].T
        d = ang(R_tv.T @ R_rec)
        per_image[i].append((d, n_in, rec.image(j).name)); per_image[j].append((d, n_in, rec.image(i).name))

truth = json.load(open(sys.argv[2])); tmap = {t["frame"]: t for t in truth["frames"]}
P, Q, RP, RQ = [], [], [], []
for i in ids:
    img = rec.image(i); cfw = img.cam_from_world()
    Rw = np.asarray(cfw.rotation.matrix()); t = np.asarray(cfw.translation)
    P.append(-Rw.T @ t); Q.append(matrix_to_quat(cv_rotation_to_camerapath(Rw.T)))
    f = tmap[int(img.name[1:7])]; RP.append(f["location"]); RQ.append(blender_quat_to_camerapath(np.array(f["quaternion_wxyz"])))
tf, method = align_poses(np.array(P), Q, np.array(RP), RQ)
print(f"alignment: {method}")
print(f"{'image':12s} {'pairs':>5s} {'tv median':>10s} {'tv max':>8s} {'TRUTH err':>10s}")
for k, i in enumerate(ids):
    ds = per_image[i]; vals = [d for d, _, _ in ds]
    terr = float(np.degrees(quat_angular_distance(tf.apply_quaternion(Q[k]), RQ[k])))
    print(f"{rec.image(i).name:12s} {len(ds):5d} {np.median(vals) if vals else float('nan'):10.3f} "
          f"{max(vals) if vals else float('nan'):8.3f} {terr:10.3f}")
from collections import Counter
print("pairs without relative pose:", len(missing), Counter(missing))
