import sys, json, os
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Load transforms
with open(r'C:\Users\徐子屹\Desktop\claudecode\Gaussfluid复现\4DGS_data\transforms_train.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"=== Dataset Analysis ===")
print(f"camera_angle_x: {d['camera_angle_x']:.6f} rad ({np.degrees(d['camera_angle_x']):.1f} deg)")
print(f"Total frames: {len(d['frames'])}")

# Time analysis
times = sorted(set(fr.get('time', 0) for fr in d['frames']))
print(f"Unique timestamps: {len(times)}")
print(f"Time range: [{min(times)}, {max(times)}]")

# t=0 analysis
t0 = [fr for fr in d['frames'] if abs(fr.get('time', 0)) < 1e-6]
print(f"\nt=0 frames: {len(t0)}")

# Camera centers from c2w matrices
centers = []
for fr in d['frames']:
    m = np.array(fr['transform_matrix'])
    cam_center = m[:3, 3]  # c2w => camera center is translation column
    centers.append(cam_center)
centers = np.array(centers)

print(f"\n=== Camera Center Stats ===")
print(f"  x: [{centers[:,0].min():.3f}, {centers[:,0].max():.3f}]")
print(f"  y: [{centers[:,1].min():.3f}, {centers[:,1].max():.3f}]")
print(f"  z: [{centers[:,2].min():.3f}, {centers[:,2].max():.3f}]")
extent = np.linalg.norm(centers.max(0) - centers.min(0))
print(f"  Extent (diagonal): {extent:.3f}")

# Check how many unique camera positions at t=0
t0_centers = []
for fr in t0:
    m = np.array(fr['transform_matrix'])
    t0_centers.append(tuple(np.round(m[:3, 3], 4)))
unique_t0 = len(set(t0_centers))
print(f"\nt=0 unique camera positions: {unique_t0}")

# Check images
base = r'C:\Users\徐子屹\Desktop\claudecode\Gaussfluid复现\4DGS_data'
sample_path = os.path.join(base, d['frames'][0]['file_path'] + '.png')
print(f"\n=== Image Analysis ===")
if os.path.exists(sample_path):
    img = Image.open(sample_path)
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    img_np = np.array(img)
    print(f"Image array shape: {img_np.shape}")
    if img.mode == 'RGBA':
        alpha = img_np[:,:,3]
        fg_pixels = (alpha > 25).sum()
        total = alpha.size
        print(f"Alpha channel: min={alpha.min()}, max={alpha.max()}")
        print(f"Foreground pixels (alpha>25): {fg_pixels} / {total} ({100*fg_pixels/total:.1f}%)")
    print(f"RGB range: [{img_np[:,:,:3].min()}, {img_np[:,:,:3].max()}]")
else:
    print(f"Image not found: {sample_path}")

# Check multiple time steps
print(f"\n=== Time distribution ===")
for t in times[:3]:
    count = sum(1 for fr in d['frames'] if abs(fr.get('time', 0) - t) < 1e-6)
    print(f"  t={t:.4f}: {count} views")
print(f"  ...")
for t in times[-2:]:
    count = sum(1 for fr in d['frames'] if abs(fr.get('time', 0) - t) < 1e-6)
    print(f"  t={t:.4f}: {count} views")

# Check image at different times to see motion
t_last = times[-1]
t_last_frames = [fr for fr in d['frames'] if abs(fr.get('time', 0) - t_last) < 1e-6]
if t_last_frames:
    last_path = os.path.join(base, t_last_frames[0]['file_path'] + '.png')
    if os.path.exists(last_path):
        img_last = Image.open(last_path)
        img_last_np = np.array(img_last)
        if img_last.mode == 'RGBA':
            alpha_last = img_last_np[:,:,3]
            fg_last = (alpha_last > 25).sum()
            print(f"\nt={t_last} first view foreground: {fg_last} pixels")
            
            # Compare with t=0
            t0_path = os.path.join(base, t0[0]['file_path'] + '.png')
            img_t0 = Image.open(t0_path)
            t0_np = np.array(img_t0)
            alpha_t0 = t0_np[:,:,3]
            fg_t0 = (alpha_t0 > 25).sum()
            print(f"t=0.0 first view foreground: {fg_t0} pixels")
            
            # Check if same camera angle
            m0 = np.array(t0[0]['transform_matrix'])
            ml = np.array(t_last_frames[0]['transform_matrix'])
            print(f"\nt=0 cam0 c2w:\n{m0}")
            print(f"t={t_last} cam0 c2w:\n{ml}")
            cam_diff = np.linalg.norm(m0 - ml)
            print(f"Camera matrix difference: {cam_diff:.6f}")

# Scene scale analysis
print(f"\n=== Scene Scale ===")
# Look at all camera positions relative to origin
dists_to_origin = np.linalg.norm(centers, axis=1)
print(f"Camera distances to origin: [{dists_to_origin.min():.3f}, {dists_to_origin.max():.3f}], mean={dists_to_origin.mean():.3f}")

# Estimate scene center from camera look-at
# Cameras look at -z in camera space, pointing toward scene center
look_dirs = []
for fr in d['frames'][:28]:
    m = np.array(fr['transform_matrix'])
    look = -m[:3, 2]  # -z direction in world
    look_dirs.append(look)
look_dirs = np.array(look_dirs)
print(f"Camera look directions (mean): {look_dirs.mean(axis=0)}")
print(f"This suggests scene is near origin offset by camera translation")
