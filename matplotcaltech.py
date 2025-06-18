import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg
import numpy as np

# Load Doppler data
filepath = r'c:\Programming\Deep FUS\data\CalTech\data\doppler\doppler_S27_R99+normcorre.mat'
with h5py.File(filepath, 'r') as f:
    doppler = f['iDop'][:]  # Shape: (242, 15, 128, 103)
    print("Shape of iDop:", doppler.shape)

# Choose a slice (depth index)
slice_idx = 64  # Mid-brain slice
frames = doppler[:, :, slice_idx, :]  # Shape: (242, 15, 103)

# Set up plot
fig, ax = plt.subplots()
im = ax.imshow(frames[:, :, 0], cmap='gray', aspect='auto')
ax.set_title(f'Slice {slice_idx} - Time 0')
ax.axis('off')

# Animation function
def update(frame_idx):
    im.set_array(frames[:, :, frame_idx])
    ax.set_title(f'Slice {slice_idx} - Time {frame_idx}')
    return [im]

# Create animation object
ani = animation.FuncAnimation(
    fig, update, frames=frames.shape[2], interval=50, blit=True
)

# Tell matplotlib where ffmpeg is
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# Save animation as .mp4
ani.save('doppler_slice_time.mp4', writer='ffmpeg', fps=20)

print("✅ Saved video as 'doppler_slice_time.mp4'")
plt.close()
