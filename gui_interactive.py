from core import Figure8Curve, C2ClothoidFigure8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

def run_gui(default_a: float = 0.85, default_c: float = 12.0, frenet_spacing: float = 0.5):
    c1_curve = Figure8Curve(a=default_a)
    c2_curve = C2ClothoidFigure8(a=default_a)

    fig = plt.figure(figsize=(14, 12))
    plt.subplots_adjust(left=0.1, bottom=0.28, right=0.85, top=0.92, hspace=0.45)

    gs = GridSpec(3, 1, height_ratios=[5, 1.2, 1.2])
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax_kappa = fig.add_subplot(gs[1])
    ax_tau = fig.add_subplot(gs[2])

    frenet_artists = []

    def redraw(a_val, c_val, show_c1, show_c2, show_frenet, spacing):
        nonlocal frenet_artists, c1_curve, c2_curve
        if abs(a_val - c1_curve.a) > 1e-6:
            c1_curve = Figure8Curve(a_val)
            c2_curve = C2ClothoidFigure8(a_val)

        t = np.linspace(0, 1, 4000)
        s = t * c1_curve.total_length()

        ax3d.cla()

        if show_c1:
            x1, y1, z1 = c1_curve.get_dense_points(4000, c_val)
            ax3d.plot(x1, y1, z1, color='#1f77b4', linewidth=6, label='C¹ Path')

        if show_c2:
            x2, y2, z2 = c2_curve.get_dense_points(4000, c_val)
            ax3d.plot(x2, y2, z2, color='#ff7f0e', linewidth=6, label='C² Clothoid')

        for artist in frenet_artists:
            artist.remove()
        frenet_artists.clear()

        if show_frenet:
            n_frames = max(6, min(70, int(c1_curve.total_length() / max(spacing, 0.05))))
            ts_frames = np.linspace(0, 1, n_frames)
            if show_c1:
                for tt in ts_frames:
                    T, N, B = c1_curve.frenet_frame(tt, c_val)
                    pos = np.array(c1_curve.evaluate(tt, c_val))
                    scale = 0.25
                    qT = ax3d.quiver(pos[0], pos[1], pos[2], T[0], T[1], 0, length=int(scale), arrow_length_ratio=0.25)
                    qN = ax3d.quiver(pos[0], pos[1], pos[2], N[0], N[1], 0, length=int(scale), arrow_length_ratio=0.25)
                    qB = ax3d.quiver(pos[0], pos[1], pos[2], B[0], B[1], 0, length=int(scale), arrow_length_ratio=0.25)
                    frenet_artists.extend([qT, qN, qB])
            if show_c2:
                for tt in ts_frames:
                    T, N, B = c2_curve.frenet_frame(tt, c_val)
                    pos = np.array(c2_curve.evaluate(tt, c_val))
                    scale = 0.25
                    qT = ax3d.quiver(pos[0], pos[1], pos[2], T[0], T[1], 0, color='#ff4444', length=int(scale), arrow_length_ratio=0.25)
                    qN = ax3d.quiver(pos[0], pos[1], pos[2], N[0], N[1], 0, color='#44ff44', length=int(scale), arrow_length_ratio=0.25)
                    qB = ax3d.quiver(pos[0], pos[1], pos[2], B[0], B[1], 0, color='#4444ff', length=int(scale), arrow_length_ratio=0.25)
                    frenet_artists.extend([qT, qN, qB])

        # Equal axes
        if show_c1 or show_c2:
            coords = []
            if show_c1: coords.append(np.array(c1_curve.get_dense_points(4000, c_val)))
            if show_c2: coords.append(np.array(c2_curve.get_dense_points(4000, c_val)))
            all_coords = np.concatenate(coords, axis=1)
            max_range = max(all_coords[0].max()-all_coords[0].min(), all_coords[1].max()-all_coords[1].min(), all_coords[2].max()-all_coords[2].min()) * 1.15
            mid = all_coords.mean(axis=1)
        else:
            max_range, mid = 3.0, (0.0, 0.0, c_val/2)

        ax3d.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
        ax3d.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
        ax3d.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
        ax3d.set_box_aspect([1, 1, 1])

        ax3d.set_title(f'C¹ + Optional C² Fiber Figure-8\n'
                       f'a = {a_val:.4f} | c = {c_val:.1f} | Frenet spacing = {spacing:.2f}',
                       fontsize=15, fontweight='bold')
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        if show_c1 or show_c2:
            ax3d.legend(loc='upper left')

        # Curvature tape
        ax_kappa.cla()
        if show_c1:
            ax_kappa.plot(s, c1_curve.curvature(t, c_val), color='#1f77b4', linewidth=3, label='C¹ κ(s)')
        if show_c2:
            ax_kappa.plot(s, c2_curve.curvature(t, c_val), color='#ff7f0e', linewidth=3, label='C² κ(s)')
        ax_kappa.set_ylabel('Curvature κ(s)')
        ax_kappa.set_xlabel('Arc length s')
        ax_kappa.grid(True, alpha=0.5)
        if show_c1 or show_c2:
            ax_kappa.legend()

        # Torsion tape
        ax_tau.cla()
        if show_c1:
            ax_tau.plot(s, c1_curve.torsion(t, c_val), color='#1f77b4', linewidth=3, label='C¹ τ(s)')
        if show_c2:
            ax_tau.plot(s, c2_curve.torsion(t, c_val), color='#ff7f0e', linewidth=3, label='C² τ(s)')
        ax_tau.set_ylabel('Torsion τ(s)')
        ax_tau.set_xlabel('Arc length s')
        ax_tau.grid(True, alpha=0.5)
        if show_c1 or show_c2:
            ax_tau.legend()

        fig.canvas.draw_idle()

    # Widgets
    ax_a = plt.axes((0.15, 0.22, 0.65, 0.03))
    slider_a = Slider(ax_a, 'a (green distance)', 0.20, 2.50, valinit=default_a, valstep=0.001)

    ax_c = plt.axes((0.15, 0.18, 0.65, 0.03))
    slider_c = Slider(ax_c, 'c (z-rise per loop)', -20, 20, valinit=default_c, valstep=0.1)

    ax_spacing = plt.axes((0.15, 0.14, 0.65, 0.03))
    slider_spacing = Slider(ax_spacing, 'Frenet frame spacing (units)', 0.1, 2.0, valinit=frenet_spacing, valstep=0.05)

    ax_check = plt.axes((0.82, 0.75, 0.16, 0.15))
    check = CheckButtons(ax_check, ['Show C¹ Path', 'Show C² Clothoid', 'Show Frenet Frames'], [True, True, True])

    def on_change(_):
        show_c1 = check.get_status()[0]
        show_c2 = check.get_status()[1]
        show_frenet = check.get_status()[2]
        redraw(slider_a.val, slider_c.val, show_c1, show_c2, show_frenet, slider_spacing.val)

    for s in (slider_a, slider_c, slider_spacing):
        s.on_changed(on_change)
    check.on_clicked(on_change)

    redraw(default_a, default_c, True, True, True, frenet_spacing)
    plt.show()

if __name__ == "__main__":
    run_gui(default_a=0.85, default_c=12.0, frenet_spacing=0.6)