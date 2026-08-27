import h5py
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

#plt.style.use('seaborn-v0_8-white')
#plt.style.use('fivethirtyeight')
#plt.style.use('grayscale')
#plt.style.use('classic')

plt.style.use("grayscale")

plt.rcParams.update({'font.size': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.titlesize': 24, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

def compile_P_counts(selection=False):
    if selection:
        csv_name="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/P_counts_selected.csv"

    else:
        csv_name="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/P_counts.csv"
        
    H = np.loadtxt(csv_name, delimiter=",")
     
    return H

def read_sparse_Q(selection, zmax_index, num_gals, print_stats=False):
    if selection:
        filename="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/Q_sparse_selected.h5"
    else:
        filename="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/Q_sparse.h5"

    if selection:
        file_length=19983948

    else:
        file_length=40894394

    rng=np.random.default_rng(seed=123)
    random_indices=np.sort(rng.choice(file_length, size=num_gals, replace=False))

    with h5py.File(filename, "r") as f:
        indices = list(f["indices"][random_indices])
        values = list(f["values"][random_indices])

    indices_new = []
    values_new = []
    norms = []

    for idx, val in zip(indices, values):
        rows, cols = np.unravel_index(idx, (45, 40))

        mask = rows <= zmax_index

        rows_new = rows[mask]
        cols_new = cols[mask]
        vals_new = val[mask]

        idx_new = np.ravel_multi_index(
            (rows_new, cols_new),
            (zmax_index+1, 40)
        )

        indices_new.append(idx_new)
        values_new.append(vals_new)
        norms.append(np.sum(vals_new))

    max_len = max(len(x) for x in indices_new)
    total_entries = sum(len(x) for x in indices_new)
    average_len = total_entries / len(indices_new)

    if print_stats:
        print("max_len:", max_len)
        print("average_len:", average_len)
        print("total padded entries:", len(indices_new) * max_len)
        print("total actual entries:", total_entries)
        print(
            "padding factor:",
            len(indices_new) * max_len / total_entries
        )

    return indices_new, values_new, np.array(norms)

def animate_Q(
    n_frames,
    selection,
    fps=10,
    z_idx=19
):

    z_values = np.linspace(0.1, 4.5, 45)
    z_max = z_values[z_idx]

    if selection:
        filename = (
            "/projects/public/u6dl/KiDS_Legacy_HBM/"
            "Saved_arrays/Q_sparse_selected.h5"
        )
    else:
        filename = (
            "/projects/public/u6dl/KiDS_Legacy_HBM/"
            "Saved_arrays/Q_sparse.h5"
        )

    nz = 45
    nM = 40

    zmin, zmax = 0, z_max
    Mmin, Mmax = 8, 12

    with h5py.File(filename, "r") as f:

        indices_full = f["indices"]
        values_full = f["values"]

        rng=np.random.default_rng(seed=123)
        random_inds = np.sort(
            rng.choice(
                len(indices_full),
                size=n_frames,
                replace=False
            )
        )

        indices = indices_full[random_inds]
        values = values_full[random_inds]

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#f5f5f5')

        image = ax.imshow(
            np.zeros((z_idx, nM)).T,
            origin="lower",
            extent=[zmin, zmax, Mmin, Mmax],
            aspect="auto",
            interpolation="nearest",
            cmap='cubehelix_r'
        )

        ax.grid(axis='x', alpha=0.2)
        ax.set_xlabel("z")
        ax.set_ylabel("logM")

        title = ax.set_title(
            "KiDS-Legacy posteriors",
            fontsize=20,
            pad=12
        )

        galaxy=ax.text(
            0.05, 0.95,
            "Galaxy",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor='white', alpha=1)
        )

        cbar = fig.colorbar(image, ax=ax, pad=0.01)
        cbar.set_label("Counts")

        def update(i):
            nonlocal random_inds

            Q = np.zeros((nz, nM), dtype=np.float32)

            idx = np.asarray(indices[i], dtype=np.uint16)
            vals = np.asarray(values[i], dtype=np.uint16)

            Q_flat = Q.ravel()
            Q_flat[idx] = vals

            image.set_data(Q[:z_idx+1].T)

            image.set_clim(
                vmin=0,
                vmax=50
            )

            galaxy.set_text(f"Galaxy #{random_inds[i]}")

            return image, title

        animation = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000/fps,
            blit=True
        )

        # -------------------------------
        # Save GIF
        # -------------------------------

        animation.save(
            './Sampling/Plots/Data/Q samples.gif',
            writer=PillowWriter(fps=fps),
            dpi=300,
            savefig_kwargs={
                'facecolor': '#f5f5f5'
            }
        )

        # -------------------------------
        # Save PNG frames
        # -------------------------------

        frame_folder = Path(
            './Sampling/Plots/Animations/Q samples frames'
        )

        frame_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for i in range(n_frames):

            update(i)

            fig.savefig(
                frame_folder / f'Q samples-{i}.png',
                dpi=300,
                facecolor='#f5f5f5',
                bbox_inches='tight'
            )

        plt.close(fig)

def plot_P(selection, zmax_idx=19, ax=None):
    savefig= ax is None

    if savefig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    zvals=np.arange(0.1, 4.6, 0.1)
    zmax=zvals[zmax_idx]

    H=compile_P_counts(selection=selection)
    H=H/H.sum()
    H=H[:zmax_idx+1, :]
    
    im=ax.imshow(H.T, interpolation='auto', origin='lower', aspect='auto', cmap='cubehelix_r', extent=[0, zmax, 8, 12])

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log{M}$')
    ax.set_title('pop-cosmos mock population', pad=12, fontsize=32)

    cbar=fig.colorbar(im, ax=ax, shrink=1, pad=0.01)
    cbar.set_label(r'$P_{jk}$')

    fig.patch.set_facecolor('#f5f5f5')
    
    if savefig:
        plt.savefig('./Sampling/Plots/Data/Prior mock catalog.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())


def plot_z_hist(selection, ax=None, bins=20):
    savefig=ax is None
    if savefig:
        fig, ax = plt.subplots(figsize=(6, 4))

    filename = (
        "/projects/public/u6dl/KiDS_Legacy_HBM/"
        "Saved_arrays/Z_selected.h5"
        if selection else
        "/projects/public/u6dl/KiDS_Legacy_HBM/"
        "Saved_arrays/Z.h5"
    )

    with h5py.File(filename, "r") as f:
        z = f["Z_B"][:]

    print((z<1).mean())

    ax.hist(z, bins=bins, range=(0, 2), color='grey', edgecolor='black')

    ax.set_xlabel(r"$z_{BPZ}$")
    ax.set_ylabel("Number of galaxies")
    ax.set_title(rf"$z$ (selection={selection})", pad=10)
    ax.set_yscale('log')
    ax.set_xlim(0, 2)
    ax.tick_params(
        axis='y',
        which='both',
        direction='in'
    )
    ax.tick_params(axis='y', which='minor', labelleft=False)


    if savefig:
        plt.savefig('./Sampling/Plots/Data/z histogram.png', dpi=300, bbox_inches='tight')

    return ax

def plot_dataspan(num_gals, selection, zmax_index=19):
    zvals = np.arange(0.1, 4.6, 0.1)
    zmax = zvals[zmax_index]

    H = compile_P_counts(selection=selection)
    P = H / H.sum()
    P = P[:zmax_index + 1]

    indices, values, _ = read_sparse_Q(
        selection=selection,
        zmax_index=zmax_index,
        num_gals=num_gals
    )

    all_indices = np.concatenate(indices)
    all_values = np.concatenate(values)

    hist = np.bincount(
        all_indices,
        weights=all_values,
        minlength=(zmax_index + 1) * 40
    ).reshape(zmax_index + 1, 40)

    hist=hist/hist.sum()

    ratio = hist / P

    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    im0 = axs[0].imshow(
        hist.T, interpolation='auto', origin='lower',
        cmap='cubehelix_r', extent=[0, zmax, 8, 12]
    )

    im1 = axs[1].imshow(
        P.T, interpolation='auto', origin='lower',
        cmap='cubehelix_r', extent=[0, zmax, 8, 12]
    )

    im2 = axs[2].imshow(
        ratio.T, interpolation='auto', origin='lower',
        cmap='cubehelix_r', extent=[0, zmax, 8, 12]
    )

    nan_rows, nan_cols = np.where(np.isnan(ratio))
    axs[2].scatter(
        zvals[nan_rows]-0.05,
        8 + (nan_cols + 0.5) * (4 / 40),
        marker='x',
        color='black',
        s=30
    )

    pad = 14
    axs[0].set_title('KiDS galaxy posteriors', pad=pad)
    axs[1].set_title('pop-cosmos prior', pad=pad)
    axs[2].set_title('KiDS posteriors / pop-cosmos prior',pad=pad)

    for ax in axs:
        ax.set_xlabel('z')
        ax.set_ylabel('logM')

    fig.colorbar(im0, ax=axs[0], label='Total Q')
    fig.colorbar(im1, ax=axs[1], label='P')
    fig.colorbar(im2, ax=axs[2], label='Ratio')

    if selection:
        filename='./Sampling/Plots/Data/Dataspan, selected.png'
        plt.suptitle('Selection=True')

    else:
        filename='./Sampling/Plots/Data/Dataspan, unselected.png'
        plt.suptitle('Selection=False')

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

def main():
    animate_Q(n_frames=20, selection=True, fps=2, z_idx=19)
    #plot_P(selection=True, zmax_idx=19)
    #plot_z_hist(selection=True)
    #plot_dataspan(num_gals=50000, selection=True)

if __name__ == "__main__":
    main()