import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
from pathlib import Path


#plt.style.use('seaborn-v0_8-white')
#plt.style.use('fivethirtyeight')
#plt.style.use('grayscale')
#plt.style.use('classic')

plt.style.use('seaborn-v0_8-white')

plt.rcParams.update({'font.size': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.titlesize': 24, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',})

plt.rcParams['xtick.top']=True
plt.rcParams['ytick.right']=True

zmax=2.0

def load_F(filename):
    path="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/F/"
    full_path=os.path.join(path, filename)

    return np.load(full_path)

def get_z_centres(F_mean):
    zedges=np.linspace(0, zmax, F_mean.shape[0]+1)
    zcentres=0.5*(zedges[:-1]+zedges[1:])
    zcentres=np.round(zcentres, 2)
    
    return zcentres

def get_m_centres(F_mean):
    Medges = np.linspace(8, 12, F_mean.shape[1] + 1)
    Mcentres = 0.5 * (Medges[:-1] + Medges[1:])
    
    return Mcentres

def normalise_grid(F_sample): #normalises over the columns
    F=np.copy(F_sample)
    
    for i in range(F.shape[0]):
        F[i]=F[i]/(F[i].sum())

    return F

def hist(F_sample, ax=None, colorbar=False, colorbar_label=None, vmin=None, vmax=None, lognorm=False, filename='F.png', cmap='cubehelix_r', location='right'): #does the binning
    savefig=ax is None

    if savefig:
        colorbar=True
        colorbar_label=r"$F_{jk}$"
        filepath = os.path.join("./Sampling/Plots/pSMF", filename)
        

    if savefig:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.figure

    if lognorm:
        norm=LogNorm()
    else:
        norm=None
    
    im=ax.imshow(F_sample.T
                , origin='lower'
                , extent=[0, zmax, 8, 12]
                , interpolation='auto'
                , aspect='equal'
                , cmap=cmap
                , vmin=vmin
                , vmax=vmax,
                norm=norm
                )

    if colorbar:
        if location=='bottom':
            cbar=fig.colorbar(im, ax=ax, location=location, pad=0.08)
        else:
            cbar=fig.colorbar(im, ax=ax, location=location)

        cbar.ax.tick_params(colors='k')
        cbar.set_label(colorbar_label)


    if savefig:
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\log{M}$")
        ax.set_title(r'Mean $F_{jk}$', pad=8)

        fig.patch.set_facecolor('#f5f5f5')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

    return im

def F_samples(F_grid):
    F_mean=np.mean(F_grid, axis=0)
    F_std=np.std(F_grid, axis=0)
    
    F_thinned=F_grid[::10]
    F_thinned_mean=np.mean(F_thinned, axis=0)

    fig, axs=plt.subplot_mosaic("ABC;DEF;GHI", 
                                figsize=(9, 12), 
                                width_ratios=[1, 1, 1], gridspec_kw={"wspace": 0})

    new_indices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    random_indices=np.random.choice(F_grid.shape[0], size=4, replace=False)

    for i in range(3):
        hist(F_grid[random_indices[i]], ax=axs[new_indices[i]], colorbar=True)
        axs[new_indices[i]].set_title(f"F[{random_indices[i]}]")

    hist(F_mean, ax=axs['D'], colorbar=True)
    hist(F_thinned_mean, ax=axs['E'], colorbar=True)
    hist(F_std, ax=axs['F'], colorbar=True)

    hist(normalise_grid(F_mean), ax=axs['G'], colorbar=True)
    hist(normalise_grid(F_thinned_mean), ax=axs['H'], colorbar=True)
    hist(normalise_grid(F_std), ax=axs['I'], colorbar=True)

    axs['D'].set_title('Mean F', pad=12)
    axs['E'].set_title('Mean F[::10])', pad=12)
    axs['F'].set_title(r"$\sigma_F$", pad=12)

    axs['H'].set_title("(Column-normalised)")

    plt.tight_layout()
    
    plt.savefig('./Sampling/Plots/pSMF/F summary.png', dpi=300)

def mass_functions(F_grid, zidx, titlewords='Mass functions', ax=None, color='grey'):
    F_mean=np.mean(F_grid, axis=0)
    F_norm=normalise_grid(F_mean)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    zcentres=get_z_centres(F_mean)
    Mcentres=get_m_centres(F_mean)
    
    ax.plot(Mcentres, F_norm[zidx, :], color=color, lw=3, label=np.round(zcentres[zidx], 2))
    ax.set_xlabel('logM')
    ax.set_ylabel('p(logM|z)')
    ax.set_yscale('log')
    ax.set_title(titlewords)
    ax.legend(loc='best', frameon=True, title=r'central $z$', title_fontsize=15, fontsize=10)

    plt.savefig('./Sampling/Plots/pSMF/Mass functions.png', dpi=300, bbox_inches='tight')

def calculate_percentile_error_log(F_grid, z_idx=0):
    F_grid=np.copy(F_grid)

    for i in range(F_grid.shape[0]):
        F_grid[i, z_idx]=F_grid[i, z_idx]/np.sum(F_grid[i, z_idx])
        
    log_samples=np.log10(F_grid)
    
    Medges=np.linspace(8, 12, F_grid.shape[2]+1)
    Mcentres=0.5*(Medges[:-1]+Medges[1:])
    
    log_p16, log_median, log_p84=np.percentile(log_samples, [16, 50, 84], axis=0)
    
    return Mcentres, log_median, log_p16, log_p84
    
def single_psmf(F_grid, z_idx, ax=None, linewidth=1, gridtext=True):
    Mcentres, log_median, log_p16, log_p84 = calculate_percentile_error_log(F_grid=F_grid, z_idx=z_idx)

    savefig=ax is None

    if savefig:
        gridtext=False

    log_median = log_median[z_idx]
    lower = log_p16[z_idx]
    upper = log_p84[z_idx]

    z_centres=get_z_centres(F_grid[0])
    z_gap=0.05
    z_label=f'{(z_centres[z_idx]-z_gap):.1f}<z<{(z_centres[z_idx]+z_gap):.1f}'

    if savefig:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.tick_params(axis='y', direction='in', left=True, width=1, length=3, right=False)
            ax.tick_params(axis='x', direction='in', bottom=True, width=1, length=5, top=False)
            ax.grid(axis='x', alpha=0.5)
            ax.set_xlabel(r'$\log{M}$')
            ax.set_ylabel(r'$\log {p(\log{M})}$')
            ax.set_title(f"{z_label}", pad=12)

    ax.step(Mcentres, log_median, color='k', lw=linewidth, label=z_label, zorder=6)
    ax.fill_between(Mcentres, lower, upper,
                    color='orange', alpha=0.2,
                    step='pre',
                    zorder=4
                    )

    ax.set_xticks(range(8, 13))

    if gridtext:
        ax.text(0.5, 0.1, 
                        z_label,
                        transform=ax.transAxes,
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor='lightgrey', alpha=1),
                        zorder=7,
                        fontsize=10
                    )
        ax.set_ylim(-6, 0)

    if savefig:
        fig.patch.set_facecolor('#f5f5f5')
        plt.savefig(f'./Sampling/Plots/pSMF/z_idx={z_idx}.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

    return ax


def psmf(F_grid, num_snakes=3):
    F_grid=np.copy(F_grid)
    F_mean=np.mean(F_grid, axis=0)

    fig, axs=plt.subplots(4, 5, figsize=(10, 6), sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
    axs=axs.flatten()

    Medges = np.linspace(8, 12, F_mean.shape[1] + 1)
    Mcentres = 0.5 * (Medges[:-1] + Medges[1:])

    for i in range(F_mean.shape[0]): #indexes z bin
        single_psmf(F_grid=F_grid, z_idx=i, ax=axs[i])

        if num_snakes>0:
            for j in range(num_snakes): #indexes sample
                F_grid[j, i]=F_grid[j, i]/(F_grid[j, i].sum()) #normalise every z bin function
                log_F=np.log10(F_grid[j, i])
                
                axs[i].step(Mcentres, log_F, lw=0.5, zorder=5)
            
            titlewords=f'pSMF ({num_snakes} snakes)'
            filewords=f'./Sampling/Plots/pSMF/{num_snakes} snakes.png'

        else:
            titlewords=r'pSMF in bins of width $\Delta z=0.1$'
            filewords='./Sampling/Plots/pSMF/pSMF.png'


    for i, ax in enumerate(axs):
        if i%5==0:
            ax.set_ylabel('p(logM)')

        if i>=15:
            ax.set_xlabel('logM')

            if i!=15:
                ax.set_xticks(range(9, 13))

        ax.grid(axis='both', alpha=0.2)

    fig.patch.set_facecolor('#f5f5f5')
    plt.suptitle(titlewords, fontsize=25)
    plt.tight_layout(pad=5)
    
    plt.savefig(filewords, dpi=500, bbox_inches='tight', facecolor=fig.get_facecolor())

def animate_F(F_grid, normalise=False):

    F_grid = F_grid.copy()

    fig = plt.figure(figsize=(9.5, 9))

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 1, 0.05],
        wspace=0
    )

    axs = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1])
    ]

    cax = fig.add_subplot(gs[2])

    if normalise:

        F_norm = np.copy(F_grid)

        for i in range(F_grid.shape[0]):
            F_norm[i] = normalise_grid(F_grid[i])

        titlewords = 'Samples of F normalised'
        filewords = './Sampling/Plots/Animations/Normalised F samples.gif'
        frame_folder = Path('./Sampling/Plots/Animations/Normalised F samples frames')

        F_plot = F_norm

    else:

        titlewords = 'Samples of F'
        filewords = './Sampling/Plots/Animations/F samples.gif'
        frame_folder = Path('./Sampling/Plots/Animations/F samples frames')

        F_plot = F_grid

    # Create folder for PNG frames
    frame_folder.mkdir(parents=True, exist_ok=True)

    vmax = 0.04

    im0 = axs[0].imshow(
        F_plot.mean(axis=0).T,
        origin='lower',
        aspect='equal',
        cmap='cubehelix_r',
        extent=[0, zmax, 8, 12],
        interpolation='auto',
        vmax=vmax
    )

    im1 = axs[1].imshow(
        F_plot[0].T,
        origin='lower',
        aspect='equal',
        cmap='cubehelix_r',
        extent=[0, zmax, 8, 12],
        interpolation='auto',
        vmax=vmax
    )

    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_label('F', fontsize=25)

    for ax in axs:
        ax.grid(axis='y', alpha=0.5)
        ax.set_xlabel(r'$z$', fontsize=24)

    axs[0].set_ylabel(r'$\log{M}$', fontsize=24)

    axs[0].set_title(
        r'Mean $F$ over samples',
        fontsize=25,
        pad=12
    )

    axs[1].tick_params(
        which='both',
        axis='y',
        labelleft=False
    )

    axs[1].set_xlabel(r'$z$')
    axs[1].set_ylabel(None)

    axs[1].set_title(
        titlewords,
        fontsize=25,
        pad=12
    )

    # Select the same 50 samples used in the animation
    indices = np.sort(
        np.random.choice(
            F_grid.shape[0],
            size=50,
            replace=False
        )
    )

    def update(frame):

        index = indices[frame]

        im1.set_data(F_plot[index].T)

        if normalise:
            axs[1].set_title(
                f'Sample {index} from F normalised',
                fontsize=25,
                pad=12
            )
        else:
            axs[1].set_title(
                f'Sample {index} from F',
                fontsize=25,
                pad=12
            )

        return (im1,)

    fig.patch.set_facecolor('#f5f5f5')

    fps = 10

    ani = FuncAnimation(
        fig,
        update,
        frames=50,
        interval=1000 // fps,
        blit=False,
        repeat=True
    )

    # -----------------------------
    # Save GIF
    # -----------------------------

    ani.save(
        filewords,
        writer='pillow',
        fps=15,
        dpi=200,
        savefig_kwargs={
            'facecolor': '#f5f5f5'
        }
    )

    # -----------------------------
    # Save PNG frames
    # -----------------------------

    for frame in range(50):

        update(frame)

        fig.savefig(
            frame_folder / f'F samples-{frame}.png',
            dpi=300,
            facecolor='#f5f5f5',
            bbox_inches='tight'
        )

    return ani

def run_all(F_grid):
    F_samples(F_grid)
    single_psmf(F_grid, z_idx=5)
    psmf(F_grid, num_snakes=3)
    psmf(F_grid, num_snakes=0)

    animate_F(F_grid, normalise=True)
    animate_F(F_grid, normalise=False)


def compare_Fs(F_grids, comparison):
    if comparison=='samples':
        F_grid=np.copy(F_grids)
        num_sizes=5
        sample_sizes=np.linspace(1, F_grid.shape[0], num_sizes).astype(np.int32)
        sample_sizes[0]=50
        sample_sizes[-1]=F_grid.shape[0]

        column_headers=sample_sizes

        titlewords='Changing number of samples'
        filewords='./Sampling/Plots/Sampling process/Compare sample size.png'
    
        num_cols=num_sizes
        column_types='samples'

        F_grids=[F_grid[:sample_size] for sample_size in sample_sizes]


    elif comparison=='warmups':
        F_grids=F_grids.copy()

        num_warmups=[100, 300, 500, 1000, 3000, 10000]
        
        titlewords='Changing number of warmup iterations'

        column_headers=num_warmups

        filewords='./Sampling/Plots/Sampling process/Compare warmup lengths.png'

        num_cols=len(num_warmups)
        column_types='warmup steps'

    num_files = len(F_grids)
    
    fig = plt.figure(figsize=(12, 12))

    gs = fig.add_gridspec(
        3, num_files + 1,
        width_ratios=[1] * num_files + [0.08],
        wspace=0.01,
        hspace=0.1
    )

    # Plot axes
    ax = np.array([
        [fig.add_subplot(gs[row, col]) for col in range(num_files)]
        for row in range(3)
    ])

    ax = ax.flatten()

    # Colourbar axes — all same x position
    cax1 = fig.add_subplot(gs[0, -1])
    cax2 = fig.add_subplot(gs[1, -1])
    cax3 = fig.add_subplot(gs[2, -1])

    for index, file in enumerate(F_grids):
        diff=np.abs(file.mean(axis=0)-F_grids[-1].mean(axis=0))/np.std(F_grids[-1], axis=0)

        im_top=hist(file.mean(axis=0)/F_grids[-1].mean(axis=0), ax=ax[index], vmin=0.9, vmax=1.1)
        im_mid=hist(np.std(file, axis=0)/np.std(F_grids[-1], axis=0), ax=ax[index+len(F_grids)], vmin=0.9, vmax=1.1)
        im_bottom=hist(diff, colorbar_label=r'$F_{\mathrm{Mean}}', ax=ax[index+2*len(F_grids)], vmin=0, vmax=0.1)

    for i, axis in enumerate(ax):
        if i not in [0, len(F_grids), 2*len(F_grids)]:
            axis.set_yticks([])
            axis.set_ylabel(None)

        if i < 2*len(F_grids):
            axis.set_xticks([])
            axis.set_xlabel(None)
            if i < len(F_grids):
                axis.set_title(f"{column_headers[i]}\n{column_types}", fontsize=16)

        if i%num_cols==0:
            axis.set_ylabel('logM')

        if i>=2*num_cols:
            axis.set_xlabel('z')


    cbar1 = fig.colorbar(im_top, cax=cax1)
    cbar2 = fig.colorbar(im_mid, cax=cax2)
    cbar3 = fig.colorbar(im_bottom, cax=cax3)

    cbar1.set_label(r'$\frac{F}{F_{\star}}$')
    cbar2.set_label(r'$\frac{\sigma_F}{\sigma_{\star}}$')
    cbar3.set_label(r'$\frac{|F - F_{\star}|}{\sigma_F}$')

    plt.suptitle(titlewords, fontsize=30)
    plt.savefig(filewords, dpi=300, bbox_inches='tight')

def compare_sample_size_minor(F_grid):
    F_grid=np.copy(F_grid)
    sample_sizes=[50, 1000, 5000, 10000, 50000]

    num_sizes=len(sample_sizes)
    
    titlewords='Optimising sampler parameters: number of steps'
    filewords='./Sampling/Plots/Slides/Compare sample size minor.png'

    F_grids=[F_grid[:sample_size] for sample_size in sample_sizes]

    fig = plt.figure(figsize=(18, 9))

    gs = fig.add_gridspec(
        1, num_sizes+1,
        width_ratios=[1]*num_sizes + [0.08],
        wspace=0
    )

    axs = np.array([
        fig.add_subplot(gs[0, i]) for i in range(num_sizes)
    ])

    cax1 = fig.add_subplot(gs[0, num_sizes])

    for index, file in enumerate(F_grids):
        im = hist(
            file.mean(axis=0) / F_grids[-1].mean(axis=0),
            ax=axs[index],
            vmin=0.9,
            vmax=1.1,
            cmap='coolwarm'
        )

    for i, ax in enumerate(axs):
        ax.tick_params(
            axis='y',
            which='both',
            left=False,
            top=False
        )
        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False
        )
        ax.set_title(f'{sample_sizes[i]} samples', pad=8)
        ax.set_xlabel(r'$z$', fontsize=20)
        if i != 0:
            ax.tick_params(labelleft=False)

    axs[-1].set_title(rf"{sample_sizes[-1]} samples (${{\star}}$)", pad=8)

    cbar1 = fig.colorbar(im, cax=cax1)
    cbar1.set_label(r'$\frac{F_{\text{mean}}}{F_{\text{mean}}^{\star}}$', fontsize=24)

    pos = axs[0].get_position()
    cax1.set_position([
        cax1.get_position().x0,
        pos.y0,
        cax1.get_position().width,
        pos.height
    ])

    axs[0].set_ylabel(r"$\log{M}$", fontsize=20)

    fig.patch.set_facecolor('#f5f5f5')
    plt.suptitle(titlewords, fontsize=30)
    plt.savefig(filewords, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

def compare_warmup_size_minor(F_grids):
    F_grids=F_grids.copy()
    warmup_sizes=[10, 50, 100, 500, 1000]
    
    num_grids=len(warmup_sizes)

    titlewords='Optimising sampler parameters: burn-in phase'
    filewords='./Sampling/Plots/Slides/Compare warmup size minor.png'

    fig = plt.figure(figsize=(18, 9))

    gs = fig.add_gridspec(
        1, num_grids+1,
        width_ratios = [1] * num_grids + [0.08],
        wspace=0
    )

    axs = np.array([
        fig.add_subplot(gs[0, i]) for i in range(num_grids)
    ])

    cax1 = fig.add_subplot(gs[0, num_grids])

    for index, file in enumerate(F_grids):
        im = hist(
            file.mean(axis=0) / F_grids[-1].mean(axis=0),
            ax=axs[index],
            vmin=0.9,
            vmax=1.1,
            cmap='coolwarm'
        )

    for i, ax in enumerate(axs):
        ax.tick_params(
            axis='y',
            which='both',
            left=False,
            top=False
        )
        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False
        )
        ax.set_title(f'{warmup_sizes[i]} \nburn-in samples', pad=8, fontsize=20)
        ax.set_xlabel(r'$z$', fontsize=20)
        if i != 0:
            ax.tick_params(labelleft=False)

    axs[-1].set_title(f"{warmup_sizes[-1]}\nburn-in samples ($\\star$)", pad=8, fontsize=20)

    cbar1 = fig.colorbar(im, cax=cax1)
    cbar1.set_label(r'$\frac{F_{\text{mean}}}{F_{\text{mean}}^{\star}}$', fontsize=24)

    pos = axs[0].get_position()
    cax1.set_position([
        cax1.get_position().x0,
        pos.y0,
        cax1.get_position().width,
        pos.height
    ])

    axs[0].set_ylabel(r"$\log{M}$", fontsize=20)

    fig.patch.set_facecolor('#f5f5f5')
    plt.suptitle(titlewords, fontsize=30)
    plt.savefig(filewords, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

def thinning(F_grid):
    F_grid=F_grid.copy()

    fig, axs=plt.subplots(1, 2, figsize=(7, 9.5), gridspec_kw={'wspace': 0})

    thin_factor=10
    F_grid_sample=F_grid[::thin_factor]

    grids=[F_grid, F_grid_sample]

    means=[grid.mean(axis=0) for grid in grids]
    stds=[grid.std(axis=0) for grid in grids]

    for index in [0, 1]:
        hist(means[index], colorbar_label=r"$\mathrm{Mean} F$", ax=axs[index], vmax=0.04, colorbar=True, location='bottom')

        axs[index].tick_params(axis='x', which='both', bottom=True, top=False, length=5, direction='in', labelbottom=True)
        axs[index].set_xlabel(r'$z$', fontsize=20)

    axs[1].set_xticks(axs[1].get_xticks()[1:])
    axs[1].tick_params(axis='y', which='both', right=False, length=5, direction='in', labelleft=False)
    axs[0].set_ylabel(r'$\log{M}$', fontsize=20)
        
    axs[0].set_title('All samples', pad=12)
    axs[1].set_title(f'Thinned by {thin_factor}', pad=12)

    fig.patch.set_facecolor('#f5f5f5')
    plt.suptitle('Independent adjacent samples', fontsize=32)
    plt.tight_layout()

    plt.savefig('./Sampling/Plots/Slides/Thinning.png', dpi=300, bbox_inches='tight')

def show_galaxy_size(F_grids):
    #F_grids_copy=F_grids.copy
    fig, axs=plt.subplots(1, 4, figsize=(16, 8), sharey=True, gridspec_kw={'wspace': 0})

    galaxy_sizes=[10000, 50000, 80000, 100000]

    for i in range(4):
        single_psmf(F_grids[i], z_idx=5, ax=axs[i], linewidth=3, gridtext=False)
        
        axs[i].set_xlabel(r"$\log {M}$", fontsize=22)
        axs[i].set_xticks(np.arange(8, 13))

        axs[i].tick_params(
            axis='x',
            which='major',
            bottom=True,
            top=True,
            length=6,
            width=1,
            direction='in'
        )
        axs[i].grid(axis='x')

        axs[i].text(0.5, 0.05, 
                    f"{galaxy_sizes[i]} galaxies",
                    transform=axs[i].transAxes,
                    ha='center',
                    va='bottom',
                    bbox=dict(facecolor='lightgrey')
                )

    axs[0].set_ylabel(r"$\log{p}$", fontsize=22)
    fig.patch.set_facecolor('#f5f5f5')

    titlewords='Increasing number of galaxies'
    filewords='./Sampling/Plots/Slides/show_galaxy_size.png'
    plt.suptitle(titlewords, fontsize=30)
    plt.tight_layout()
    plt.savefig(filewords, dpi=300, bbox_inches='tight')


def signal_loss(F_grid):

    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True, gridspec_kw={'hspace': 0}, height_ratios=[6, 1])

    filename = (
        "/projects/public/u6dl/KiDS_Legacy_HBM/"
        "Saved_arrays/Z_selected.h5"
    )

    with h5py.File(filename, "r") as f:
        z = f["Z_B"][:]

    im=axs[0].imshow(F_grid.mean(axis=0).T, cmap='cubehelix_r', aspect='auto', interpolation='auto', origin='lower', extent=[0, 2, 8, 12])
    axs[0].set_ylabel(r'$\log{M}$')
    axs[0].tick_params(axis='x', which='both', top=False, bottom=True, labelbottom=False, length=3, width=2, direction='in')


    axs[1].hist(z, bins=20, range=(0, 2), color='grey', edgecolor='black')

    axs[1].set_xlabel(r"$z$")
    axs[1].set_ylabel(r"$N_{galaxies}$")
    axs[1].set_yscale('log')
    axs[1].set_xlim(0, 2)
    axs[1].tick_params(
        axis='y',
        which='both',
        direction='in'
    )
    axs[1].tick_params(axis='y', which='minor', labelleft=False)
    axs[1].tick_params(axis='x', which='both', top=True, bottom=False, labeltop=False, length=3, width=2, direction='in')
    axs[1].grid(axis='y', alpha=0.5)
    axs[1].set_xticks(np.linspace(0, 2, 6))

    for ax in axs:
        ax.grid(axis='x', alpha=0.5)

    plt.suptitle('The importance of galaxy count', fontsize=24)
    plt.tight_layout()
    fig.patch.set_facecolor('#f5f5f5')
    plt.savefig('./Sampling/Plots/Slides/Signal loss.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

def computation():
    x=[10000, 50000, 80000, 100000]
    y=[0.6, 3.55, 6.57, 8.58]

    fig, ax=plt.subplots(figsize=(6, 3))

    ax.scatter(x, y, color='maroon')
    ax.set_ylim(0, 10)
    ax.set_ylabel('Computation time (hours)')
    ax.set_xlabel('Number of KiDS galaxies')
    ax.grid(axis='both', alpha=0.5)

    fig.patch.set_facecolor('#f5f5f5')

    plt.savefig('./Sampling/Plots/Slides/Computation.png', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())

def main():
    warmup_F_grids=[
        load_F("test_num_warmup_10.npy"),
        load_F("test_num_warmup_50.npy"),
        load_F("test_num_warmup_100.npy"),
        load_F('test_num_warmup_500.npy'),
        load_F('test_num_warmup_1000.npy')
    ]

    show_galaxy_size_grids=[
        load_F("increase_num_gals0.npy"),
        load_F("increase_num_gals1.npy"),
        load_F("increase_num_gals2.npy"),
        load_F("increase_num_gals3.npy"),
    ]

    F_grid_long=load_F("long_chain.npy")
    F_grid=show_galaxy_size_grids[-1]

    #F_samples(F_grid)
    #single_psmf(F_grid, z_idx=6)
    #psmf(F_grid, num_snakes=0)
    #animate_F(F_grid, normalise=False)
    
    #run_all(F_grid)

    #compare_Fs(F_grids=F_grid, comparison='samples')
    #compare_Fs(F_grids=F_grids, comparison='warmups')
    
    #compare_warmup_size_minor(F_grids=warmup_F_grids)
    #compare_sample_size_minor(F_grid=F_grid_long)
    #thinning(F_grid)
    #show_galaxy_size(show_galaxy_size_grids)

    #hist(F_grid.mean(axis=0))

    #signal_loss(F_grid=F_grid)
    #computation()
main()