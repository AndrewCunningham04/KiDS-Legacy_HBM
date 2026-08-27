import h5py
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def read_h5(filename, show=False):
    fs=h5py.File(filename)
    
    if show:
        print(fs.keys())
        
    return fs

def compile_P_counts(selection=False, write=False, bins=(45, 40)):
    if selection:
        csv_name="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/P_counts_selected.csv"

    else:
        csv_name="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/P_counts.csv"
        
    if not write:
        H = np.loadtxt(csv_name, delimiter=",")
        
    else:
        filenames=['/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_I.h5'
                   , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_II.h5'
                   , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_III.h5'
                   , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_IV.h5'
                   , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_V.h5'
                   , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_VI.h5']
        
        zvals_tot, Mvals_tot=[], []
        
        for filename in filenames:
            data3=read_h5(filename, show=False)
            zvals=data3['sps_parameters'][:, -1]
            Mvals=data3['derived_parameters'][:, 1]
            
            if selection:
                selection_mask=(data3['magnitudes_log'][:, 2]<=23.5) & (data3['magnitudes_log'][:, 2]>=20)
                
                zvals=zvals[selection_mask]
                Mvals=Mvals[selection_mask]
            
            zvals_tot.append(zvals)
            Mvals_tot.append(Mvals)
            
        zvals_tot=np.concatenate(zvals_tot)
        Mvals_tot=np.concatenate(Mvals_tot)

        mask=(Mvals_tot<=12) & (Mvals_tot>=8) & (zvals_tot<=4.5)
        zvals_tot, Mvals_tot=zvals_tot[mask], Mvals_tot[mask]
        
        H, xedges, yedges=np.histogram2d(zvals_tot, Mvals_tot, bins=bins)
        
        np.savetxt(csv_name, H, delimiter=",")

    return H

def create_sparse_Q_array(chunk_size, selection, n_files=41):
    H = compile_P_counts(selection=selection).astype(np.float32)
    P = H / np.sum(H)

    P_mask = P == 0
    h5_file = '/projects/public/u6dl/WLCAT_MCMC_files/mcmc_histograms_0-40894394.h5'

    with h5py.File(h5_file, "r") as f:
        QN = f["QN"]
        total_gals = QN.shape[0]

        if selection:
            ks = [
                fits.open(
                    '/projects/public/u6dl/KiDS_DR5/KiDS_Legacy_WLCAT/'
                    'WLCAT_ESOmatched_files/WLCAT_ESOmatched_{:d}_{:d}.fits'
                    .format(1000000*n, 1000000*(n+1))
                )
                for n in range(n_files-1)
            ]

            if n_files == 41:
                ks.append(
                    fits.open(
                        '/projects/public/u6dl/KiDS_DR5/KiDS_Legacy_WLCAT/WLCAT_ESOmatched_files/WLCAT_ESOmatched_40000000_40894394.fits'
                    )
                )

            rs = np.zeros(
                np.min([40894394, n_files*1000000]),
                dtype=np.float32
            )

            for n, k in enumerate(ks):
                start = 1000000*n
                end = 1000000*(n+1) if n < 40 else 40894394
                rs[start:end] = k[1].data['MAG_AUTO']
                k.close()  # release the fits file once its data is copied out

            selection_mask = (rs >= 20) & (rs <= 23.5)
            num_selected = int(selection_mask.sum())

        else:
            selection_mask = np.ones(total_gals, dtype=bool)
            num_selected = total_gals

        print(f"Creating sparse Q array for {num_selected} galaxies", flush=True)

        if selection:
            output_file = (
                "/projects/public/u6dl/KiDS_Legacy_HBM/"
                "Saved_arrays/Q_sparse_selected.h5"
            )
        else:
            output_file = (
                "/projects/public/u6dl/KiDS_Legacy_HBM/"
                "Saved_arrays/Q_sparse.h5"
            )

        with h5py.File(output_file, "w") as fout:
            index_dtype = h5py.vlen_dtype(np.dtype("uint16"))
            value_dtype = h5py.vlen_dtype(np.dtype("uint16"))
            norm_dtype = np.dtype("uint16")

            indices_dataset = fout.create_dataset(
                "indices", shape=(num_selected,), dtype=index_dtype
            )
            values_dataset = fout.create_dataset(
                "values", shape=(num_selected,), dtype=value_dtype
            )
            norm_dataset = fout.create_dataset(
                "norm", shape=(num_selected,), dtype=norm_dtype
            )

            out_pos = 0  # running write position into the *selected* output arrays

            for start in range(0, total_gals, chunk_size):
                end = min(start + chunk_size, total_gals)

                local_mask = selection_mask[start:end]
                n_local = int(local_mask.sum())

                if n_local == 0:
                    continue

                print(f"Processing galaxies {start}:{end} "
                      f"({n_local} selected)", flush=True)

                # contiguous read straight off disk
                block = np.asarray(QN[start:end, :, :], dtype=np.uint16)

                # keep only the selected rows within this block
                counts = block[local_mask]
                del block

                # remove bins where P=0
                counts[:, P_mask] = 0

                # flatten z,M dimensions
                counts_flat = counts.reshape(n_local, -1)
                del counts

                for i in range(n_local):
                    mask = counts_flat[i] > 0

                    idx = np.where(mask)[0].astype(np.uint16)
                    vals = counts_flat[i, mask].astype(np.uint16)
                    norm = counts_flat[i].sum()

                    indices_dataset[out_pos + i] = idx
                    values_dataset[out_pos + i] = vals
                    norm_dataset[out_pos + i] = norm

                out_pos += n_local
                del counts_flat

        print("Finished writing:", output_file, flush=True)

def create_z_array(selection, n_files=41):
    output_file = (
        "/projects/public/u6dl/KiDS_Legacy_HBM/"
        "Saved_arrays/Z_selected.h5"
        if selection else
        "/projects/public/u6dl/KiDS_Legacy_HBM/"
        "Saved_arrays/Z.h5"
    )

    # First determine the total number of selected galaxies
    if selection:
        num_selected = 0

        for n in range(n_files):

            if n < 40:
                filename = (
                    '/projects/public/u6dl/KiDS_DR5/KiDS_Legacy_WLCAT/'
                    'WLCAT_ESOmatched_files/'
                    f'WLCAT_ESOmatched_{1000000*n}_{1000000*(n+1)}.fits'
                )
            else:
                filename = (
                    '/projects/public/u6dl/KiDS_DR5/KiDS_Legacy_WLCAT/'
                    'WLCAT_ESOmatched_files/'
                    'WLCAT_ESOmatched_40000000_40894394.fits'
                )

            print(f"Counting selected galaxies in file {n+1}/{n_files}",
                  flush=True)

            with fits.open(filename, memmap=True) as k:
                data = k[1].data

                mask = (
                    (data['MAG_AUTO'] >= 20) &
                    (data['MAG_AUTO'] <= 23.5)
                )

                num_selected += mask.sum()

        print(
            f"Total selected galaxies: {num_selected}",
            flush=True
        )

    else:
        # 40 × 1,000,000 + final file
        num_selected = 40_894_394

    # Create output HDF5 file
    with h5py.File(output_file, "w") as fout:

        z_dataset = fout.create_dataset(
            "Z_B",
            shape=(num_selected,),
            dtype=np.float32
        )

        out_pos = 0

        # Process one FITS file at a time
        for n in range(n_files):

            if n < 40:
                filename = (
                    '/projects/public/u6dl/KiDS_DR5/KiDS_Legacy_WLCAT/'
                    'WLCAT_ESOmatched_files/'
                    f'WLCAT_ESOmatched_{1000000*n}_{1000000*(n+1)}.fits'
                )
            else:
                filename = (
                    '/projects/public/u6dl/KiDS_DR5/KiDS_Legacy_WLCAT/'
                    'WLCAT_ESOmatched_files/'
                    'WLCAT_ESOmatched_40000000_40894394.fits'
                )

            print(
                f"Processing file {n+1}/{n_files}: {filename}",
                flush=True
            )

            with fits.open(filename, memmap=True) as k:
                data = k[1].data

                if selection:
                    mask = (
                        (data['MAG_AUTO'] >= 20) &
                        (data['MAG_AUTO'] <= 23.5)
                    )

                    z = np.asarray(
                        data['Z_B'][mask],
                        dtype=np.float32
                    )

                else:
                    z = np.asarray(
                        data['Z_B'],
                        dtype=np.float32
                    )

                n_z = len(z)

                z_dataset[out_pos:out_pos + n_z] = z

                out_pos += n_z

                print(
                    f"  wrote {n_z} galaxies "
                    f"(total: {out_pos}/{num_selected})",
                    flush=True
                )

                del z

    print(
        f"Finished writing {num_selected} Z_B values to {output_file}",
        flush=True
    )

def main():
    #create_sparse_Q_array(chunk_size=1000000, selection=False, n_files=41)
    #create_sparse_Q_array(chunk_size=1000000, selection=True, n_files=41)
    create_z_array(selection=False)

if __name__ == "__main__":
    main()