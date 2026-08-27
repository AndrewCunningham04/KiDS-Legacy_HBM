
import h5py
import numpy as np

def read_h5(filename, show=False):
    fs=h5py.File(filename)
    
    if show:
        print(fs.keys())
        
    return fs

def compute_selection():
    filenames=['/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_I.h5'
                , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_II.h5'
                , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_III.h5'
                , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_IV.h5'
                , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_V.h5'
                , '/projects/public/u6dl/KiDS_Legacy_HBM/Data/mock_catalog_Ch1_26_VI.h5']
    
    zvals_tot, Mvals_tot, zvals_mask, Mvals_mask=[], [], [], []
    
    for filename in filenames:
        data3=read_h5(filename, show=False)
        
        zvals=data3['sps_parameters'][:, -1]
        Mvals=data3['derived_parameters'][:, 1]
        rvals=data3['magnitudes_log'][:, 2]
        
        mask1=(Mvals<=12) & (Mvals>=8) & (zvals<=4.5)
        zvals, Mvals, rvals=zvals[mask1], Mvals[mask1], rvals[mask1]
        
        mask2=(rvals>=20) & (rvals<=23.5)
        
        zvals_mask.append(zvals[mask2])
        Mvals_mask.append(Mvals[mask2])
        
        zvals_tot.append(zvals)
        Mvals_tot.append(Mvals)
        
    zvals_tot=np.concatenate(zvals_tot)
    Mvals_tot=np.concatenate(Mvals_tot)
    
    zvals_mask=np.concatenate(zvals_mask)
    Mvals_mask=np.concatenate(Mvals_mask)
    
    z_edges=np.linspace(0, 4.5, 46)
    M_edges=np.linspace(8, 12, 41)
    
    H, _, _, =np.histogram2d(zvals_tot, Mvals_tot, bins=[z_edges, M_edges])
    H_masked, _, _, =np.histogram2d(zvals_mask, Mvals_mask, bins=[z_edges, M_edges])
    
    s=H_masked/H
    
    np.savetxt(
        "/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/s_array.csv",
        s,
        delimiter=","
    )

    return s

def main():
    compute_selection()

if __name__ == "__main__":
    main()