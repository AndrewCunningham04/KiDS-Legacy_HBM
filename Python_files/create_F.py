#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:18:57 2026

@author: ac2938
"""
import h5py
import numpy as np
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import time
from jax.ops import segment_sum
from numpyro.diagnostics import summary
import argparse
import os

def read_h5(filename, show=False):
    fs=h5py.File(filename)
    
    if show:
        print(fs.keys())
        
    return fs


def compile_P_counts(zmax_idx, selection=False, write=False):
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
        
        H, xedges, yedges=np.histogram2d(zvals_tot, Mvals_tot, bins=(45, 40))

        np.savetxt(csv_name, H, delimiter=",")

    H=H[:zmax_idx+1, :]

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
        
        #norms.append(np.sum(vals_new))
        norms.append(512)

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

def read_selection(selection, zmax_index):
    if selection:
            s=np.loadtxt("/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/s_array.csv", delimiter=",")
            
    else:
        s=np.zeros(shape=(45, 40))

    s=s[:zmax_index+1, :]

    return s

def log_likelihood(F, indices, C_values, galaxy_ids, num_gals, s, delta, selection):
    F_flat = F.flatten()

    terms = C_values * F_flat[indices]

    galaxy_sums = segment_sum(
        terms,
        galaxy_ids,
        num_segments=num_gals
    )

    galaxy_terms = jnp.log(
        delta * galaxy_sums
    )

    if selection:
        selection_term = delta * jnp.sum(F * s)
    else:
        selection_term = 0.0

    return galaxy_terms.sum() - selection_term

def model(indices, C_values, galaxy_ids, num_gals, alpha, shape, s, delta, selection):
    F_flat = numpyro.sample(
        "F",
        dist.Dirichlet(alpha)
    )

    F = F_flat.reshape(shape)

    numpyro.factor(
        "likelihood",
        log_likelihood(
            F,
            indices,
            C_values,
            galaxy_ids,
            num_gals,
            s,
            delta,
            selection
        )
    )

def sample(num_warmup, num_samples, num_gals, zmax_index, selection=False, num_chains=1, rebin=False,
           filename="F_grid.npy"):
    s = read_selection(selection, zmax_index)

    indices, values, norms = read_sparse_Q(
        selection,
        zmax_index,
        num_gals=num_gals
    )

    lengths = np.array(
        [len(x) for x in indices],
        dtype=np.int32
    )

    idx_flat = np.concatenate(indices).astype(np.int32)
    value_flat = np.concatenate(values).astype(np.float32)
    galaxy_ids = np.repeat(
        np.arange(num_gals, dtype=np.int32),
        lengths
    )

    H = compile_P_counts(zmax_index, selection).astype(np.float32)
    P = H / H.sum()

    idx_flat = jnp.array(idx_flat)
    value_flat = jnp.array(value_flat)
    galaxy_ids = jnp.array(galaxy_ids)
    norms = jnp.array(norms, dtype=jnp.float32)
    P = jnp.array(P, dtype=jnp.float32)
    s = jnp.array(s, dtype=jnp.float32)

    C_values = (
        (value_flat / norms[galaxy_ids])
        / P.flatten()[idx_flat]
    )

    C_values = jnp.asarray(C_values, dtype=jnp.float32)

    alpha = jnp.ones(P.size)

    P_shape=P.shape

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized"
    )

    start = time.time()

    mcmc.run(
        jax.random.key(0),
        idx_flat,
        C_values,
        galaxy_ids,
        num_gals,
        alpha,
        P_shape,
        s,
        delta=1,
        selection=selection,
        extra_fields=("num_steps",)
    )
    end = time.time()

    print(f"MCMC time: {((end - start)/3600):.2f} hours", flush=True)

    samples = mcmc.get_samples()

    if num_chains>1:
        samples_by_chain = mcmc.get_samples(group_by_chain=True)  # shape (num_chains, num_samples, ...)
        summary_dict = summary(samples_by_chain)

        r_hat = summary_dict["F"]["r_hat"]  # array, same shape as one F sample
        print("Max r_hat:", r_hat.max())
        print("Mean r_hat:", r_hat.mean())
        print("Frac > 1.01:", (r_hat > 1.01).mean())

    F_grid = samples["F"].reshape(
        -1,
        P.shape[0],
        P.shape[1]
    )

    path="/projects/public/u6dl/KiDS_Legacy_HBM/Saved_arrays/F/"
    full_path=os.path.join(path, filename)

    np.save(
        full_path,
        np.array(F_grid)
    )

    print(f"Saved F_grid to file as {filename}; selection={selection}\n{num_warmup} warmup, {num_samples} samples, {num_gals} galaxies", flush=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_warmup", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_gals", type=int, default=10000)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--filename", type=str, default="F_grid.npy")

    args = parser.parse_args()

    print(f"\nStarting run:\n{args.num_chains} chains")
    print(f"Selection {args.selection}")

    sample(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_gals=args.num_gals,
        zmax_index=19,
        selection=args.selection,
        num_chains=args.num_chains,
        rebin=False,
        filename=args.filename
    )

if __name__=="__main__":
    main()