"""3D amorphous silica (a-SiO2): data, defect detectors, requests and the conditional NequIP
denoiser for the 3D extension (docs/SILICA_3D_PLAN.md, docs/SILICA_QUICKSTART.md).

Species convention (as in DM2's training code, which label-encodes atomic numbers):
0 = O, 1 = Si. Lengths in Angstrom, cooling rates in K/ps.

The model needs the optional dependencies ``pip install -e ".[silica]"`` (e3nn 0.4.4,
torch_geometric, ase, and DM2's ``graphite`` package); data, geometry, defects and requests
need only numpy, scipy and torch.
"""
