import argopy
from argopy import DataFetcher as ArgoDataFetcher
import os

WMO = 6902746  # replace with your float WMO number

print(f"Fetching all data for float {WMO}...")

loader = ArgoDataFetcher(src="erddap").float(WMO)
ds = loader.to_xarray()
ds = ds.argo.point2profile()

out_dir = f"argo_{WMO}"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, f"{WMO}_profiles.nc")
ds.to_netcdf(out_path)

print(f"Saved {len(ds.N_PROF)} profiles to {out_path}")
print(ds)