import h5py

file = "datasets/visuomotor-based/annotated_dataset-OOD-5.hdf5"

with h5py.File(file, "r") as f:
	print("data attrs: ")
	for k, v in f["data"].attrs.items():
		print(k, "=", v)
