import h5py

src = "datasets/visuomotor-based/annotated_dataset-OOD.hdf5"
dst = "datasets/visuomotor-based/annotated_dataset-OOD-5.hdf5"

keep_demos = [
    "demo_0",
    "demo_1",
    "demo_2",
    "demo_3",
    "demo_4",
]

with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:

    fin.copy("data", fout)

    data=fout["data"]

    for demo in list(data.keys()):
        if demo not in keep_demos:
            del data[demo]

    # update demo count
    data.attrs["num_demos"] = len(keep_demos)

print("saved:", dst)
