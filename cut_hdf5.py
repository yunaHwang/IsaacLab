import h5py

src = "datasets/visuomotor-based/clean_clean_clean_gen_dataset_10-ID.hdf5"
dst = "datasets/visuomotor-based/please_clean_clean_gen_dataset_9-ID.hdf5"

keep_demos = [
    "demo_0",
    "demo_1",
    "demo_2",
    "demo_3",
    "demo_4",
]

del_demos = [
    # "demo_11",
    # "demo_3",
    # "demo_12",
    # "demo_9"
    # "demo_0"
    "demo_14"
]

with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:

    fin.copy("data", fout)

    data=fout["data"]

    for demo in list(data.keys()):
        # if demo not in keep_demos:
        #     del data[demo]

        if demo in del_demos:
            del data[demo]

    # update demo count
    data.attrs["num_demos"] = len(keep_demos)

print("saved:", dst)
