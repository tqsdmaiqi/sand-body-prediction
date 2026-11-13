from dataset import read_csv


def describe_group(data, group):
    return (
        data.groupby(group)
        .agg(
            sample_count=("sand_thickness", "size"),
            thickness_min=("sand_thickness", "min"),
            thickness_mean=("sand_thickness", "mean"),
            thickness_max=("sand_thickness", "max"),
            weight_mean=("sample_weight", "mean"),
        )
        .reset_index()
    )


def run(*, params):
    data = read_csv(params["local_train_csv"], options=params["csv_options"])
    output = params["out_dir"]
    output.mkdir(parents=True, exist_ok=True)
    describe_group(data, "sample_type").to_csv(
        output / "03_statistics_by_sample_type.csv", index=False
    )
    virtual = data[data["sample_type"] == "virtual"]
    describe_group(virtual, "neighbor_order").to_csv(
        output / "03_statistics_by_neighbor_order.csv", index=False
    )
