from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent

RAW_PATH = ROOT / "data" / "raw" / "encoded_ml_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "county_zscore_split"

TRAIN_ALL_OUTPUT = OUTPUT_DIR / "train_scaled_all_counties.csv"
TEST_ALL_OUTPUT = OUTPUT_DIR / "test_scaled_all_counties.csv"

SCALER_OUTPUT_DIR = OUTPUT_DIR / "scalers"
COUNTY_OUTPUT_DIR = OUTPUT_DIR / "by_county"

NON_SCALE_COLUMNS = {
    "month",
    "date",
    "year",
    "county_label",
    "industry_label",
    "carrier_type_label",
    "donation_ratio",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    if "month" not in df.columns:
        raise ValueError("encoded_ml_dataset.csv is missing required column: month")

    if "county_label" not in df.columns:
        raise ValueError("encoded_ml_dataset.csv is missing required column: county_label")

    df["date"] = pd.to_datetime(df["month"].astype(str) + "-01")
    df["year"] = df["date"].dt.year

    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["year"] < 2024].copy()
    test_df = df[df["year"] >= 2024].copy()

    return train_df, test_df


def get_scale_columns(train_df: pd.DataFrame) -> list[str]:
    numeric_columns = train_df.select_dtypes(include="number").columns.tolist()

    scale_columns = [
        col
        for col in numeric_columns
        if col not in NON_SCALE_COLUMNS
    ]

    return scale_columns


def split_by_county(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    county_frames = {
        int(county_label): group_df.copy()
        for county_label, group_df in df.groupby("county_label", sort=True)
    }

    return county_frames


def scale_each_county_group(
    train_groups: dict[int, pd.DataFrame],
    test_groups: dict[int, pd.DataFrame],
    scale_columns: list[str],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[int, StandardScaler]]:
    scaled_train_groups = {}
    scaled_test_groups = {}
    scalers = {}

    for county_label, train_group in train_groups.items():
        train_scaled = train_group.copy()

        test_group = test_groups.get(county_label)
        if test_group is None:
            test_group = pd.DataFrame(columns=train_group.columns)

        test_scaled = test_group.copy()

        scaler = StandardScaler()

        train_scaled[scale_columns] = scaler.fit_transform(
            train_group[scale_columns]
        )

        if len(test_scaled) > 0:
            test_scaled[scale_columns] = scaler.transform(
                test_group[scale_columns]
            )

        scaled_train_groups[county_label] = train_scaled
        scaled_test_groups[county_label] = test_scaled
        scalers[county_label] = scaler

    return scaled_train_groups, scaled_test_groups, scalers


def save_outputs(
    scaled_train_groups: dict[int, pd.DataFrame],
    scaled_test_groups: dict[int, pd.DataFrame],
    scalers: dict[int, StandardScaler],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COUNTY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_test = []

    for county_label, train_group in scaled_train_groups.items():
        county_dir = COUNTY_OUTPUT_DIR / f"county_{county_label}"
        county_dir.mkdir(parents=True, exist_ok=True)

        test_group = scaled_test_groups.get(county_label)

        train_output_path = county_dir / "train_scaled.csv"
        test_output_path = county_dir / "test_scaled.csv"
        scaler_output_path = SCALER_OUTPUT_DIR / f"county_{county_label}_standard_scaler.joblib"

        train_group.to_csv(train_output_path, index=False)

        if test_group is not None:
            test_group.to_csv(test_output_path, index=False)

        joblib.dump(scalers[county_label], scaler_output_path)

        all_train.append(train_group)

        if test_group is not None and len(test_group) > 0:
            all_test.append(test_group)

    train_all_df = pd.concat(all_train, axis=0).sort_index()
    test_all_df = pd.concat(all_test, axis=0).sort_index() if all_test else pd.DataFrame()

    train_all_df.to_csv(TRAIN_ALL_OUTPUT, index=False)
    test_all_df.to_csv(TEST_ALL_OUTPUT, index=False)


def main() -> None:
    df = load_dataset()

    train_df, test_df = split_train_test(df)

    scale_columns = get_scale_columns(train_df)

    print("Raw dataset shape:", df.shape)
    print("Train dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)
    print("Train years:", sorted(train_df["year"].unique().tolist()))
    print("Test years:", sorted(test_df["year"].unique().tolist()))
    print("Scale columns:", scale_columns)

    if "year" in scale_columns:
        raise ValueError("year should not be included in scale_columns.")

    train_groups = split_by_county(train_df)
    test_groups = split_by_county(test_df)

    print("Number of train county groups:", len(train_groups))
    print("Number of test county groups:", len(test_groups))

    scaled_train_groups, scaled_test_groups, scalers = scale_each_county_group(
        train_groups=train_groups,
        test_groups=test_groups,
        scale_columns=scale_columns,
    )

    save_outputs(
        scaled_train_groups=scaled_train_groups,
        scaled_test_groups=scaled_test_groups,
        scalers=scalers,
    )

    print("Done.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
