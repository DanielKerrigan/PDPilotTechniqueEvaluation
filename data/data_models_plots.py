"""Download datasets, train models, and calculate PDP and ICE plots."""

import argparse
import optuna
from train_models import train_models
import pandas as pd


def main(trial, jobs):
    """Main method for script."""

    if trial:
        results_file = "trial_datasets_and_models.json"

        regression_datasets = [
            "522_pm10",
            "547_no2",
            "583_fri_c1_1000_50",
            "666_rmftsa_ladata",
            "feynman_II_36_38",
            "feynman_test_6",
        ][0:2]

        classification_datasets = [
            "breast_w",
            "crx",
            "GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1",
            "irish",
            "monk1",
            "monk2",
            "monk3",
            "pima",
            "profb",
            "threeOf9",
            "tic_tac_toe",
            "tokyo1",
            "wdbc",
            "xd6",
        ][0:2]

    else:
        results_file = "datasets_and_models.json"

        regression_datasets = [
            "1191_BNG_pbc",
            "1193_BNG_lowbwt",
            "1196_BNG_pharynx",
            "1199_BNG_echoMonths",
            "1201_BNG_breastTumor",
            "1203_BNG_pwLinear",
            "197_cpu_act",
            "215_2dplanes",
            "225_puma8NH",
            "344_mv",
            "503_wind",
            "529_pollen",
            "537_houses",
            "564_fried",
            "574_house_16H",
            "588_fri_c4_1000_100",
            "feynman_I_9_18",
            "feynman_test_1",
        ]

        classification_datasets = [
            "adult",
            "banana",
            "chess",
            "churn",
            "clean2",
            "coil2000",
            "dis",
            "GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1",
            "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001",
            "german",
            "Hill_Valley_with_noise",
            "hypothyroid",
            "magic",
            "mofn_3_7_10",
            "mushroom",
            "parity5+5",
            "phoneme",
            "ring",
            "spambase",
            "titanic",
            "twonorm",
        ]

    regression_results = train_models(regression_datasets, "regression", jobs=jobs)

    classification_results = train_models(classification_datasets, "binary", jobs=jobs)

    results = pd.concat([regression_results, classification_results]).reset_index(
        drop=True
    )

    results.to_json(results_file, orient="records", indent=2, index=False)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Download datasets, train models, and calculate PDP and ICE plots.",
    )
    parser.add_argument(
        "-t", "--trial", action="store_true", help="trial run with different datasets"
    )
    parser.add_argument(
        "-j", "--jobs", help="number of jobs to use", default=1, type=int
    )
    args = parser.parse_args()

    main(args.trial, args.jobs)
