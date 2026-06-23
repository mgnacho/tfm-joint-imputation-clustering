from joint_imputation_clustering.analysis.runtime import (
    baseline_runtime_record,
    proposed_runtime_record,
)


def test_proposed_runtime_includes_all_candidates_and_both_model_calls() -> None:
    row = proposed_runtime_record(
        candidate_generation_runtime=3.0,
        train_total_runtime=10.0,
        test_total_runtime=2.0,
        data_loading_runtime=1.0,
        preprocessing_runtime=0.5,
        missingness_runtime=0.25,
    )
    assert row["pipeline_runtime"] == 15.0
    assert row["end_to_end_runtime"] == 16.75


def test_baseline_runtime_uses_only_its_imputer_and_clustering() -> None:
    row = baseline_runtime_record(
        imputation_runtime=2.0,
        clustering_runtime=1.5,
        data_loading_runtime=1.0,
        preprocessing_runtime=0.5,
        missingness_runtime=0.25,
    )
    assert row["pipeline_runtime"] == 3.5
    assert row["end_to_end_runtime"] == 5.25
