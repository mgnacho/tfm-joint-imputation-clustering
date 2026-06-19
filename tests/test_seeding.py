from joint_imputation_clustering.utils.seeding import (
    normalize_gurobi_seed,
    stable_seed,
)


def test_stable_seed_is_deterministic_and_order_sensitive():
    assert stable_seed(60, 2, "mask") == stable_seed(60, 2, "mask")
    assert stable_seed(60, 2, "mask") != stable_seed(2, 60, "mask")


def test_gurobi_seed_is_always_in_valid_range():
    seeds = [
        0,
        1,
        2_000_000_000,
        2_000_000_001,
        3_020_358_935,
        4_241_891_574,
        4_294_967_295,
    ]

    for seed in seeds:
        normalized = normalize_gurobi_seed(seed)
        assert 0 <= normalized <= 2_000_000_000