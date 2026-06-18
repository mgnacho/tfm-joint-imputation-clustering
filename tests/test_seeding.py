from joint_imputation_clustering.utils.seeding import stable_seed


def test_stable_seed_is_deterministic_and_order_sensitive():
    assert stable_seed(60, 2, "mask") == stable_seed(60, 2, "mask")
    assert stable_seed(60, 2, "mask") != stable_seed(2, 60, "mask")
