"""Import smoke tests for the MARL-BattleGrounds package."""


def test_package_imports() -> None:
    """The package should be importable after installation."""
    import marl_battlegrounds

    assert marl_battlegrounds.__name__ == "marl_battlegrounds"
