"""Check that the package can be imported through its public name."""


def test_package_imports() -> None:
    import marl_battlegrounds

    assert marl_battlegrounds.__name__ == "marl_battlegrounds"
