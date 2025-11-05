def test_import_example_module():

    # Check that the module imports correctly
    import atomsmltr.examples.atomsmltr2025 as atomsmltr2025

    configs = [
        "config_1D_MOT_Yb",
        "config_3D_MOT_Yb",
        "config_Doppler_limit",
    ]
    for conf in configs:
        assert hasattr(atomsmltr2025, conf)
        cfg = getattr(atomsmltr2025, conf)
        cfg.print_info()


if __name__ == "__main__":
    test_import_example_module()
