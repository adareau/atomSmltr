def test_info_string():
    from atomsmltr.utils.infostring import InfoString

    info = InfoString("LASER PARAMETERS")
    info.add_section("Settings")
    info.add_element("polarization", "Circular")
    info.add_element("waist", f"{25.4e-6:.2e}µm")
    info.add_element("power", f"{15.8:.2f} mW")
    info.add_section("Another section")
    info.add_element("elem", "param")
    info.add_element("elem2", "param2")

    print(info.generate())


if __name__ == "__main__":
    test_info_string()
