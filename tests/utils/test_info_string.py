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

    info2 = InfoString("Environment")
    info2.add_section("Laser list")
    info2.add_element("laser 1")
    info2.add_element("laser 2")
    info2.absorb_section(info, "Settings", "Laser 1 - Settings")

    print(info2.generate())

    info3 = InfoString("MAG FIELDS")
    info3.add_section("Settings A")
    info3.add_element("param1", "val1")
    info3.add_element("param2", "val2")
    info3.add_section("Settings B")
    info3.add_element("param1", "val1")
    info3.add_element("param2", "val2")

    info2.merge(info3, prefix="(mag fields) ")
    print(info2.generate())

    info3.rm_element("param1", "Settings A")
    print(info3.generate())


if __name__ == "__main__":
    test_info_string()
