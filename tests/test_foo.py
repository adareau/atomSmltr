from atomsmltr import foo


def test_bar():
    assert (foo.bar(5)) == 6
    return foo.bar(5)


if __name__ == "__main__":
    print(test_bar())
