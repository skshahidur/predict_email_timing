import sys

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
        required_minor = 7
    else:
        raise ValueError("Unrecognized python: {}".format(REQUIRED_PYTHON))

    if (system_major != required_major) or (system_minor != required_minor):
        raise TypeError("This project requires Python {} but found Python {}".format(required_major, sys.version))
    else:
        print("Passed all the tests")


if __name__ == '__main__':
    main()
