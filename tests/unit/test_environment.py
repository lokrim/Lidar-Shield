def test_declared_stage1_dependencies_import() -> None:
    import jinja2
    import matplotlib
    import numpy
    import pandas
    import pyarrow
    import pydantic
    import pypcd4
    import quaternion
    import scipy
    import seaborn
    import sklearn
    import yaml

    modules = (
        jinja2,
        matplotlib,
        numpy,
        pandas,
        pyarrow,
        pydantic,
        pypcd4,
        quaternion,
        scipy,
        seaborn,
        sklearn,
        yaml,
    )
    assert all(module.__name__ for module in modules)
