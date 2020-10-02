import os
import pickle
from typing import Any, Optional


def get_project_root(prj_name: str):
    current_path, filename = os.path.split(os.path.abspath(__file__))
    components = current_path.split(os.sep)
    # noinspection PyTypeChecker
    return str.join(os.sep, components[:components.index(prj_name) + 1])


def pickleize_object(file_path: str, obj: Any):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    return


def get_pickled_object(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        return None


if __name__ == '__main__':
    project_dir = get_project_root('ml-trials')
    _file_path = os.path.join(project_dir, "shared", ".transient", "test.pickle")

    o = get_pickled_object(_file_path)
    assert o is None
    pickleize_object(_file_path, ['some', 'random', 'obj'])

    o = get_pickled_object(_file_path)
    assert o is not None

    os.remove(_file_path)
