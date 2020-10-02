import sys
from typing import Any, Optional


def handler(args) -> Optional[Any]:
    return


if __name__ == '__main__':
    result = handler(sys.argv)
    exit(0 if result else -1)
