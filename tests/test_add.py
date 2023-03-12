import pytest

from open_layout_lm.add import add


@pytest.mark.parametrize(
    ("a", "b", "expected_result"),
    [
        (1, 2, 3),
        (-1, 1, 0)
    ]
)
def test_add(a: int,b: int, expected_result: int) -> None:
    assert add(a, b) == expected_result
