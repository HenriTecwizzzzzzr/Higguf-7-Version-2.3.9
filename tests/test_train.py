from higguf.train import infinite_batches


def test_infinite_batches_restarts_after_each_pass() -> None:
    batches = [[1], [2]]
    iterator = infinite_batches(batches)  # type: ignore[arg-type]
    assert [next(iterator) for _ in range(5)] == [[1], [2], [1], [2], [1]]
