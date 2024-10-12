from typing import Any, Iterable, List


def chunks(lst: Iterable, n: int) -> List[Any]:
    # https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        i_start = i
        i_end = i + n
        yield lst[i_start:i_end]
