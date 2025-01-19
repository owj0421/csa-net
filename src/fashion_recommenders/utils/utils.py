from typing import Iterable, Any, Optional
from itertools import islice
from tqdm import tqdm

def batch_iterable(
    iterable: Iterable[Any],
    batch_size: int,
    desc: Optional[str] = None,
):
    iterator = iter(iterable)
    total = len(iterable) if hasattr(iterable, '__len__') else None
    
    pbar = tqdm(
        total=(total + batch_size - 1) // batch_size if total else None,
        desc=desc,
    )
    
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
        if pbar.total:  # Update progress only if total is known
            pbar.update(1)
