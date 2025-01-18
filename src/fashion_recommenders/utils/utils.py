from tqdm import tqdm
from typing import List, Any, Optional, Dict, Union, Tuple, Iterable



def batch_iterable(
    iterable: Iterable[Any],
    batch_size: int,
    desc: Optional[str] = None,
):
    pbar = tqdm(
        range(0, len(iterable), batch_size), 
        desc=desc, 
        total=(len(iterable) + batch_size - 1) // batch_size
    )
    
    for i in pbar:
        yield iterable[i:i + batch_size]