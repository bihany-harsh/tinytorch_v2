import tinytorch._core as _core
from typing import List, Union, Any
import numbers

def _flatten_and_get_shape(data: Any) -> tuple[List[float], List[int]]:
    # function to handle arbitrary python lists and provide to the cpp backend
    
    def get_shape(lst):
        if not isinstance(lst, (list, tuple)):
            return []
        if len(lst) == 0:
            return [0]
        
        inner_shape = get_shape(lst[0])
        
        # verify consistency of shape
        for item in lst[1:]:
            if get_shape(item) != inner_shape:
                raise ValueError("Invalid tensor shape, tensor rows should be of same size.")
            
        return [len(lst)] + inner_shape
    
    def flatten(lst):
        if not isinstance(lst, (list, tuple)):
            if not isinstance(lst, numbers.Number):
                raise TypeError(f"Expected numeric type, receieved {type(lst)}")
            return [lst]

        result = []
        for item in lst:
            result.extend(flatten(item))
        return result
    
    return flatten(data), get_shape(data)

def tensor(data: Union[List, float, int], dtype=None):
    """
    Create a tensor:
        Args:
            data: Nested python list, scalar or flat list
            dtype: Data type (Float32, Float64, Int32, Int64, Bool)
    """
    
    if isinstance(data, numbers.Number):
        data = [data]
        shape = [1]
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], numbers.Number):
        shape = [len(data)]
    else:
        data, shape = _flatten_and_get_shape(data)
        
    if dtype is None:
        dtype = _core.Dtype.Float32
        
    return _core.tensor.Tensor(data, shape, dtype)