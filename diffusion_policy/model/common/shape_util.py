from typing import Dict, List, Tuple, Callable, Optional
import torch
import torch.nn as nn

def get_module_device(m: nn.Module):
    device = torch.device('cpu')
    try:
        param = next(iter(m.parameters()))
        device = param.device
    except StopIteration:
        pass
    return device

@torch.no_grad()
def get_output_shape(
        input_shape: Tuple[int],
        net: Callable[[torch.Tensor], torch.Tensor]
    ):  
        device = get_module_device(net)
        test_input = torch.zeros((1,)+tuple(input_shape), device=device)
        test_output = net(test_input)
        output_shape = tuple(test_output.shape[1:])
        return output_shape

def assert_shape(
    tensor: torch.Tensor, 
    expected_shape: Tuple[Optional[int], ...],
    name: str = "Tensor"
):
    """
    Asserts that a tensor has a given shape, allowing for wildcard dimensions.

    Args:
        tensor (torch.Tensor): The tensor to check.
        expected_shape (Tuple[Optional[int], ...]): 
            The expected shape. Use `None` as a wildcard for any dimension size.
        name (str, optional): 
            The name of the tensor for clearer error messages. Defaults to "Tensor".

    Raises:
        AssertionError: If the tensor's shape does not match the expected shape.
    """
    # Check that a tensor was passed
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input '{name}' must be a torch.Tensor, but got {type(tensor).__name__}.")

    actual_shape = tuple(tensor.shape)
    
    # Check if the number of dimensions is correct
    if len(actual_shape) != len(expected_shape):
        raise AssertionError(
            f"Shape mismatch for '{name}'. "
            f"Expected {len(expected_shape)} dimensions, but got {len(actual_shape)}. "
            f"Expected shape: {expected_shape}, Actual shape: {actual_shape}"
        )

    # Check each dimension
    for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
        if expected_dim is not None and actual_dim != expected_dim:
            raise AssertionError(
                f"Shape mismatch for '{name}' at dimension {i}. "
                f"Expected shape: {expected_shape}, Actual shape: {actual_shape}"
            )
