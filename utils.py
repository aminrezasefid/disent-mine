def to_numpy(array) -> np.ndarray:
    """
    Handles converting any array like object to a numpy array.
    specifically with support for a tensor
    """
    # TODO: replace... maybe with kornia
    if torch.is_tensor(array):
        return array.cpu().detach().numpy()
    # recursive conversion
    # not super efficient but allows handling of PIL.Image and other nested data.
    elif isinstance(array, (list, tuple)):
        return np.stack([to_numpy(elem) for elem in array], axis=0)
    else:
        return np.array(array)

def generate_batch_factor_code(
    dataset,
    representation_function,
    num_points: int,
    batch_size: int,
    show_progress: bool = False,
):
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations, current_factors = dataset.dataset_sample_batch_with_factors(
            num_points_iter, mode="input"
        )
        if i == 0:
            factors = current_factors
            representations = to_numpy(representation_function(current_observations))
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, to_numpy(representation_function(current_observations))))
        i += num_points_iter
        bar.update(num_points_iter)
    return np.transpose(representations), np.transpose(factors)