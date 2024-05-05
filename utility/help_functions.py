def find(arr, prop, value):
    """
    Find the first dictionary in the given list that has a specific property value.

    Args:
        arr (list): The list of dictionaries to search.
        prop (str): The property name to check.
        value: The value to search for.

    Returns:
        The first dictionary that has the specified property value, or None if no such dictionary is found.
    """
    for item in arr:
        if item.get(prop) == value:
            return item
    return None