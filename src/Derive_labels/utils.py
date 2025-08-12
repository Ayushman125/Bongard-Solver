def robust_flatten_and_stringify(lst):
    result = []
    if isinstance(lst, list):
        for x in lst:
            if isinstance(x, list):
                result.extend(robust_flatten_and_stringify(x))
            elif hasattr(x, 'raw_command'):
                result.append(str(x.raw_command))
            elif type(x).__name__ in ['LineAction', 'ArcAction']:
                result.append(str(getattr(x, 'raw_command', str(x))))
            else:
                result.append(str(x))
    else:
        result.append(str(lst))
    return result
