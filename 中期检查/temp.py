while dict is not empty:
    oldest= dict[dict.keys()[0]]
    for key in dict.keys():
        if dict[key] is older than dict[oldest]:
            oldest= key
    print(oldest)
    dict.remove(oldest)