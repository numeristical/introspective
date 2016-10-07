def contig_table(vec1, vec2, normalize=None):
    """Returns a Dataframe that's contingency table of two vectors.

    Usage:
    vec1 = np.random.binomial(10,.2,500)
    vec2 = np.random.binomial(30,.3,500)
    contig_table(asd,asd2)
    """
    from collections import Counter

    vals1 = np.array(list(set(vec1)))  #This is faster than calling np.unique (at least for dtype = 'object')
    vals2 = np.array(list(set(vec2)))
    vals1.sort()
    vals2.sort()
    dim1 = len(vals1)
    dim2 = len(vals2)
    dict1 = {key:value for (key,value) in zip(vals1,range(dim1))}
    dict2 = {key:value for (key,value) in zip(vals2,range(dim2))}
    data_array = dict(Counter(zip(vec1,vec2)))
    output_array = np.zeros((dim1, dim2)).astype(int)
    for (row, col), val in data_array.items():
        output_array[dict1[row], dict2[col]] = val
    if normalize=='by_row':
        row_sums = np.sum(output_array,axis=1)
        output_array = output_array/np.tile(np.array([row_sums]).T,(1,dim2))
    if normalize=='by_col':
        col_sums = np.sum(output_array,axis=0)
        output_array = output_array/np.tile(np.array([col_sums]),(dim1,1))

    return pd.DataFrame(data = output_array,index=vals1,columns=vals2)
