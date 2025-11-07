def rainbow(n):
    '''
    Generate rainbow colormap with n colors
    Returns an n x 3 array of RGB colors spanning the rainbow spectrum
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    n           number of colors to generate                              1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    cMap        RGB color array                                           n x 3
    '''

    # Use matplotlib's rainbow/jet colormap
    cmap = cm.get_cmap('rainbow', n)
    colors = np.array([cmap(i)[:3] for i in range(n)]) # Get RGB, exclude alpha

    return colors

