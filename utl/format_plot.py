def format_plot(fontsize, linewidth, markersize):
    '''
    Format plot with specified parameters
    Sets default matplotlib rcParams for consistent plot styling
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    fontsize    font size for labels and text                            1 x 1
    linewidth   line width for plots                                     1 x 1
    markersize  marker size for scatter plots                            1 x 1
    '''

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'legend.fontsize': fontsize - 2,
        'lines.linewidth': linewidth,
        'lines.markersize': markersize,
        'axes.linewidth': linewidth / 2
    })
