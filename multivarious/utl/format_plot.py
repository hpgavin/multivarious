def format_plot(font_size, line_width, marker_size):
    '''
    Format plot with specified parameters
    Sets default matplotlib rcParams for consistent plot styling
    
    INPUT         DESCRIPTION                                       DIMENSION
    ----------    -----------------------------------------------   ---------
    font_size     font size for labels and text                        1 x 1
    line_width    line width for plots                                 1 x 1
    marke_rsize   marker size for scatter plots                        1 x 1
    '''

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2,
        'lines.linewidth': line_width,
        'lines.markersize': marker_size,
        'axes.linewidth': line_width / 2
    })
