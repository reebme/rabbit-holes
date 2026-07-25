import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def pretty_histogram(
                    data_column: pd.Series,
                    bins_no: int | None = None,
                    title: str | None = None,
                    x_axis_title: str | None = None,
                    y_axis_title: str | None = None,
                    density: bool | None = None,
                    vertical_lines: list[tuple(float, str)] | None = None,
                    density_fn: list[tuple[Callable[[np.ndarray], np.ndarray], str, str, str]] | None = None,
                    annotation: dict | list[dict] | None = None,
                    save_file_name: str | None = None):
    '''
    Plot a histogram of a numeric variable.

    The number of bins may be specified explicitly 
    or determined automatically 
    using Matplotlib's default bin selection algorithm.

    Parameters
    ----------
    data_column : pd.Series
        Numeric data to be visualized.

    bins_no : int | None, default = None
        Number of histogram bins. If None, Matplotlib's
        automatic bin selection ('auto') is used.

    title : str | None, default = None
        Plot title.

    x_axis_title : str | None, default = None
        Label for the x-axis.

    y_axis_title : str | None, default = None
        Label for the y-axis.
        
    vertical_lines: list[float] | None = None
        Optional dashed vertical lines at values in the list.

    save_file_name : str | None, default = None
        Path where the figure should be saved. If None,
        the figure is displayed but not saved.

    Returns
    -------
    None
        Displays the histogram and optionally saves it to disk.
    '''
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # put the grid behind the plot
    ax.set_axisbelow(True)

    # use bins when available
    bins = bins_no if bins_no is not None else 'auto'
   
    # density
    if density is not None:
        density = True
    else: density = False

    counts, bin_edges, _ = ax.hist(
        data_column,
        #color='#708A81',
        color='#a5c4b6',
        edgecolor='#D2CCC3',
        #edgecolor='ghostwhite',
        linewidth=0.5,
        bins=bins,
        density = density
    )
    
    if density_fn is not None:
        x = np.linspace(bin_edges[0], bin_edges[-1], 500)

        hist_max = counts.max()
        y_cap = hist_max * 1.1
        
        for f, color, label, linestyle in density_fn:
            y = f(x)
            # cap the function which blow up
            y = np.where(y <= y_cap, y, np.nan)

            ax.plot(
                x,
                y,
                linewidth=0.75,
                color=color,
                linestyle=linestyle,
                label = label
                #linestyle = '--',
                #alpha=0.75
            )
        
        ax.legend()

    if vertical_lines is not None:
        for x, color in vertical_lines:
            ax.axvline(
                x = x,
                linestyle = '--',
                #linestyle=(0, (5, 10)),
                linewidth = 1,
                # color = '#1237c3',    # blue
                color = color,    # dark teal
                alpha = 0.75
            )

    # labels
    ax.set_title(
        title,
        fontsize=14,
        pad=12
    )
    
    ax.set_xlabel(
        x_axis_title,
        fontsize=11
    )
    
    ax.set_ylabel(
        y_axis_title,
        fontsize=11
    )

    # optional annotations
    if annotation:
        if isinstance(annotation, dict):
            ax.annotate(**annotation)
        else:
            for a in annotation:
                ax.annotate(**a)

    # styling

    # background color
    fig.patch.set_facecolor('whitesmoke')
    ax.set_facecolor('whitesmoke')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    ax.grid(
        True,
        color = '#D2CCC3',
        linewidth = 0.8,
        alpha = 0.7,
    )

    # save the plot
    if save_file_name:
        plt.savefig(
            save_file_name,
            dpi = 300,
            bbox_inches = "tight"
        )

    plt.show()
    # prevent figures from accumulating in memory
    plt.close(fig)
