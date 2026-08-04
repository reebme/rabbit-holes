import numpy as np
import numpy.typing as npt
import pandas as pd

import matplotlib.pyplot as plt

HIST_COLOR = '#a5c4b6'
CURVE_COLOR ='#246A73'
BCKGND_COLOR = 'whitesmoke'
GRID_COLOR = '#D2CCC3'

def pretty_plot(x_ax: npt.ArrayLike,
                y_ax: npt.ArrayLike,
                title: str | None = None,
                x_axis_title: str | None = None,
                y_axis_title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_ax, y_ax, linewidth=1.5, c=CURVE_COLOR)

    ax.set_xticks(x_ax)
    ax.set_yticks(np.linspace(min(y_ax), max(y_ax), num = 10, endpoint=True))

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

    # background color
    fig.patch.set_facecolor(BCKGND_COLOR)
    ax.set_facecolor(BCKGND_COLOR)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    ax.grid(
        True,
        color = GRID_COLOR,
        linewidth = 0.5,
        alpha = 0.5,
    )
    
    ax.grid(True)
    plt.show()

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

def compute_cube_limit(points, pad=0.1):
    """
    Half-width of a symmetric plotting cube containing the points.
    """
    return np.round((1 + pad) * np.max(np.abs(points)))

def plot_plane_in_cube(
    ax,
    normal=(1, 1, 1),
    offset=0,
    lim=4,
    alpha=0.05,
    color=None,
    linewidth=0,
):
    """
    Plot the part of the plane

        normal[0] * x + normal[1] * y + normal[2] * z = offset

    inside the cube [-lim, lim]^3.
    """

    normal = np.asarray(normal, dtype=float)

    if np.allclose(normal, 0):
        raise ValueError("normal vector cannot be zero")

    vertices = []

    # Intersections with cube edges:
    # Fix two coordinates at cube boundaries, solve for the third.
    for i, j in combinations(range(3), 2):
        k = ({0, 1, 2} - {i, j}).pop()

        if np.isclose(normal[k], 0):
            continue

        for a, b in product([-lim, lim], repeat=2):
            p = np.zeros(3)
            p[i] = a
            p[j] = b
            p[k] = (offset - normal[i] * a - normal[j] * b) / normal[k]

            if -lim <= p[k] <= lim:
                vertices.append(p)

    if len(vertices) < 3:
        return None

    vertices = np.unique(np.round(vertices, 12), axis=0)

    if len(vertices) < 3:
        return None

    # Order vertices around their center
    center = vertices.mean(axis=0)

    # Build an orthonormal basis inside the plane
    # Pick a vector not parallel to the normal
    candidate = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(normal, candidate), 0):
        candidate = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, candidate)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    coords = np.column_stack([
        (vertices - center) @ u,
        (vertices - center) @ v
    ])

    angles = np.arctan2(coords[:, 1], coords[:, 0])
    vertices = vertices[np.argsort(angles)]

    return ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        alpha=alpha,
        color=color,
        linewidth=linewidth,
    )

def add_3d_shadows(
    ax,
    x,
    y,
    z,
    lim,
    s=5,
    alpha=0.35,
    color="lightgrey",
):
    """
    Add projected shadows of a 3D point cloud onto the three cube walls.
    Assumes axes limits are [-lim, lim] in all directions.
    """

    xmin, xmax = -lim, lim
    ymin, ymax = -lim, lim
    zmin, zmax = -lim, lim

    # xy projection, onto bottom z-wall
    ax.scatter(x, y, zs=zmin, zdir="z", s=s, alpha=alpha, color=color)

    # xz projection, onto back y-wall
    ax.scatter(x, z, zs=ymax, zdir="y", s=s, alpha=alpha, color=color)

    # yz projection, onto left x-wall
    ax.scatter(y, z, zs=xmin, zdir="x", s=s, alpha=alpha, color=color)

def pretty_3d_axes(ax, lim, n_ticks=5):
    """
    Make a matplotlib 3D axis pretty
    """

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ticks = np.linspace(-lim, lim, n_ticks)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.xaxis.label.set_color("0.4")
    ax.yaxis.label.set_color("0.4")
    ax.zaxis.label.set_color("0.4")

    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.25)
        axis.line.set_color((0.8, 0.8, 0.8, 0.3))

    ax.tick_params(colors="0.7")
    ax.set_box_aspect([1, 1, 1])

def pretty_3Dscatter_points(points,
                            pad = 0.1,
                            scatter_color='#6376bb',
                            hyperplane=None,
                            add_shadows = False,
                            azim=-30,
                            save_file=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    
    lim = compute_cube_limit(points, pad)
    
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    
    # the main 3D cloud
    ax.scatter(x, y, z, s=2, alpha=0.4, color = scatter_color)

    if hyperplane is not None:
        # the hyperplane they are on
        plot_plane_in_cube(
            ax,
            normal=hyperplane,
            offset=0,
            lim=lim,
            alpha=0.05
        )

    if add_shadows:
        add_3d_shadows(ax, x, y, z, lim, s = 2)
    
    ax.set_xlabel("coordinate 1", fontsize=14)
    ax.set_ylabel("coordinate 2", fontsize=14)
    ax.set_zlabel("coordinate 3", fontsize=14)
    
    pretty_3d_axes(ax, lim)
    
    ax.view_init(elev=10, azim=azim)

    if save_file:
        plt.savefig(save_file,
                dpi=300,
                bbox_inches="tight")
        
    plt.show()
