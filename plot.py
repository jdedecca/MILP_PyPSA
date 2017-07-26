
## Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David
## Schlachtberger (FIAS)
## Copyright 2017 João Gorenstein Dedecca

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


# make the code as Python 3 compatible as possible
from __future__ import division
from __future__ import absolute_import
import six
from six import iteritems

import pandas as pd
import numpy as np

__author__ = """
Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS)
Joao Gorenstein Dedecca
"""

__copyright__ = """
Copyright 2015-2017 Tom Brown (FIAS), Jonas Hoersch (FIAS), David Schlachtberger (FIAS), GNU GPL 3
Copyright 2017 João Gorenstein Dedecca, GNU GPL 3
"""

plt_present = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except:
    plt_present = False

basemap_present = True
try:
    from mpl_toolkits.basemap import Basemap
except:
    basemap_present = False



def plot(network, margin=0.05, ax=None, basemap=True, bus_colors='b',
         line_colors='g', bus_sizes=10, line_widths=2, title="",
         line_cmap=None, bus_cmap=None, boundaries=None,
         geometry=False, branch_components=['Line', 'Link']):
    """
    Plot the network buses and lines using matplotlib and Basemap.

    Parameters
    ----------
    margin : float
        Margin at the sides as proportion of distance between max/min x,y
    ax : matplotlib ax, defaults to plt.gca()
        Axis to which to plot the network
    basemap : bool, default True
        Switch to use Basemap
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "b"
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 10
    line_colors : dict/pandas.Series
        Colors for the lines, defaults to "g" for Lines and "cyan" for
        Links. Colors for branches other than Lines can be
        specified using a pandas Series with a MultiIndex.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2. Widths for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    title : string
        Graph title
    line_cmap : plt.cm.ColorMap/str|dict
        If line_colors are floats, this color map will assign the colors.
        Use a dict to specify colormaps for more than one branch type.
    bus_cmap : plt.cm.ColorMap/str
        If bus_colors are floats, this color map will assign the colors
    boundaries : list of four floats
        Boundaries of the plot in format [x1,x2,y1,y2]
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.

    Returns
    -------
    bus_collection, branch_collection1, ... : tuple of Collections
        Collections for buses and branches.
    """

    defaults_for_branches = {
        'Link': dict(color="cyan", width=2),
        'Line': dict(color="b", width=2)
    }

    if not plt_present:
        logger.error("Matplotlib is not present, so plotting won't work.")
        return

    if ax is None:
        ax = plt.gca()

    def compute_bbox_with_margins(margin, x, y):
        #set margins
        pos = np.asarray((x, y))
        minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
        xy1 = minxy - margin*(maxxy - minxy)
        xy2 = maxxy + margin*(maxxy - minxy)
        return tuple(xy1), tuple(xy2)

    x = network.buses["x"]
    y = network.buses["y"]

    if basemap and basemap_present:
        if boundaries is None:
            (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
        else:
            x1, x2, y1, y2 = boundaries
        bmap = Basemap(resolution='l', epsg=network.srid,
                       llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                       urcrnrlon=x2, ax=ax)
        bmap.drawcountries()
        bmap.drawcoastlines()

        x, y = bmap(x.values, y.values)
        x = pd.Series(x, network.buses.index)
        y = pd.Series(y, network.buses.index)

    c = pd.Series(bus_colors, index=network.buses.index)
    if c.dtype == np.dtype('O'):
        c.fillna("b", inplace=True)
        c = list(c.values)
    s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)
    bus_collection = ax.scatter(x, y, c=c, s=s, cmap=bus_cmap)

    def as_branch_series(ser):
        if isinstance(ser, pd.Series):
            if isinstance(ser.index, pd.MultiIndex):
                return ser
            index = ser.index
            ser = ser.values
        else:
            index = network.lines.index
        return pd.Series(ser,
                         index=pd.MultiIndex(levels=(["Line"], index),
                                             labels=(np.zeros(len(index)),
                                                     np.arange(len(index)))))

    line_colors = as_branch_series(line_colors)
    line_widths = as_branch_series(line_widths)
    if not isinstance(line_cmap, dict):
        line_cmap = {'Line': line_cmap}

    branch_collections = []
    for c in network.iterate_components(branch_components):
        l_defaults = defaults_for_branches[c.name]
        l_widths = line_widths.get(c.name, l_defaults['width'])
        l_nums = None
        if c.name in line_colors:
            l_colors = line_colors[c.name]

            if issubclass(l_colors.dtype.type, np.number):
                l_nums = l_colors
                l_colors = None
            else:
                l_colors.fillna(l_defaults['color'], inplace=True)
        else:
            l_colors = l_defaults['color']

        if not geometry:
            segments = (np.asarray(((c.df.bus0.map(x),
                                     c.df.bus0.map(y)),
                                    (c.df.bus1.map(x),
                                     c.df.bus1.map(y))))
                        .transpose(2, 0, 1))
        else:
            from shapely.wkt import loads
            from shapely.geometry import LineString
            linestrings = c.df.geometry.map(loads)
            assert all(isinstance(ls, LineString) for ls in linestrings), \
                "The WKT-encoded geometry in the 'geometry' column must be composed of LineStrings"
            segments = np.asarray(list(linestrings.map(np.asarray)))
            if basemap and basemap_present:
                segments = np.transpose(bmap(*np.transpose(segments, (2, 0, 1))), (1, 2, 0))

        l_collection = LineCollection(segments,
                                      linewidths=l_widths,
                                      antialiaseds=(1,),
                                      colors=l_colors,
                                      transOffset=ax.transData)

        if l_nums is not None:
            l_collection.set_array(np.asarray(l_nums))
            l_collection.set_cmap(line_cmap.get(c.name, None))
            l_collection.autoscale()

        ax.add_collection(l_collection)
        l_collection.set_zorder(1)

        branch_collections.append(l_collection)

    bus_collection.set_zorder(2)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()

    ax.set_title(title)

    return (bus_collection,) + tuple(branch_collections)


def plot_uni(network, margin=0.05, ax=None, basemap=True, bus_colors='b',
         line_colors='r', line_styles = 'solid',bus_sizes=35, line_widths=2, title="",
         line_cmap=None, bus_cmap=None, boundaries=None,
         geometry=False, branch_components=['Line', 'Link'],bc_min = None, bc_max = None, bus_line_widths = 1, bus_markers = 'o',edgecolors='k'):
    """
    Plot the network buses and lines using matplotlib and Basemap.

    Parameters
    ----------
    margin : float
        Margin at the sides as proportion of distance between max/min x,y
    ax : matplotlib ax, defaults to plt.gca()
        Axis to which to plot the network
    basemap : bool, default True
        Switch to use Basemap
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "b"
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 10
    line_colors : dict/pandas.Series
        Colors for the lines, defaults to "g" for Lines and "cyan" for
        Links. Colors for branches other than Lines can be
        specified using a pandas Series with a MultiIndex.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 2. Widths for branches other
        than Lines can be specified using a pandas Series with a
        MultiIndex.
    title : string
        Graph title
    line_cmap : plt.cm.ColorMap/str|dict
        If line_colors are floats, this color map will assign the colors.
        Use a dict to specify colormaps for more than one branch type.
    bus_cmap : plt.cm.ColorMap/str
        If bus_colors are floats, this color map will assign the colors
    boundaries : list of four floats
        Boundaries of the plot in format [x1,x2,y1,y2]
    branch_types : list of str or pypsa.component
        Branch types to be plotted, defaults to Line and Link.

    Returns
    -------
    bus_collection, branch_collection1, ... : tuple of Collections
        Collections for buses and branches.
    """

    defaults_for_branches = {
        'Link': dict(color="cyan", width=2,style='solid'),
        'Line': dict(color="r", width=2,style='solid')
    }

    if not plt_present:
        logger.error("Matplotlib is not present, so plotting won't work.")
        return

    if ax is None:
        ax = plt.gca()

    def compute_bbox_with_margins(margin, x, y):
        #set margins
        pos = np.asarray((x, y))
        minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
        xy1 = minxy - margin*(maxxy - minxy)
        xy2 = maxxy + margin*(maxxy - minxy)
        return tuple(xy1), tuple(xy2)

    x = network.buses["x"]
    y = network.buses["y"]

    if basemap and basemap_present:
        if boundaries is None:
            (x1, y1), (x2, y2) = compute_bbox_with_margins(margin, x, y)
        else:
            x1, x2, y1, y2 = boundaries
        bmap = Basemap(resolution='i', epsg=network.srid,
                       llcrnrlat=y1, urcrnrlat=y2, llcrnrlon=x1,
                       urcrnrlon=x2, ax=ax)
        bmap.drawcountries(zorder=1)
        bmap.drawcoastlines(zorder=1,linewidth=0.5)
        bmap.fillcontinents(color='0.9',zorder=0)

        x, y = bmap(x.values, y.values)
        x = pd.Series(x, network.buses.index)
        y = pd.Series(y, network.buses.index)

    c = pd.Series(bus_colors, index=network.buses.index)

    bus_mask = network.buses[network.buses.carrier != 'DC'].index

    if c.dtype == np.dtype('O'):
        c.fillna("b", inplace=True)
        #c = list(c.values)
    s = pd.Series(bus_sizes, index=network.buses.index, dtype="float").fillna(10)
    m = pd.Series(bus_markers, index=network.buses.index).fillna('o')
    for marker in m.unique():
        buses = m[m == marker].index.intersection(bus_mask)
        bus_collection = ax.scatter(x[buses], y[buses], c=c[buses], s=s[buses], marker = marker,linewidths=bus_line_widths,edgecolors=edgecolors, cmap=bus_cmap,vmin = bc_min,vmax = bc_max,zorder = 3)

    def as_branch_series(ser):
        if isinstance(ser, pd.Series):
            if isinstance(ser.index, pd.MultiIndex):
                return ser
            index = ser.index
            ser = ser.values
        else:
            index = network.lines.index
        return pd.Series(ser,
                         index=pd.MultiIndex(levels=(["Line"], index),
                                             labels=(np.zeros(len(index)),
                                                     np.arange(len(index)))))

    line_colors = as_branch_series(line_colors)
    line_widths = as_branch_series(line_widths)
    line_styles = as_branch_series(line_styles)
    if not isinstance(line_cmap, dict):
        line_cmap = {'Line': line_cmap, 'Link': line_cmap}

    branch_collections = []
    for c in network.iterate_components(branch_components):
        l_defaults = defaults_for_branches[c.name]
        l_widths = line_widths.get(c.name, l_defaults['width'])
        # try:
        #     l_widths
        # except NameError:
        #     l_widths = line_widths.get(t.name, l_defaults['width'])
        # else:
        #     l_widths = l_widths.append(line_widths.get(t.name, l_defaults['width']))
        l_nums = None

        if c.name in line_colors:
            l_colors = line_colors[c.name]

            if issubclass(l_colors.dtype.type, np.number):
                l_nums = l_colors
                l_colors = None
            else:
                l_colors.fillna(l_defaults['color'], inplace=True)
        else:
            l_colors = l_defaults['color']


        if c.name in line_styles:
            l_styles= line_styles[c.name]

            if issubclass(l_styles.dtype.type, np.number):
                l_nums = l_styles
                l_styles = None
            else:
                l_styles.fillna(l_defaults['style'], inplace=True)
        else:
            l_styles = l_defaults['style']

        if not geometry:
            segments = (np.asarray(((c.df.bus0.map(x),
                                     c.df.bus0.map(y)),
                                    (c.df.bus1.map(x),
                                     c.df.bus1.map(y))))
                        .transpose(2, 0, 1))
        else:
            from shapely.wkt import loads
            from shapely.geometry import LineString
            linestrings = c.df.geometry.map(loads)
            assert all(isinstance(ls, LineString) for ls in linestrings), \
                "The WKT-encoded geometry in the 'geometry' column must be composed of LineStrings"
            segments = np.asarray(list(linestrings.map(np.asarray)))
            if basemap and basemap_present:
                segments = np.transpose(bmap(*np.transpose(segments, (2, 0, 1))), (1, 2, 0))

        l_collection = LineCollection(segments,
                                      linewidths=l_widths,
                                      antialiaseds=(1,),
                                      linestyles=l_styles,
                                      colors=l_colors,
                                      alpha = .8,
                                      transOffset=ax.transData)

        if l_nums is not None:
            l_collection.set_array(np.asarray(l_nums))
            l_collection.set_cmap(line_cmap.get(c.name, None))
            l_collection.autoscale()

        ax.add_collection(l_collection)
        l_collection.set_zorder(2)

        branch_collections.append(l_collection)

    bus_collection.set_zorder(3)

    ax.update_datalim(compute_bbox_with_margins(margin, x, y))
    ax.autoscale_view()

    ax.set_title(title)

    return (bus_collection,) + tuple(branch_collections)
