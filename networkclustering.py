## Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS)

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

"""Functions for computing network clusters
"""

from __future__ import absolute_import, division

__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"
__copyright__ = "Copyright 2015-2016 Tom Brown (FIAS), Jonas Hoersch (FIAS), GNU GPL 3"

import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict, namedtuple
from itertools import repeat
from six.moves import zip, range
from six import itervalues, iteritems
import six

import logging
logger = logging.getLogger(__name__)


from .descriptors import OrderedGraph
from .components import Network

from . import components, io



def _consense(x):
    v = x.iat[0]
    assert ((x == v).all() or x.isnull().all())
    return v

def _haversine(coords):
    lon, lat = np.deg2rad(np.asarray(coords)).T
    a = np.sin((lat[1]-lat[0])/2.)**2 + np.cos(lat[0]) * np.cos(lat[1]) * np.sin((lon[0] - lon[1])/2.)**2
    return 6371.000 * 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )

def aggregategenerators(network, busmap, with_time=True):
    attrs = network.components["Generator"]["attrs"]
    columns = set(attrs.index[~attrs.varying]) | {'weight'}
    generators = network.generators.assign(bus=lambda df: df.bus.map(busmap))
    grouper = [generators.bus, generators.carrier]

    weighting = generators.weight.groupby(grouper, axis=0).transform(lambda x: (x/x.sum()).fillna(1.))
    generators['p_nom_max'] /= weighting
    strategies = {'p_nom_max': np.min, 'weight': np.sum, 'p_nom': np.sum}
    strategies.update(zip(columns.difference(strategies), repeat(_consense)))
    new_df = generators.groupby(grouper, axis=0).agg(strategies)

    new_index = pd.Index([bus + ' ' + carrier
                          for bus, carrier in new_df.index.get_values()])

    new_pnl = dict()
    if with_time:
        for attr, df in iteritems(network.generators_t):
            if not df.empty:
                time_dependent_old_b, time_dependent_new_b = \
                    ((generators[attr + '_t'], new_df[attr + '_t'])
                     if attr + '_t' in generators
                     else (slice(None), slice(None)))

                if attr == 'p_max_pu':
                    df = df.multiply(weighting.loc[time_dependent_old_b], axis=1)
                pnl_df = df.groupby(grouper, axis=1).sum().reindex(columns=new_df.index[time_dependent_new_b])
                pnl_df.columns = new_index[time_dependent_new_b]
                new_pnl[attr] = pnl_df

    new_df.set_index(new_index, inplace=True)
    return new_df, new_pnl

def aggregateoneport(network, busmap, component, with_time=True):

    attrs = network.components[component]["attrs"]
    columns = set(attrs.index[~attrs.varying])

    old_df = getattr(network, network.components[component]["list_name"]).assign(bus=lambda df: df.bus.map(busmap))
    if 'carrier' in columns:
        grouper = [old_df.bus, old_df.carrier]
    else:
        grouper = old_df.bus
    strategies = {attr: (np.sum
                         if attr in {'p', 'q', 'p_set', 'q_set',
                                     'p_nom', 'p_nom_max', 'p_nom_min'}
                         else _consense)
                  for attr in columns}
    new_df = old_df.groupby(grouper).agg(strategies)
    if 'carrier' in columns:
        new_index = pd.Index([bus + ' ' + carrier
                              for bus, carrier in new_df.index.get_values()])
    else:
        new_index = new_df.index

    new_pnl = dict()
    if with_time:
        old_pnl = network.pnl(component)
        for attr, df in iteritems(old_pnl):
            if not df.empty:
                pnl_df = df.groupby(grouper, axis=1).sum()
                time_dependent_b  = (new_df[attr + '_t']
                                     if attr + '_t' in new_df
                                     else slice(None))
                pnl_df = pnl_df.reindex(columns=new_df.index[time_dependent_b])
                pnl_df.columns = new_index[time_dependent_b]
                new_pnl[attr] = pnl_df

    new_df.set_index(new_index, inplace=True)
    return new_df, new_pnl

def aggregatebuses(network, busmap, custom_strategies=dict()):
    attrs = network.components["Bus"]["attrs"]
    columns = set(attrs.index[~attrs.varying])

    strategies = dict(x=np.mean, y=np.mean,
                      v_nom=np.max,
                      v_mag_pu_max=np.min, v_mag_pu_min=np.max)
    strategies.update(zip(columns.difference(strategies), repeat(_consense)))
    strategies.update(custom_strategies)

    return network.buses \
            .groupby(busmap).agg(strategies) \
            .reindex(columns=[f
                              for f in network.buses.columns
                              if f in columns or f in custom_strategies])

def aggregatelines(network, buses, interlines, line_length_factor=1.0):

    #make sure all lines have same bus ordering
    positive_order = interlines.bus0_s < interlines.bus1_s
    interlines_p = interlines[positive_order]
    interlines_n = interlines[~ positive_order].rename(columns={"bus0_s":"bus1_s", "bus1_s":"bus0_s"})
    interlines_c = pd.concat((interlines_p,interlines_n))

    attrs = network.components["Line"]["attrs"]
    columns = set(attrs.index[~attrs.varying]).difference(('bus0', 'bus1'))

    def aggregatelinegroup(l):

        # l.name is a tuple of the groupby index (bus0_s, bus1_s)
        length_s = _haversine(buses.loc[list(l.name),['x', 'y']])*line_length_factor
        v_nom_s = _consense(buses.loc[list(l.name),'v_nom'])

        voltage_factor = (np.asarray(network.buses.loc[l.bus0,'v_nom'])/v_nom_s)**2
        length_factor = length_s/l['length']

        data = dict(
            r=1./(voltage_factor/(length_factor * l['r'])).sum(),
            x=1./(voltage_factor/(length_factor * l['x'])).sum(),
            g=(voltage_factor * length_factor * l['g']).sum(),
            b=(voltage_factor * length_factor * l['b']).sum(),
            terrain_factor=l['terrain_factor'].mean(),
            s_nom=l['s_nom'].sum(),
            s_nom_min=l['s_nom_min'].sum(),
            s_nom_max=l['s_nom_max'].sum(),
            s_nom_extendable=l['s_nom_extendable'].any(),
            capital_cost=l['capital_cost'].sum(),
            length=length_s,
            sub_network=_consense(l['sub_network']),
            v_ang_min=l['v_ang_min'].max(),
            v_ang_max=l['v_ang_max'].min()
        )
        data.update((f, _consense(l[f])) for f in columns.difference(data))
        return pd.Series(data, index=[f for f in l.columns if f in columns])

    lines = interlines_c.groupby(['bus0_s', 'bus1_s']).apply(aggregatelinegroup)
    lines['name'] = [str(i+1) for i in range(len(lines))]

    linemap_p = interlines_p.join(lines['name'], on=['bus0_s', 'bus1_s'])['name']
    linemap_n = interlines_n.join(lines['name'], on=['bus0_s', 'bus1_s'])['name']
    linemap = pd.concat((linemap_p,linemap_n))

    return lines, linemap_p, linemap_n, linemap

def get_buses_linemap_and_lines(network, busmap, line_length_factor=1.0, bus_strategies=dict()):
    # compute new buses
    buses = aggregatebuses(network, busmap, bus_strategies)

    lines = network.lines

    lines['bus0_s'] = lines.bus0.map(busmap)
    lines['bus1_s'] = lines.bus1.map(busmap)

    # lines between different clusters
    interlines = lines.loc[lines['bus0_s'] != lines['bus1_s']]
    lines, linemap_p, linemap_n, linemap = aggregatelines(network, buses, interlines, line_length_factor)
    return (buses,
            linemap,
            linemap_p,
            linemap_n,
            lines.reset_index()
                 .rename(columns={'bus0_s': 'bus0', 'bus1_s': 'bus1'}, copy=False)
                 .set_index('name'))

Clustering = namedtuple('Clustering', ['network', 'busmap', 'linemap',
                                       'linemap_positive', 'linemap_negative'])

def get_clustering_from_busmap(network, busmap, with_time=True, line_length_factor=1.0,
                               aggregate_generators_weighted=False, aggregate_one_ports={},
                               bus_strategies=dict()):

    buses, linemap, linemap_p, linemap_n, lines = get_buses_linemap_and_lines(network, busmap, line_length_factor, bus_strategies)

    network_c = Network()

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")

    if with_time:
        network_c.set_snapshots(network.snapshots)

    one_port_components = components.one_port_components

    if aggregate_generators_weighted:
        one_port_components.remove("Generator")
        generators, generators_pnl = aggregategenerators(network, busmap, with_time=with_time)
        io.import_components_from_dataframe(network_c, generators, "Generator")
        if with_time:
            for attr, df in iteritems(generators_pnl):
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, "Generator", attr)

    for one_port in aggregate_one_ports:
        one_port_components.remove(one_port)
        new_df, new_pnl = aggregateoneport(network, busmap, component=one_port, with_time=with_time)
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)


    for c in network.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            network_c,
            c.df.assign(bus=c.df.bus.map(busmap)),
            c.name
        )

    if with_time:
        for c in network.iterate_components(one_port_components):
            for attr, df in iteritems(c.pnl):
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, c.name, attr)

    io.import_components_from_dataframe(
        network_c,
        network.links.assign(bus0=network.links.bus0.map(busmap),
                             bus1=network.links.bus1.map(busmap)),
        "Link"
    )

    io.import_components_from_dataframe(
        network_c,
        network.carriers,
        "Carrier"
    )

    if with_time:
        for attr, df in iteritems(network.links_t):
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Link", attr)

    network_c.determine_network_topology()

    return Clustering(network_c, busmap, linemap, linemap_p, linemap_n)


################
# Length

def busmap_by_linemask(network, mask):
    mask = network.lines.loc[:,['bus0', 'bus1']].assign(mask=mask).set_index(['bus0','bus1'])['mask']
    G = nx.OrderedGraph()
    G.add_nodes_from(network.buses.index)
    G.add_edges_from(mask.index[mask])
    return pd.Series(OrderedDict((n, str(i))
                                 for i, g in enumerate(nx.connected_components(G))
                                 for n in g),
                     name='name')

def busmap_by_length(network, length):
    return busmap_by_linemask(network, network.lines.length < length)

def length_clustering(network, length):
    busmap = busmap_by_length(network, length=length)
    return get_clustering_from_busmap(network, busmap)

################
# SpectralClustering

try:
    # available using pip as scikit-learn
    from sklearn.cluster import spectral_clustering as sk_spectral_clustering

    def busmap_by_spectral_clustering(network, n_clusters, **kwds):
        lines = network.lines.loc[:,['bus0', 'bus1']].assign(weight=1./network.lines.x).set_index(['bus0','bus1'])
        G = OrderedGraph()
        G.add_nodes_from(network.buses.index)
        G.add_edges_from((u,v,dict(weight=w)) for (u,v),w in lines.itertuples())
        return pd.Series(sk_spectral_clustering(nx.adjacency_matrix(G), n_clusters, **kwds) + 1,
                         index=network.buses.index)

    def spectral_clustering(network, n_clusters=8, **kwds):
        busmap = busmap_by_spectral_clustering(network, n_clusters=n_clusters, **kwds)
        return get_clustering_from_busmap(network, busmap)

except ImportError:
    pass

################
# Louvain

try:
    # available using pip as python-louvain
    import community

    def busmap_by_louvain(network, level=-1):
        lines = network.lines.loc[:,['bus0', 'bus1']].assign(weight=1./network.lines.x).set_index(['bus0','bus1'])
        G = nx.Graph()
        G.add_nodes_from(network.buses.index)
        G.add_edges_from((u,v,dict(weight=w)) for (u,v),w in lines.itertuples())
        dendrogram = community.generate_dendrogram(G)
        if level < 0:
            level += len(dendrogram)
        return pd.Series(community.partition_at_level(dendrogram, level=level),
                         index=network.buses.index)

    def louvain_clustering(network, level=-1, **kwds):
        busmap = busmap_by_louvain(network, level=level)
        return get_clustering_from_busmap(network, busmap)

except ImportError:
    pass


################
# k-Means clustering based on bus properties

try:
    # available using pip as scikit-learn
    from sklearn.cluster import KMeans

    def busmap_by_kmeans(network, bus_weightings, n_clusters, buses_i=None, ** kwargs):
        """
        Create a bus map from the clustering of buses in space with a
        weighting.

        Parameters
        ----------
        network : pypsa.Network
            The buses must have coordinates x,y.
        bus_weightings : pandas.Series
            Series of integer weights for buses, indexed by bus names.
        n_clusters : int
            Final number of clusters desired.
        kwargs
            Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs)

        Returns
        -------
        busmap : pandas.Series
            Mapping of network.buses to k-means clusters (indexed by
            non-negative integers).
        """

        if buses_i is None:
            buses_i = network.buses.index

        # since one cannot weight points directly in the scikit-learn
        # implementation of k-means, just add additional points at
        # same position
        points = (network.buses.loc[buses_i, ["x","y"]].values
                  .repeat(bus_weightings.reindex(buses_i).astype(int), axis=0))

        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, ** kwargs)

        kmeans.fit(points)

        busmap = pd.Series(data=kmeans.predict(network.buses.loc[buses_i, ["x","y"]]),
                           index=buses_i).astype(str)

        return busmap

    def kmeans_clustering(network, bus_weightings, n_clusters, line_length_factor=1.0, ** kwargs):
        """
        Cluster then network according to k-means clustering of the
        buses.

        Buses can be weighted by an integer in the series
        `bus_weightings`.

        Note that this clustering method completely ignores the
        branches of the network.

        Parameters
        ----------
        network : pypsa.Network
            The buses must have coordinates x,y.
        bus_weightings : pandas.Series
            Series of integer weights for buses, indexed by bus names.
        n_clusters : int
            Final number of clusters desired.
        line_length_factor : float
            Factor to multiply the crow-flies distance between new buses in order to get new
            line lengths.
        kwargs
            Any remaining arguments to be passed to KMeans (e.g. n_init, n_jobs)


        Returns
        -------
        Clustering : named tuple
            A named tuple containing network, busmap and linemap
        """

        busmap = busmap_by_kmeans(network, bus_weightings, n_clusters, ** kwargs)
        return get_clustering_from_busmap(network, busmap, line_length_factor=line_length_factor)

except ImportError:
    pass






################
# Rectangular grid clustering


def busmap_by_rectangular_grid(buses, divisions=10):
    busmap = pd.Series(0, index=buses.index)
    if isinstance(divisions, tuple):
        divisions_x, divisions_y = divisions
    else:
        divisions_x = divisions_y = divisions
    gb = buses.groupby([pd.cut(buses.x, divisions_x), pd.cut(buses.y, divisions_y)])
    for nk, oks in enumerate(itervalues(gb.groups)):
        busmap.loc[oks] = nk
    return busmap

def rectangular_grid_clustering(network, divisions):
    busmap = busmap_by_rectangular_grid(network.buses, divisions)
    return get_clustering_from_busmap(network, busmap)





################
# Reduce stubs/dead-ends, i.e. nodes with valency 1, iteratively to remove tree-like structures

def busmap_by_stubs(network):
    """Create a busmap by reducing stubs and stubby trees
    (i.e. sequentially reducing dead-ends).

    Parameters
    ----------
    network : pypsa.Network

    Returns
    -------
    busmap : pandas.Series
        Mapping of network.buses to k-means clusters (indexed by
        non-negative integers).

    """

    busmap = pd.Series(network.buses.index,network.buses.index)

    network = network.copy(with_time=False)

    count = 0

    while True:
        old_count = count
        logger.info("{} buses".format(len(network.buses)))
        graph = network.graph()
        for u in graph.node:
            neighbours = list(graph.adj[u].keys())
            if len(neighbours) == 1:
                neighbour = neighbours[0]
                count +=1
                lines = list(graph.adj[u][neighbour].keys())
                for line in lines:
                    network.remove("Line",line.name)
                network.remove("Bus",u)
                busmap[busmap==u] = neighbour
        logger.info("{} deleted".format(count))
        if old_count == count:
            break
    return busmap

def stubs_clustering(network,use_reduced_coordinates=True, line_length_factor=1.0):
    """Cluster network by reducing stubs and stubby trees
    (i.e. sequentially reducing dead-ends).

    Parameters
    ----------
    network : pypsa.Network
    use_reduced_coordinates : boolean
        If True, do not average clusters, but take from busmap.
    line_length_factor : float
        Factor to multiply the crow-flies distance between new buses in order to get new
        line lengths.

    Returns
    -------
    Clustering : named tuple
        A named tuple containing network, busmap and linemap
    """

    busmap = busmap_by_stubs(network)

    #reset coordinates to the new reduced guys, rather than taking an average
    if use_reduced_coordinates:
        # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
        # i.e. network is changed in place!!
        network.buses.loc[busmap.index,['x','y']] = network.buses.loc[busmap,['x','y']].values

    return get_clustering_from_busmap(network, busmap, line_length_factor=line_length_factor)
