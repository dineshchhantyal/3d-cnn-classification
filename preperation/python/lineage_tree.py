"""
OO structure for cell lineage tracking.
"""

import os
import numpy as np
from collections import defaultdict
import math


class Node:
    "A cell in a timestamp."

    def __init__(self, node_id, timestamp_ordinal, label=None):
        self.node_id = node_id
        self.timestamp_ordinal = timestamp_ordinal
        self.label = label
        self.state = None
        self.parent = None
        self.id_to_child = {}
        self.reset()

    def clean_clone(self):
        return Node(self.node_id, self.timestamp_ordinal, self.label)

    def reset(self):
        self._track = None
        self._lineage_root = None
        self._lineage_index = None
        self._width = None
        self._offset = None
        self._is_isolated = None  # not determined
        self.color = None
        self.color_array = None

    def json_object(self):
        parent_id = None
        if self.parent is not None:
            parent_id = self.parent.node_id
        isolated = (self.parent is None) and (len(self.id_to_child) < 1)
        return dict(
            identity=self.node_id,
            timestamp_ordinal=self.timestamp_ordinal,
            label=self.label,
            color=self.color,
            offset=self._offset,
            parent_id=parent_id,
            isolated=isolated,
        )

    def same(self, other):
        return (
            (self.label == other.label)
            and (self.timestamp_ordinal == other.timestamp_ordinal)
            and (self.node_id == other.node_id)
        )

    def track_ancestor(self):
        if self._track is not None:
            return self._track
        parent = self.parent
        if parent is None:
            result = self
        elif len(parent.id_to_child) > 1:
            result = self
        else:
            result = self.parent.track_ancestor()
        self._track = result
        return result

    def lineage_ancestor(self):
        if self._lineage_root is not None:
            return self._lineage_root
        if self.parent is not None:
            result = self.parent.track_ancestor().lineage_ancestor()
        else:
            result = self
        self._lineage_root = result
        return result

    def __repr__(self):
        pid = None
        if self.parent is not None:
            pid = self.parent.node_id
        t = None
        if self._track:
            t = self._track.node_id
        l = None
        if self._lineage_root:
            l = self._lineage_root.node_id
        return "Node" + repr(
            (self.node_id, self.timestamp_ordinal, self.label, pid, t, l)
        )

    def set_parent(self, parent):
        assert type(parent) is Node, "bad parent type: " + repr([parent, type(parent)])
        # assert (
        #     self.timestamp_ordinal > parent.timestamp_ordinal
        # ), "Bad Node parent: " + repr([self, parent])
        warning = (
            self.timestamp_ordinal <= parent.timestamp_ordinal
        )  # this is a warning, not an error
        if warning:
            print(
                "WARNING: Bad Node parent: "
                + repr([self, parent])
                + " (timestamp_ordinal %d <= %d)"
                % (self.timestamp_ordinal, parent.timestamp_ordinal)
            )
        self.parent = parent

    def set_child(self, child):
        assert type(child) is Node, "bad child type: " + repr([child, type(child)])
        i2c = self.id_to_child
        # should have no more than 2 children? xxxx
        child.set_parent(self)
        i2c[child.node_id] = child

    def children_ids(self):
        return sorted(self.id_to_child.keys())

    """def width(self):
        if self._width is not None:
            return self._width
        result = 1
        for c in self.id_to_child.values():
            result += c.width()
        self._width = result
        return result"""

    """def offset(self, starting_from):
        if self._offset is not None:
            return self._offset
        result = starting_from + 1
        child_ids = self.children_ids()
        if child_ids:
            first = self.id_to_child[child_ids[0]]
            result = starting_from + first.width + 1
        self._offset = result
        return result"""

    def assign_offsets(self, starting_from=0, cursor_ref=None):
        assert starting_from is not None
        if cursor_ref is None:
            cursor_ref = [starting_from]
        child_ids = self.children_ids()
        i2c = self.id_to_child
        nc = len(child_ids)
        if nc < 1:
            # result = starting_from # + 1
            self._offset = cursor_ref[0]
            # no increment
        else:
            first_id = child_ids[0]
            first = i2c[first_id]
            remainder = child_ids[1:]
            if nc == 1:
                first.assign_offsets(cursor_ref=cursor_ref)
                self._offset = first._offset
            else:
                assert nc > 1
                first.assign_offsets(cursor_ref=cursor_ref)
                cursor_ref[0] += 1
                # self._offset = cursor_ref[0]
                # cursor_ref[0] += 1
                for r_id in remainder:
                    r = i2c[r_id]
                    result = r.assign_offsets(cursor_ref=cursor_ref)
                self._offset = 0.5 * (first._offset + r._offset)
        return cursor_ref[0]


class NodeGroup:
    "Superclass for node collection."

    id_to_node = None

    def add_node(self, node):
        self.check_node(node)
        if self.id_to_node is None:
            self.id_to_node = {}
        self.id_to_node[node.node_id] = node

    def reset(self):
        pass  # by default do nothing

    def isolated(self):
        return len(self.id_to_node) < 2

    def mark_isolated(self):
        test = self.isolated()
        for node in self.id_to_node.values():
            node._is_isolated = test


class TimeStamp(NodeGroup):
    "A time reference point."

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.label_to_node = {}

    def __repr__(self):
        return "Timestamp(%s)" % self.ordinal

    def check_node(self, node):
        ord = self.ordinal
        assert ord == node.timestamp_ordinal, "node not in timestamp: " + repr(
            [ord, node]
        )
        if node.label is not None:
            self.label_to_node[node.label] = node

    def farthest_parent_ordinal(self):
        result = self.ordinal
        for node in self.id_to_node.values():
            if node.parent is not None:
                ordinal = node.parent.timestamp_ordinal
                if (result is None) or (result > ordinal):
                    result = ordinal
        return result

    def color_mapping_array(self, maxlabel=None):
        l2n = self.label_to_node
        if l2n:
            max_node_label = max(l2n.keys())
            if maxlabel is None:
                maxlabel = max_node_label
            else:
                maxlabel = max(maxlabel, max_node_label)
        elif maxlabel is None:
            maxlabel = 0
        result = np.zeros((maxlabel + 1, 3), dtype=np.ubyte)
        # map any unassigned labels to grey
        result[1:] = 128
        for label, node in l2n.items():
            result[label] = node.color_array
        return result

    def json_object(self):
        return dict(ordinal=self.ordinal, node_id=sorted(self.id_to_node.keys()))


class Lineage(NodeGroup):
    "A group of tracks originating at a single ancestor cell."

    def __init__(self, root):
        self.root = root
        self.index = None

    def check_node(self, node):
        node_root = node.lineage_ancestor()
        assert node_root is self.root, "wrong lineage: " + repr(
            [node, node_root, self.root]
        )

    def set_index(self, index):
        self.index = index
        for node in self.id_to_node.values():
            node._lineage_index = index


class Track(Lineage):
    "A collection of nodes representing the same cell over many time steps."

    def check_node(self, node):
        node_root = node.track_ancestor()
        assert node_root is self.root, "wrong track: " + repr(
            [node, node_root, self.root]
        )


class Forest:
    "A collection of lineages"

    def __init__(self):
        self.id_to_node = {}
        self.ordinal_to_timestamp = {}
        self.label_volume_loader = None
        self.image_volume_loader = None
        self.reset()

    def clean_clone(self):
        result = Forest()
        node_map = {}
        for node in self.id_to_node.values():
            node_id = node.node_id
            new_node = result.add_node(node_id, node.timestamp_ordinal, node.label)
            node_map[node_id] = new_node
        for node in self.id_to_node.values():
            if node.parent is not None:
                pid = node.parent.node_id
                new_node = node_map[node.node_id]
                new_parent = node_map[pid]
                new_parent.set_child(new_node)
        result.label_volume_loader = self.label_volume_loader
        result.image_volume_loader = self.image_volume_loader
        return result

    def use_same_loaders(self, result):
        result.label_volume_loader = self.label_volume_loader
        result.image_volume_loader = self.image_volume_loader
        return result

    def reset(self):
        for node in self.id_to_node.values():
            node.reset()
        for ts in self.ordinal_to_timestamp.values():
            ts.reset()
        self.id_to_lineage = None
        self.id_to_track = {}
        self.track_order = None

    def load_image_for_timestamp(self, ts_ordinal):
        loader = self.image_volume_loader
        assert loader is not None, "No loader for images defined."
        return loader(ts_ordinal)

    def load_labels_for_timestamp(self, ts_ordinal):
        loader = self.label_volume_loader
        assert loader is not None, "No loader for labels defined."
        return loader(ts_ordinal)

    def create_nodes_for_labels_in_timestamp(self, ts_ordinal):
        labels = self.load_labels_for_timestamp(ts_ordinal)
        if labels is None:
            print("WARNING: NO LABELS FOUND FOR ORDINAL", ts_ordinal)
        else:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label > 0:
                    node_name = "%03d_%03d" % (ts_ordinal, label)
                    self.add_node(node_name, ts_ordinal, label)

    def load_klb_using_file_patterns(
        self,
        image_pattern="klbOut_Cam_Long_%(ordinal)05d.crop.klb",  # eg
        label_pattern="klbOut_Cam_Long_%(ordinal)05d.crop_cp_masks.klb",  # eg
    ):
        def img_loader(ordinal, pattern=image_pattern):
            import pyklb

            # subs = {"ordinal": ordinal}
            subs = {"ordinal": ordinal}  # files are zero based?
            path = pattern % subs
            if os.path.exists(path):
                return pyklb.readfull(path)
            else:
                return None  # no data for this timeslice.

        def label_loader(ordinal):
            return img_loader(ordinal, label_pattern)

        self.image_volume_loader = img_loader
        self.label_volume_loader = label_loader
        self.image_pattern = image_pattern
        self.label_pattern = label_pattern

    def use_trivial_null_loaders(self):
        def null_loader(ordinal):
            return None

        self.image_volume_loader = null_loader
        self.label_volume_loader = null_loader

    def check_labels(self, trivial=True):
        """
        Check for existence of label files for timestamps
        and whether the labels for the timestamps match the labels in the label volume.
        """
        o2t = self.ordinal_to_timestamp
        for ordinal in sorted(o2t.keys()):
            volume = self.load_labels_for_timestamp(ordinal)
            if volume is None:
                print("No volume file for", ordinal)
            elif not trivial:
                ts = o2t[ordinal]
                ts_labels = set(ts.label_to_node.keys())
                volume_labels = set(np.unique(volume)) - set([0])
                volume_missing = sorted(ts_labels - volume_labels)
                ts_missing = sorted(volume_labels - ts_labels)
                if volume_missing or ts_missing:
                    print("ts ordinal", ordinal)
                    print("     labels missing in lineage", ts_missing)
                    print("     labels missing in volume", volume_missing)
                else:
                    pass
                    # print("    labels match.")

    def assign_colors_to_lineages(self):
        return self.assign_colors_to_tracks(id_to_collection=self.id_to_lineage)

    def assign_colors_to_tracks(self, id_to_collection=None):
        if id_to_collection is None:
            id_to_collection = self.id_to_track
        for index, identifier in enumerate(sorted(id_to_collection.keys())):
            collection = id_to_collection[identifier]
            color_array = color_list.indexed_color(index + 1)
            color = color_list.rgbhtml(color_array)
            for node in collection.id_to_node.values():
                node.color = color
                node.color_array = color_array

    def find_tracks_and_lineages(self):
        self.reset()
        i2l = {}
        i2t = {}
        for node in self.id_to_node.values():
            tr = node.track_ancestor()
            ln = node.lineage_ancestor()
            # print(node, tr, ln)
            i2t[tr.node_id] = Track(tr)
            i2l[ln.node_id] = Lineage(ln)
        for node in self.id_to_node.values():
            tr = node.track_ancestor()
            ln = node.lineage_ancestor()
            i2t[tr.node_id].add_node(node)
            i2l[ln.node_id].add_node(node)
        self.id_to_lineage = i2l
        self.id_to_track = i2t
        # fix track ordering
        s_ids = sorted(i2t.keys())
        for i, tid in enumerate(s_ids):
            tr = i2t[tid]
            tr.set_index(i)
        self.track_order = [i2t[tid] for tid in s_ids]

    def assign_offsets(self, start_at=0):
        i2l = self.id_to_lineage
        assert i2l is not None, "lineages must be assigned first."
        cursor = start_at
        # assign isolated lineages after all non-isolated
        for rootid in i2l.keys():
            # mark isolated nodes for downstream testing
            lineage = i2l[rootid]
            lineage.mark_isolated()
        isolated_roots = []
        for rootid in sorted(i2l.keys()):
            lineage = i2l[rootid]
            if lineage.isolated():
                isolated_roots.append(rootid)
            else:
                cursor = lineage.root.assign_offsets(cursor) + 1
        # set isolated offsets higher...
        for rootid in isolated_roots:
            lineage = i2l[rootid]
            cursor = lineage.root.assign_offsets(cursor) + 1

    def add_node(self, node_id, ordinal, label=None):
        i2n = self.id_to_node
        assert node_id not in i2n, "duplicate node added: " + repr(
            [node_id, ordinal, label]
        )
        ts = self.get_or_add_timestamp(ordinal)
        n = Node(node_id, ordinal, label)
        i2n[node_id] = n
        ts.add_node(n)
        return n

    def get_or_add_timestamp(self, ordinal):
        i2t = self.ordinal_to_timestamp
        if ordinal in i2t:
            result = i2t[ordinal]
        else:
            result = i2t[ordinal] = TimeStamp(ordinal)
        return result

    def json_ob(self, exclude_isolated=True):
        (width, height) = self.dimensions()
        id_to_node = self.id_to_node
        if exclude_isolated:
            id_to_node = {
                ident: node
                for (ident, node) in self.id_to_node.items()
                if not node._is_isolated
            }
        id_to_node_json = {
            ident: node.json_object() for (ident, node) in id_to_node.items()
        }
        return dict(
            width=width,
            height=height,
            id_to_node=id_to_node_json,
        )

    def load_json(self, json_ob):
        id_to_node_json = json_ob["id_to_node"]
        for node_json in id_to_node_json.values():
            node_id = node_json["identity"]
            ordinal = node_json["timestamp_ordinal"]
            label = node_json["label"]
            self.add_node(node_id, ordinal, label)
        for node_json in id_to_node_json.values():
            parent_id = node_json["parent_id"]
            if parent_id:
                node_id = node_json["identity"]
                node = self.id_to_node[node_id]
                parent = self.id_to_node[parent_id]
                parent.set_child(node)

    def timestamp_region_json(self, ordinal):
        """
        Node info for nodes in timestamp, its predecessor and all directly connected ancestor timestamps.
        """
        # always include isolated nodes (?)
        o2t = self.ordinal_to_timestamp
        ordinals = set([ordinal])
        all_ordinals = sorted(o2t.keys())
        if ordinal not in all_ordinals:
            # hack boundary case
            return dict(
                id_to_node={},
                width=1,
                height=1,
                ordinals=[],
            )
        index = all_ordinals.index(ordinal)
        if index > 0:
            pred = all_ordinals[index - 1]
            # always add any pred timestamp
            ordinals.add(pred)
        ts = o2t[ordinal]
        farthest = ts.farthest_parent_ordinal()
        # also add directly connected older timestamps
        for ord in all_ordinals:
            if (ord <= ordinal) and (ord >= farthest):
                ordinals.add(ord)
        # dump nodes and adjust geometry
        id_to_node = {}
        for ord in ordinals:
            this_ts = o2t[ord]
            # print("updating", list(this_ts.id_to_node.keys()))
            id_to_node.update(this_ts.id_to_node)
        id_to_node_json = {}
        offsets = sorted(set(node._offset for node in id_to_node.values()))
        sordinals = sorted(ordinals)
        height = len(sordinals)
        for id, node in id_to_node.items():
            json_ob = node.json_object()
            # patch in relative geometry
            json_ob["x"] = offsets.index(node._offset)
            json_ob["y"] = height - sordinals.index(node.timestamp_ordinal) - 1
            # child marker
            json_ob["is_child"] = json_ob["timestamp_ordinal"] == ordinal
            id_to_node_json[id] = json_ob
        return dict(
            id_to_node=id_to_node_json,
            width=len(offsets),
            height=height,
            ordinals=sordinals,
        )

    def dimensions(self):
        # assuming timestamps start at 0 or 1
        height = max(self.ordinal_to_timestamp.keys()) + 1
        non_isolated = [n for n in self.id_to_node.values() if not n._is_isolated]
        width = 1
        if len(non_isolated) > 0:
            width = max(n._offset for n in non_isolated) + 1
        return (width, height)


def make_forest_from_haydens_json_graph(
    json_graph, label_assignment=None, add_parents=True, verbose=False
):
    """
    Read a JSON dump of matlab graph similar to "Gata6Nanog1.json".
    Return a forest.
    """
    # edges = json_graph['G_based_on_nn_combined']['Edges']
    nodes = edges = None
    # hack to handle random naming changes.
    edges = json_graph.get("Edges")
    nodes = json_graph.get("Nodes")
    for name in json_graph:
        graph = json_graph[name]
        if name.startswith("G"):
            edges = graph["Edges"]
            nodes = graph["Nodes"]
    assert edges is not None, "Could not find edges: " + repr(list(json_graph.keys()))
    result = Forest()
    all_ids = set()
    parent_map = {}
    for thing in nodes:
        node_id = thing["Name"]
        all_ids.add(node_id)
    for thing in edges:
        [parent_id, child_id] = thing["EndNodes"]
        all_ids.add(parent_id)
        all_ids.add(child_id)
        parent_map[child_id] = parent_id
    # parsed_strings = {}
    timestamps = set()
    node_map = {}
    for s in all_ids:
        [ts_string, label_string] = s.split("_")
        parsed = (ts, label) = (int(ts_string), int(label_string))
        timestamps.add(ts)
        # parsed_strings[s] = parsed
        if label_assignment is not None:
            assigned = label_assignment.get(s)
            if assigned is not None:
                label = assigned
                # print(s, "assigned to", assigned)
            else:
                if verbose:
                    print("no correction label assigned for node", s)
                label = None
        n = result.add_node(s, ts, label)
        node_map[s] = n
    if add_parents:
        for child_id, parent_id in parent_map.items():
            child = node_map[child_id]
            parent = node_map[parent_id]
            parent.set_child(child)
    return result


def read_json_file(path):
    """
    Read a JSON file and return the parsed object.
    """
    import json

    with open(path, "r") as f:
        json_ob = json.load(f)

    return make_forest_from_haydens_json_graph(json_ob)


def get_nodes_around(node, forest, window=4):
    """
    Given a node and forest, return the list of nodes before and after (excluding current node)
    for the same track (nucleus), sorted by timestamp_ordinal.
    Returns: (nodes_before, nodes_after)
    """
    track_root = node.track_ancestor()
    track = forest.id_to_track[track_root.node_id]
    ts_to_node = {n.timestamp_ordinal: n for n in track.id_to_node.values()}
    ordinals = sorted(ts_to_node.keys())
    idx = ordinals.index(node.timestamp_ordinal)

    nodes_before = []
    nodes_after = []
    for i in range(1, window + 1):
        if idx - i >= 0:
            nodes_before.append(ts_to_node[ordinals[idx - i]])
        if idx + i < len(ordinals):
            nodes_after.append(ts_to_node[ordinals[idx + i]])
    return list(reversed(nodes_before)), nodes_after


def classify_node(node, final_frame, forest=None, stable_window=4):
    """
    Classify a node based on its lineage and children count.

    Args:
        node: Node to classify
        final_frame: The last frame in the dataset (used for death classification)
        forest: Optional lineage forest for stable window checks
        stable_window: Number of frames to consider for stable classification
    Returns:
        str: Classification type of the node

    Classification types:
    - "mitotic": Node has 2 or more children (indicating division)
    - "new_daughter": Node has 1 child and its parent has 2 or more children
    - "death": Node has no children and its timestamp is before the final frame
    - "stable": Node has 1 child and no division events in the stable window
    - "unknown": Node does not fit any of the above categories
    """

    children_count = len(node.id_to_child)

    if children_count >= 2:
        return "mitotic"
    elif node.parent and len(node.parent.id_to_child) >= 2:
        return "new_daughter"
    elif children_count == 0 and node.timestamp_ordinal < final_frame:
        return "death"
    elif children_count == 1 and forest is not None:
        nodes_before, nodes_after = get_nodes_around(node, forest, window=stable_window)
        window_nodes = nodes_before + nodes_after
        print(
            f"Current node: {node.node_id}, Window nodes: {[n.node_id for n in window_nodes]}"
        )

        if len(window_nodes) < math.floor(
            stable_window * 1.5
        ):  # if not enough nodes in the window
            return "unknown"

        for n in window_nodes:
            # Only check direct event, not recursively
            c = _classify_event_type(n, final_frame)

            if (
                c != "stable"
            ):  # if any of the nodes in the window is not stable then current node is not stable
                return "unknown"
        return "stable"
    else:
        return "unknown"


def _classify_event_type(node, final_frame):
    children_count = len(node.id_to_child)
    if children_count >= 2:
        return "mitotic"
    elif node.parent and len(node.parent.id_to_child) >= 2:
        return "new_daughter"
    elif children_count == 0 and node.timestamp_ordinal < final_frame:
        return "death"
    elif children_count == 1:
        return "stable"
    else:
        return "unknown"


def classify_nodes_by_timestamp_example(
    nodes, forest, timestamp, final_frame, stable_window=4
):
    """
    Classify nodes in a forest based on their lineage and children count at a specific timestamp.

    Args:
        forest: Forest containing the nodes
        timestamp: Timestamp to classify nodes for
        final_frame: The last frame in the dataset (used for death classification)
        stable_window: Number of frames to consider for stable classification
    Returns:
        dict: Classification of nodes at the specified timestamp
    """

    classified_nodes = {}
    for node in nodes:
        classification = classify_node(node, final_frame, forest, stable_window)
        classified_nodes[node.node_id] = classification
    return classified_nodes


if __name__ == "__main__":
    # Sample usage of the lineage tree code
    forest = read_json_file(
        "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/220321_stack11/LineageGraph.json"
    )

    forest.find_tracks_and_lineages()

    # Group nodes by timestamp
    nodes_by_timestamp = defaultdict(list)
    for node in forest.id_to_node.values():
        nodes_by_timestamp[node.timestamp_ordinal].append(node)

    sorted_timestamps = sorted(nodes_by_timestamp.keys())
    final_frame = max(sorted_timestamps)

    timestamp = 1
    stable_window = 4

    # [IN REAL USAGE, We will loop thorugh all timestamps]
    classified = classify_nodes_by_timestamp_example(
        nodes_by_timestamp[timestamp], forest, timestamp, final_frame, stable_window
    )
    print("Classified Nodes at Timestamp", timestamp)
    for node_id, classification in classified.items():
        print(f"Node ID: {node_id}, Classification: {classification}")
