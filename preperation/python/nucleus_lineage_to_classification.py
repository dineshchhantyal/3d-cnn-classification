"""
Nucleus Analyzer Module

This module provides functions to analyze nuclei in lineage trees by timestamp
and classify them based on their biological states.
"""

from collections import defaultdict


def analyze_specific_timestamp(forest, target_timestamp):
    """
    Print all nodes in a specific timestamp with their classifications.

    Args:
        forest: Forest object containing the lineage tree
        target_timestamp: The timestamp to analyze

    Returns:
        tuple: (nodes_in_timestamp, classifications_dict)
    """
    nodes_in_timestamp = []

    final_frame = max(forest.ordinal_to_timestamp.keys())  # Get the last frame

    for node in forest.id_to_node.values():
        if node.timestamp_ordinal == target_timestamp:
            nodes_in_timestamp.append(node)

    print(f"\n🎯 DETAILED ANALYSIS - TIMESTAMP {target_timestamp}")
    print(f"📊 Total nodes: {len(nodes_in_timestamp)}")
    print("=" * 60)

    classifications = defaultdict(int)

    for i, node in enumerate(nodes_in_timestamp):
        parent_info = f"{node.parent.node_id}" if node.parent else "None"
        children_info = list(node.id_to_child.keys()) if node.id_to_child else []

        # Use the centralized classification function
        classification = classify_node(node, final_frame)
        classifications[classification] += 1

        print(f"Node {i+1}/{len(nodes_in_timestamp)}: {node.node_id}")
        print(f"   Label: {node.label}")
        print(f"   Parent: {parent_info}")
        print(f"   Children: {children_info}")
        print(f"   Classification: {classification}")
        print()

    # Print summary
    print(f"📊 CLASSIFICATION SUMMARY:")
    for classification, count in classifications.items():
        percentage = (count / len(nodes_in_timestamp)) * 100
        print(f"   • {classification.upper()}: {count} nodes ({percentage:.1f}%)")

    return nodes_in_timestamp, dict(classifications)


def nucleus_extractor(forest, timeframe=1, output_dir="extracted_nuclei"):
    """
    Extract nuclei for training using lineage tree information and parallel processing.

    Args:
        forest: Forest object containing the lineage tree
        timeframe: Timeframe for extraction (default is 1)
            when 1, t-1, t, t+1 are used for extraction; when 2, t-2, t-1, t, t+1, t+2 are used, etc.
        output_dir: Directory to save extracted nuclei

    Returns:
        bool: True if extraction was successful, False otherwise
    """

    # Group nodes by timestamp
    nodes_by_timestamp = defaultdict(list)
    for node in forest.id_to_node.values():
        nodes_by_timestamp[node.timestamp_ordinal].append(node)
    sorted_timestamps = sorted(nodes_by_timestamp.keys())
    final_frame = max(sorted_timestamps)
    print(
        "📅 NODES BY TIMESTAMP"
        f" (Total timestamps: {len(sorted_timestamps)})"
        f" (Final frame: {final_frame})"
    )

    timestamps_volumnes_queue = (
        dque.Queue()
    )  # load the tif volumes for each timestamp [for eg. current, next,  timestamp we are processing -

    # now loop thoguh the timestamps and extract the nuclei in parallel
    for timestamp in sorted_timestamps:
        nodes = nodes_by_timestamp[timestamp]
        node_count = len(nodes)

        # timestamp_volume = load the tif volume

        print(f"\n🕒 TIMESTAMP {timestamp} - {node_count} nodes")
        print("-" * 40)

        # Extract nuclei for each node in this timestamp
        for node in nodes:
            parent_info = (
                f"Parent: {node.parent.node_id}" if node.parent else "Parent: None"
            )
            classification = classify_node(node, final_frame).upper()

            print(f"   Node ID: {node.node_id}")
            print(f"      Label: {node.label}")
            print(f"      {parent_info}")
            print(
                f"      Children: {list(node.id_to_child.keys()) if node.id_to_child else 'None'}"
            )
            print(f"      Classification: {classification}")
            print()

            # save the node information using the i/o operations and other cropping code


def print_nodes_by_timestamp(forest, max_timestamps=10, max_nodes_per_timestamp=20):
    """
    Print node information organized by timestamp.

    Args:
        forest: Forest object containing the lineage tree
        max_timestamps: Maximum number of timestamps to show
        max_nodes_per_timestamp: Maximum nodes to show per timestamp

    Returns:
        dict: Dictionary mapping timestamps to node lists
    """
    # Group nodes by timestamp
    nodes_by_timestamp = defaultdict(list)
    for node in forest.id_to_node.values():
        nodes_by_timestamp[node.timestamp_ordinal].append(node)

    # Sort timestamps
    sorted_timestamps = sorted(nodes_by_timestamp.keys())
    final_frame = max(sorted_timestamps)

    print("📅 NODES BY TIMESTAMP")
    print("=" * 60)

    # Show first few timestamps
    for i, timestamp in enumerate(sorted_timestamps[:max_timestamps]):
        nodes = nodes_by_timestamp[timestamp]
        node_count = len(nodes)

        print(f"\n🕒 TIMESTAMP {timestamp}")
        print(f"   Number of nodes: {node_count}")
        print("-" * 40)

        # Show nodes in this timestamp
        for j, node in enumerate(nodes[:max_nodes_per_timestamp]):
            parent_info = (
                f"Parent: {node.parent.node_id}" if node.parent else "Parent: None"
            )

            # Use the centralized classification function
            classification = classify_node(node, final_frame).upper()

            print(f"   Node {j+1}: {node.node_id}")
            print(f"      Label: {node.label}")
            print(f"      {parent_info}")
            print(
                f"      Children: {list(node.id_to_child.keys()) if node.id_to_child else 'None'}"
            )
            print(f"      Classification: {classification}")
            print()

        if len(nodes) > max_nodes_per_timestamp:
            print(f"   ... and {len(nodes) - max_nodes_per_timestamp} more nodes")

        print(f"   📊 Summary: {node_count} total nodes in timestamp {timestamp}")

    if len(sorted_timestamps) > max_timestamps:
        print(f"\n... and {len(sorted_timestamps) - max_timestamps} more timestamps")

    # Overall summary
    total_nodes = sum(len(nodes) for nodes in nodes_by_timestamp.values())
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   Total timestamps: {len(sorted_timestamps)}")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Average nodes per timestamp: {total_nodes / len(sorted_timestamps):.1f}")

    return dict(nodes_by_timestamp)


def classify_node(node, final_frame):
    """
    Classify a single node based on its properties.

    Args:
        node: Node object to classify
        final_frame: The final timestamp in the dataset

    Returns:
        str: Classification ('mitotic', 'new_daughter', 'death', 'stable', 'unknown')
    """
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


def get_timestamp_statistics(forest):
    """
    Get basic statistics about timestamps in the forest.

    Args:
        forest: Forest object containing the lineage tree

    Returns:
        dict: Statistics dictionary
    """
    timestamps = sorted(forest.ordinal_to_timestamp.keys())
    nodes_per_timestamp = defaultdict(int)

    for node in forest.id_to_node.values():
        nodes_per_timestamp[node.timestamp_ordinal] += 1

    stats = {
        "total_timestamps": len(timestamps),
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
        "total_nodes": len(forest.id_to_node),
        "average_nodes_per_timestamp": len(forest.id_to_node) / len(timestamps),
        "nodes_per_timestamp": dict(nodes_per_timestamp),
    }

    return stats


if __name__ == "__main__":
    # Example usage - would need to import lineage_tree and load forest
    print("This module provides functions for analyzing nuclei by timestamp.")
    print("Import this module and use the functions with your forest object.")
