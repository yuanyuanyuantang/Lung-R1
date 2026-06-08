"""
graphcot: Standalone Graph Storage Module
---------------------------------------------
Provides a lightweight NetworkX-based graph storage implementation.
Self-contained, no external dependencies on the original graphcot.
"""

import os
import json
import networkx as nx
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from graphcot import BaseGraphStorage


class SimpleGraphStorage(BaseGraphStorage):
    """
    A simple, standalone graph storage using NetworkX.
    Capable of loading nodes.csv and edges.csv (like the original LungKG).
    """
    
    def __init__(self):
        self._graph = nx.Graph()

    def load_from_csv(self, nodes_path: str, edges_path: str):
        """
        Load graph from CSV files (compatible with Neo4j export format).
        
        Args:
            nodes_path: Path to nodes.csv
            edges_path: Path to edges.csv
            
        Note:
            Files that do not exist are silently skipped.
            This allows loading only nodes or only edges if needed.
        """
        
        # Load Nodes
        if os.path.exists(nodes_path):
            df_nodes = pd.read_csv(nodes_path, dtype=str).fillna("")
            for _, row in df_nodes.iterrows():
                # Adapt to typical headers like :ID, name, :LABEL
                nid = row.get(":ID", row.get("id", ""))
                if not nid: continue
                
                # Store all attributes
                attrs = row.to_dict()
                # Ensure 'name' and 'entity_type' exist for generator
                attrs["name"] = row.get("name", row.get(":ID", ""))
                attrs["entity_type"] = row.get(":LABEL", row.get("type", "Entity"))
                
                self._graph.add_node(nid, **attrs)
        
        # Load Edges
        if os.path.exists(edges_path):
            df_edges = pd.read_csv(edges_path, dtype=str).fillna("")
            for _, row in df_edges.iterrows():
                src = row.get(":START_ID", row.get("source", ""))
                tgt = row.get(":END_ID", row.get("target", ""))
                if not src or not tgt: continue
                
                attrs = row.to_dict()
                # Ensure 'relation_type' exists
                attrs["relation_type"] = row.get(":TYPE", row.get("relation", "related_to"))
                
                self._graph.add_edge(src, tgt, **attrs)

    def get_all_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return all nodes with data."""
        return list(self._graph.nodes(data=True))

    def get_all_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return all edges with data."""
        return list(self._graph.edges(data=True))

    def get_node_degree(self, node_id: str) -> int:
        return self._graph.degree[node_id] if node_id in self._graph else 0
