package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

import com.clust4j.algo.AbstractHierarchicalClusterer;

public class HierarchicalClusterTree extends AbstractBinaryTree<Integer> {
	private static final long serialVersionUID = -8450284575258068092L;
	private final HierarchicalNode root;
	
	/**
	 * Used to retrieve the clusters out of the data from the tree
	 */
	private final TreeMap<Integer, double[]> data;
	final private String rep;
	final private int size;
	
	
	protected HierarchicalClusterTree(final TreeMap<Integer, EntryPair<Integer, Integer>> mapping, 
			final double[][] data, final AbstractHierarchicalClusterer clusterer) {
		super();
		
		final boolean verbose = clusterer.getVerbose();
		
		// Since we pulled the data from bottom up, we need to map it...
		int current_idx = 2*data.length - 1;
		this.data = new TreeMap<Integer, double[]>();
		for(double[] d: data)
			this.data.put(current_idx--, d);
		
		if(verbose) clusterer.info("mapping clusters to tree structure (root=1). Use '.getTree()'");
		
		this.rep = mapping.toString();
		
		// Build the root, then progressively add the values in...
		TreeMap<Integer, HierarchicalNode> nodeMapping = new TreeMap<>();
		

		Integer index; 
		HierarchicalNode current, left, right;
		EntryPair<Integer, Integer> children;
		
		root = new HierarchicalNode(1);
		nodeMapping.put(1, root);
		
		for(Map.Entry<Integer, EntryPair<Integer, Integer>> entry: mapping.entrySet()) {
			index = entry.getKey();
			children = entry.getValue();
			
			current = nodeMapping.get(index);
			if(null == current) { // Haven't seen key before
				nodeMapping.put(index, current = new HierarchicalNode(index));
			}
			
			if(null != children) {
				if(verbose) clusterer.trace("cluster " + index + " is derived from clusters " + children);
				
				left = new HierarchicalNode(children.getKey());
				right = new HierarchicalNode(children.getValue());
				
				current.left = left;
				current.right = right;
				
				nodeMapping.put(children.getKey(), left);
				nodeMapping.put(children.getValue(), right);
			}
		}
		
		size = root.size();
		if(verbose) {
			clusterer.info("hierarchical tree building complete. Total number of clusters: " + size);
			clusterer.info("root (agglomeration)=cluster 1. Use '.getTree()'");
			clusterer.info("leaf nodes = individual records. From a node, use '.getCluster()' to get records");
		}
		
		nodeMapping = null; // Force GC
	}
	
	

	@Override
	public HierarchicalNode getRoot() {
		return root;
	}
	
	@Override
	public int size() {
		return size;
	}
	
	@Override
	public String toString() {
		return rep;
	}

	
	
	
	public class HierarchicalNode extends AbstractBinaryTree.BaseBinaryTreeNode<Integer> {
		private static final long serialVersionUID = -982952921431298127L;
		
		private final Integer value;
		private HierarchicalNode left = null;
		private HierarchicalNode right = null;
		
		
		
		protected HierarchicalNode(Integer c) {
			value = c;
		}
		
		

		@Override
		protected boolean hasLeft() {
			return null != left;
		}

		@Override
		protected boolean hasRight() {
			return null != right;
		}

		@Override
		public HierarchicalNode leftChild() {
			return left;
		}

		@Override
		public HierarchicalNode rightChild() {
			return right;
		}
		
		/**
		 * Returns a copy of the records in this level of cluster
		 * @return
		 */
		public Cluster getCluster() {
			Collection<Integer> leaves = new ArrayList<Integer>();
			getLeafNodes(this, leaves);
			Cluster c = new Cluster();
			
			for(Integer leaf: leaves) {
				final double[] row = HierarchicalClusterTree.this.data.get(leaf);
				final double[] copy = new double[row.length];
				System.arraycopy(row, 0, copy, 0, row.length);
				c.add(copy);
			}
			
			return c;
		}
		
		private void getLeafNodes(HierarchicalNode node, Collection<Integer> leaves) {
			if(!node.hasLeft() && !node.hasRight()) // LEAF!
				leaves.add(node.value);
			
			if(node.hasLeft())
				getLeafNodes(node.left, leaves);
			if(node.hasRight())
				getLeafNodes(node.right, leaves);
				
			return;
		}

		@Override
		public Integer getValue() {
			return value;
		}

		/**
		 * Locates in pre-order...
		 */
		@Override
		public HierarchicalNode locate(Integer value) {
			if(this.value.intValue() == value.intValue())
				return this;
			
			HierarchicalNode result = null;
			if(hasLeft())
				result = left.locate(value);
			
			// If it's not null, it's been found
			if(null != result)
				return result;
			
			// Didn't find in left branch...
			if(hasRight())
				result = right.locate(value);
			
			// Either null or HierarchicalNode
			return result;
		}

		@Override
		protected void prune() {
			right = null;
			left = null;
		}

		/**
		 * Collects the values of the tree in PRE ORDER
		 * @return
		 */
		@Override
		public Collection<Integer> values() {
			final ArrayList<Integer> values = new ArrayList<>();
			return valuesRecurse(this, values);
		}
		
		private final Collection<Integer> valuesRecurse(final HierarchicalNode root, final Collection<Integer> coll) {
			coll.add(root.value);
			if(root.hasLeft())
				valuesRecurse(root.left, coll);
			if(root.hasRight())
				valuesRecurse(root.right, coll);
			return coll;
		}
		
	}
}
