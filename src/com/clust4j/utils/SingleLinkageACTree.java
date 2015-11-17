package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

public class SingleLinkageACTree extends AbstractBinaryTree<Integer> {
	private static final long serialVersionUID = -8450284575258068092L;
	private final AgglomNode root;
	
	/**
	 * Used to retrieve the clusters out of the data from the tree
	 */
	private final TreeMap<Integer, double[]> data;
	final private String rep;
	final private int size;
	
	
	private SingleLinkageACTree(final TreeMap<Integer, EntryPair<Integer, Integer>> mapping, final double[][] data) {
		super();
		
		
		// Since we pulled the data from bottom up, we need to map it...
		int current_idx = 2*data.length - 1;
		this.data = new TreeMap<Integer, double[]>();
		for(double[] d: data)
			this.data.put(current_idx--, d);
		
		
		
		this.rep = mapping.toString();
		
		// Build the root, then progressively add the values in...
		TreeMap<Integer, AgglomNode> nodeMapping = new TreeMap<>();

		Integer index; 
		AgglomNode current, left, right;
		EntryPair<Integer, Integer> children;
		
		root = new AgglomNode(1);
		nodeMapping.put(1, root);
		
		for(Map.Entry<Integer, EntryPair<Integer, Integer>> entry: mapping.entrySet()) {
			index = entry.getKey();
			children = entry.getValue();
			
			current = nodeMapping.get(index);
			if(null == current) { // Haven't seen key before
				nodeMapping.put(index, current = new AgglomNode(index));
			}
			
			if(null != children) {
				left = new AgglomNode(children.getKey());
				right = new AgglomNode(children.getValue());
				
				current.left = left;
				current.right = right;
				
				nodeMapping.put(children.getKey(), left);
				nodeMapping.put(children.getValue(), right);
			}
		}
		
		size = root.size();
		nodeMapping = null; // Force GC
	}
	
	
	
	public static SingleLinkageACTree build(final double[][] data, final GeometricallySeparable dist) {
		return build(data, dist, true);
	}
	
	/**
	 * Builds a SingleLinkageAgglomerativeClusterTree using Johnson's algorithm:
	 * 
	 * <ul>
	 * 1. Begin with the disjoint clustering having level L(0) = 0 and sequence number m = 0.
     * </ul>
     * 
     * <ul>
     * 2. Find the least dissimilar pair of clusters in the current clustering, 
     * say pair (r), (s), according to:
     *
     *		<ul>d[(r),(s)] = min d[(i),(j)]</ul>
	 * 
	 * where the minimum is over all pairs of clusters in the current clustering.
	 * </ul>
	 * 
	 * <ul>
	 * 3. Increment the sequence number : m = m +1. Merge clusters (r) and (s) into a 
	 * single cluster to form the next clustering m. Set the level of this clustering to:
	 * 
	 * 		<ul>L(m) = d[(r),(s)]</ul>
	 * </ul>
	 * 
	 * <ul>
	 * 4. Update the proximity matrix, D, by deleting the rows and columns corresponding 
	 * to clusters (r) and (s) and adding a row and column corresponding to the newly formed 
	 * cluster. The proximity between the new cluster, denoted (r,s) and old cluster (k) is defined in this way:
	 * 
	 * 		<ul>d[(k), (r,s)] = min d[(k),(r)], d[(k),(s)]</ul>
	 * </ul>
	 * 
	 * <ul>
	 * 5. If all objects are in one cluster, stop. Else, go to step 2.
	 * </ul>
	 * 
	 * @param dat
	 * @param dist
	 * @param copy
	 * @return the Agglomerative Cluster tree
	 */
	public static SingleLinkageACTree build(final double[][] dat, final GeometricallySeparable dist, final boolean copy) {
		final double[][] data = copy ? ClustUtils.copyMatrix(dat) : dat;
		int m = data.length;
		int currentCluster = (2 * m) - 1; // There will always be 2M-1 clusters at the end
		if(m < 1)
			throw new IllegalArgumentException("empty data");
		
		
		// The structure that will contain the cluster number mapped to the two clusters
		// that make it up. If the value is null, then they are leaf clusters
		TreeMap<Integer, EntryPair<Integer, Integer>> clusterMap = new TreeMap<>();
		HashMap<Cluster, Integer> clusterNumbers = new HashMap<>();
		
		
		// Create the N clusters of data...
		Cluster c;
		ArrayList<Cluster> clusters = new ArrayList<Cluster>();
		for(double[] d: data) {
			c = new Cluster();
			c.add(d);
			clusters.add(c);

			clusterMap.put(currentCluster, null);
			clusterNumbers.put(c, currentCluster--);
		}
		
		
		/* So we now have N 'clusters' in data... 
		 * at each section, we find the two clusters closest to one another...
		 * Create one big distance matrix, calculate the upper triangular distance.
		 * For each iteration, when finding the two closest points, merge and remove
		 * the two original rows/cols from the distanc matrix, then calculate distance
		 * from each other cluster's centroid to the new one. This constitutes the new
		 * distance matrix.
		 */
		Array2DRowRealMatrix distance = new Array2DRowRealMatrix(ClustUtils.distanceMatrix(data, dist), false); // Don't force copy
		
		
		/*
		 * At this point, index J in clusters corresponds to either row or col J in the dist matrix...
		 * need to keep this continuity...
		 */
		
		
		// While the distance matrix is not comprised of merely the last two clusters
		EntryPair<Integer, Integer> closest, mergedClusterIndices;
		Cluster a, b;
		int i, j, newM;
		double[] centroid;
		double[][] newDataRef;
		while(m > 1) {
			
			// Find the row/col indices that get merged next
			closest = minDistInDistMatrix(distance);
			i = closest.getKey();
			j = closest.getValue();
			
			// Extract the clusters to be merged...
			a = clusters.get(i);
			b = clusters.get(j);
			mergedClusterIndices = new EntryPair<Integer, Integer>(clusterNumbers.get(a), clusterNumbers.get(b));
			clusterMap.put(currentCluster, mergedClusterIndices);
			
			// Must remove `j` first to avoid left shift
			clusters.remove(j);
			clusters.remove(i);
			
			// Now merge them:
			c = merge(a, b);
			clusters.add(c);
			clusterNumbers.put(c, currentCluster--);
			centroid = c.centroid();
			
			// Now remove i,j from dist matrix... rows AND cols
			newM = m - 1;
			newDataRef = new double[newM][newM];
			int row=0; int col=0;
			for(int k = 0; k < m; k++) {
				if(k == i || k == j)
					continue;
				
				for(int u = 0; u < m; u++) {
					if(u == i || u == j)
						continue;
					
					newDataRef[row][col] = distance.getEntry(k, u);
					col++;
				}
				
				col = 0;
				row++;
			}
			
			
			// Now add in the NEW last col, which is the dist from the new centroid 
			// to the other cluster centroids...
			for(int k = 0; k < newM - 1; k++) // Skip the last one, which is the new cluster...
				newDataRef[k][newM - 1] = dist.distance(clusters.get(k).centroid(), centroid);
			
			
			// Now assign to the new distance matrix...
			distance = new Array2DRowRealMatrix(newDataRef, false);
			m = newM;
		}
		
		// Force GC to free up some space overhead
		clusters = null;
		newDataRef = null;
		
		
		return new SingleLinkageACTree(clusterMap, data);
	}
	
	final private static Cluster merge(final Cluster a, final Cluster b) {
		final Cluster merge = new Cluster();
		
		final int n = a.get(0).length;
		final Cluster[] car = new Cluster[]{a, b};
		
		for(Cluster cl: car) {
			for(double[] d: cl) {
				double[] copy = new double[n];
				System.arraycopy(d, 0, copy, 0, n);
				merge.add(copy);
			}
		}
		
		return merge;
	}
	
	final private static EntryPair<Integer, Integer> minDistInDistMatrix(final AbstractRealMatrix data) {
		final int m = data.getRowDimension();
		
		int minRow = -1;
		int minCol = -1;
		double min = Double.MAX_VALUE;
		
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				final double current = data.getEntry(i, j);
				if(current < min) {
					minRow = i;
					minCol = j;
					min = current;
				}
			}
		}
		
		return new EntryPair<>(minRow, minCol);
	}
	
	

	@Override
	public AgglomNode getRoot() {
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

	
	
	
	public class AgglomNode extends AbstractBinaryTree.BaseBinaryTreeNode<Integer> {
		private static final long serialVersionUID = -982952921431298127L;
		
		private final Integer value;
		private AgglomNode left = null;
		private AgglomNode right = null;
		
		
		
		protected AgglomNode(Integer c) {
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
		public AgglomNode leftChild() {
			return left;
		}

		@Override
		public AgglomNode rightChild() {
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
				final double[] row = SingleLinkageACTree.this.data.get(leaf);
				final double[] copy = new double[row.length];
				System.arraycopy(row, 0, copy, 0, row.length);
				c.add(copy);
			}
			
			return c;
		}
		
		private void getLeafNodes(AgglomNode node, Collection<Integer> leaves) {
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
		protected AgglomNode locate(Integer value) {
			if(this.value.intValue() == value.intValue())
				return this;
			
			AgglomNode result = null;
			if(hasLeft())
				result = left.locate(value);
			
			// If it's not null, it's been found
			if(null != result)
				return result;
			
			// Didn't find in left branch...
			if(hasRight())
				result = right.locate(value);
			
			// Either null or AgglomNode
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
		
		private final Collection<Integer> valuesRecurse(final AgglomNode root, final Collection<Integer> coll) {
			coll.add(root.value);
			if(root.hasLeft())
				valuesRecurse(root.left, coll);
			if(root.hasRight())
				valuesRecurse(root.right, coll);
			return coll;
		}
		
	}
}
