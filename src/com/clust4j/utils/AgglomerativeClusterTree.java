package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.algo.Cluster;

public class AgglomerativeClusterTree extends AbstractBinaryTree<Cluster> {
	private static final long serialVersionUID = -8450284575258068092L;
	private final AgglomNode root;
	
	
	
	private AgglomerativeClusterTree(final AgglomNode root) {
		super();
		this.root = root;
	}
	
	
	
	public static AgglomerativeClusterTree build(final double[][] data, final GeometricallySeparable dist) {
		return build(data, dist, true);
	}
	
	public static AgglomerativeClusterTree build(final double[][] dat, final GeometricallySeparable dist, final boolean copy) {
		final double[][] data = copy ? ClustUtils.copyMatrix(dat) : dat;
		int m = data.length;
		if(m < 1)
			throw new IllegalArgumentException("empty data");
		final int n = data[0].length;
		
		
		// Create the N clusters of data...
		Cluster c;
		ArrayList<Cluster> clusters = new ArrayList<Cluster>();
		for(double[] d: data) {
			c = new Cluster();
			c.add(d);
			clusters.add(c);
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
		EntryPair<Integer, Integer> closest;
		Cluster a, b;
		int i, j, newM;
		double[] centroid;
		double[][] newDataRef;
		while(m > 2) {
			
			// Find the row/col indices that get merged next
			closest = minDistInDistMatrix(distance);
			i = closest.getKey();
			j = closest.getValue();
			
			// Extract the clusters to be merged...
			a = clusters.get(i);
			b = clusters.get(j);
			
			// Must remove `j` first to avoid left shift
			clusters.remove(j);
			clusters.remove(i);
			
			// Now merge them:
			c = merge(a, b);
			clusters.add(c);
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
			
			
			// TODO : do something with C!!!
			
			m = newM;
		}
		
		// TODO Build the nodes from the root!
		return null;
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

	
	
	
	public static class AgglomNode extends AbstractBinaryTree.BaseBinaryTreeNode<Cluster> {
		private static final long serialVersionUID = -982952921431298127L;
		
		private final Cluster value;
		private AgglomNode left = null;
		private AgglomNode right = null;
		
		
		
		protected AgglomNode(Cluster c) {
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
		protected AgglomNode leftChild() {
			return left;
		}

		@Override
		protected AgglomNode rightChild() {
			return right;
		}

		@Override
		public Cluster getValue() {
			return value;
		}

		@Override
		protected AgglomNode locate(Cluster value) {
			// TODO Auto-generated method stub
			return null;
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
		public Collection<Cluster> values() {
			final ArrayList<Cluster> values = new ArrayList<>();
			return valuesRecurse(this, values);
		}
		
		private final static Collection<Cluster> valuesRecurse(final AgglomNode root, final Collection<Cluster> coll) {
			coll.add(root.value);
			if(root.hasLeft())
				valuesRecurse(root.left, coll);
			if(root.hasRight())
				valuesRecurse(root.right, coll);
			return coll;
		}
		
	}
}
