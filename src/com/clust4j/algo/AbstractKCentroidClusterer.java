package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.GeometricallySeparable;

public abstract class AbstractKCentroidClusterer extends AbstractPartitionalClusterer implements CentroidLearner {
	final public static int DEF_MAX_ITER = 100;

	final protected int maxIter;
	protected boolean isTrained = false;
	protected boolean converged = false;
	protected double cost;

	protected int[] labels = null;
	protected int iter = 0;
	
	final protected int m;
	
	/** Key is the group label, value is the corresponding centroid */
	protected ArrayList<double[]> centroids = new ArrayList<double[]>();
	protected TreeMap<Integer, ArrayList<Integer>> cent_to_record = null;
	
	public AbstractKCentroidClusterer(AbstractRealMatrix data, 
			AbstractKCentroidClusterer.BaseKCentroidPlanner planner) {
		super(data, planner, planner.k);
		
		this.maxIter = planner.maxIter;
		this.m = data.getRowDimension();
		
		initCentroids();
	}
	
	protected static class BaseKCentroidPlanner extends AbstractClusterer.BaseClustererPlanner {
		private int maxIter = DEF_MAX_ITER;
		private GeometricallySeparable dist = DEF_DIST;
		private boolean scale = DEF_SCALE;
		private int k;
		
		public BaseKCentroidPlanner(final int k) {
			this.k = k;
		}
		
		@Override
		public GeometricallySeparable getDist() {
			return dist;
		}
		
		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public BaseKCentroidPlanner setDist(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public BaseKCentroidPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}
		
		@Override
		public BaseKCentroidPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
	}

	
	public boolean didConverge() {
		return converged;
	}
	
	final private void initCentroids() {
		// Initialize centroids with K random records
		// Creates a list of integer sequence 0 -> nrow(data), then shuffles it
		// and takes the first K indices as the centroid records. Then manually
		// sets recordIndices to null to invoke GC to free up space
		ArrayList<Integer> recordIndices = new ArrayList<Integer>();
		for(int i = 0; i < data.getRowDimension(); i++) 
			recordIndices.add(i);
		Collections.shuffle(recordIndices, getSeed());
		
		for(int i = 0; i < k; i++) 
			centroids.add(data.getRow(recordIndices.get(i)));
		
		recordIndices = null;
	}
	
	@Override
	public ArrayList<double[]> getCentroids() {
		final ArrayList<double[]> cent = new ArrayList<double[]>();
		for(double[] d : centroids) {
			final double[] copy = new double[d.length];
			System.arraycopy(d, 0, copy, 0, d.length);
			cent.add(copy);
		}
		
		return cent;
	}
	
	public int getMaxIter() {
		return maxIter;
	}
	
	@Override
	public int[] getPredictedLabels() {
		return labels;
	}
	
	@Override
	public boolean isTrained() {
		return isTrained;
	}
	
	public int itersElapsed() {
		return iter;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		sb.append(getName() + " centroids: [");
		for(int i = 0; i < centroids.size(); i++)
			sb.append(Arrays.toString(centroids.get(i)) + (i == centroids.size()-1 ? "]":", "));
		return sb.toString();
	}
	
	public double totalCost() {
		return cost;
	}
}
