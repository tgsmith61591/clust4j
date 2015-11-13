package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.GeometricallySeparable;

public abstract class AbstractKCentroidClusterer extends AbstractPartitionalClusterer implements CentroidLearner {
	final public static int DEF_MAX_ITER = 100;
	final public static double DEF_MIN_CHNG = 0.005;
	

	final protected int maxIter;
	final protected double minChange;
	
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
		this.minChange = planner.minChange;
		this.m = data.getRowDimension();
		
		initCentroids();
	}
	
	public static class BaseKCentroidPlanner extends AbstractClusterer.BaseClustererPlanner {
		private int maxIter = DEF_MAX_ITER;
		private GeometricallySeparable dist = DEF_DIST;
		private boolean scale = DEF_SCALE;
		private double minChange = DEF_MIN_CHNG;
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

		public BaseKCentroidPlanner setMinChangeStoppingCriteria(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public BaseKCentroidPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
	}

	
	


	final protected TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabels() {
		/* Key is the closest centroid, value is the records that belong to it */
		TreeMap<Integer, ArrayList<Integer>> cent = new TreeMap<Integer, ArrayList<Integer>>();
		
		/* Loop over each record in the matrix */
		for(int rec = 0; rec < m; rec++) {
			final double[] record = data.getRow(rec);
			double min_dist = Double.MAX_VALUE;
			int closest_cent = 0;
			
			/* Loop over every centroid, get calculate dist from record,
			 * identify the closest centroid to this record */
			for(int i = 0; i < k; i++) {
				final double[] centroid = centroids.get(i);
				final double dis = getDistanceMetric().distance(record, centroid);
				
				/* Track the current min distance. If dist
				 * is shorter than the previous min, assign
				 * new closest centroid to this record */
				if(dis < min_dist) {
					min_dist = dis;
					closest_cent = i;
				}
			}
			
			labels[rec] = closest_cent;
			if(cent.get(closest_cent) == null)
				cent.put(closest_cent, new ArrayList<Integer>());
			
			cent.get(closest_cent).add(rec);
		}
		
		return cent;
	}
	
	public boolean didConverge() {
		return converged;
	}
	
	public double getCostOfSystem() {
		double cost = 0;
		double[] oid;
		ArrayList<Integer> medoid_members;
		for(Map.Entry<Integer, ArrayList<Integer>> medoid_entry : cent_to_record.entrySet()) {
			oid = centroids.get(medoid_entry.getKey());
			medoid_members = medoid_entry.getValue();
			cost += getCost(medoid_members, oid);
		}
		
		return cost;
	}
	
	public double getMinChange() {
		return minChange;
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
	
	abstract double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid);
}
