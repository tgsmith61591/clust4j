package com.clust4j.algo;

import java.util.ArrayList;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class KMedoids extends AbstractKCentroidClusterer {
	final public static GeometricallySeparable DEF_DIST = Distance.MANHATTAN;
	
	final public static class KMedoidsPlanner extends BaseKCentroidPlanner {
		public KMedoidsPlanner(int k) {
			super(k);
			super.setDist(DEF_DIST);
		}
		
		@Override
		public KMedoidsPlanner setDist(final GeometricallySeparable dist) {
			return (KMedoidsPlanner) super.setDist(dist);
		}
		
		@Override
		public KMedoidsPlanner setMaxIter(final int max) {
			return (KMedoidsPlanner) super.setMaxIter(max);
		}
		
		public KMedoidsPlanner setScale(final boolean scale) {
			return (KMedoidsPlanner) super.setScale(scale);
		}
	}
	
	public KMedoids(final AbstractRealMatrix data, final int k) {
		this(data, new KMedoidsPlanner(k));
	}
	
	public KMedoids(final AbstractRealMatrix data, final KMedoidsPlanner builder) {
		super(data, builder);
	}
	
	/**
	 * Calculates the intracluster cost
	 * @param inCluster
	 * @return the sum of manhattan distances between vectors and the centroid
	 */
	final double getCost(final ArrayList<Integer> inCluster, final double[] newCentroid) {
		// Now calc the dissimilarity in the cluster
		double sumI = 0;
		for(Integer rec : inCluster) { // Row nums of belonging records
			final double[] record = data.getRow(rec);
			sumI += KMedoids.DEF_DIST.distance(record, newCentroid);
		}
		
		return sumI;
	}
	
	@Override
	public String getName() {
		return "KMedoids";
	}

	@Override
	public void train() {
		if(isTrained)
			return;
		
		// TODO:
	}
}
