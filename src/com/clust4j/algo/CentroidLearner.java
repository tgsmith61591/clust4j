package com.clust4j.algo;

import java.util.ArrayList;

public interface CentroidLearner extends java.io.Serializable {
	/**
	 * Returns the centroid records
	 * @return an ArrayList of the centroid records
	 */
	public ArrayList<double[]> getCentroids();
}
