package com.clust4j.utils;

import java.util.ArrayList;

public interface CentroidLearner {
	/**
	 * Returns the centroid records
	 * @return an ArrayList of the centroid records
	 */
	public ArrayList<double[]> getCentroids();
}
