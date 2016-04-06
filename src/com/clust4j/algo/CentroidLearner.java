package com.clust4j.algo;

import java.util.ArrayList;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface CentroidLearner extends java.io.Serializable {
	/**
	 * Returns the centroid records
	 * @return an ArrayList of the centroid records
	 */
	public ArrayList<double[]> getCentroids();
	public int[] predict(AbstractRealMatrix newData);
}
