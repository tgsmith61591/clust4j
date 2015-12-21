package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.Convergeable;
import com.clust4j.utils.PredictableClassifier;

public abstract class AbstractCentroidClusterer extends AbstractPartitionalClusterer 
		implements CentroidLearner, PredictableClassifier, Convergeable{
	private static final long serialVersionUID = -424476075361612324L;
	
	final protected int maxIter;
	final protected double minChange;
	final protected int[] init_centroid_indices;
	
	volatile protected boolean converged = false;
	volatile protected double cost;
	volatile protected int[] labels = null;
	volatile protected int iter = 0;
	
	/** Key is the group label, value is the corresponding centroid */
	volatile protected ArrayList<double[]> centroids = new ArrayList<double[]>();
	volatile protected TreeMap<Integer, ArrayList<Integer>> cent_to_record = null;

	public AbstractCentroidClusterer(AbstractRealMatrix data,
			CentroidClustererPlanner planner) {
		super(data, planner, planner.getK());
		
		this.maxIter = planner.getMaxIter();
		this.minChange = planner.getMinChange();
		this.init_centroid_indices = initCentroids();
	}

	public static abstract class CentroidClustererPlanner extends BaseClustererPlanner {
		abstract public int getK();
		abstract public int getMaxIter();
		abstract public double getMinChange();
	}
	
	/**
	 * Returns the ordered indices of the centroids
	 * @return
	 */
	final private int[] initCentroids() {
		// Initialize centroids with K random records
		// Creates a list of integer sequence 0 -> nrow(data), then shuffles it
		// and takes the first K indices as the centroid records. Then manually
		// sets recordIndices to null to invoke GC to free up space
		ArrayList<Integer> recordIndices = new ArrayList<Integer>();
		for(int i = 0; i < data.getRowDimension(); i++) 
			recordIndices.add(i);
		Collections.shuffle(recordIndices, getSeed());
		
		final int[] cent_indices = new int[k];
		for(int i = 0; i < k; i++) {
			centroids.add(data.getRow(recordIndices.get(i)));
			cent_indices[i] = recordIndices.get(i);
		}
		
		recordIndices = null;
		return cent_indices;
	}
}
