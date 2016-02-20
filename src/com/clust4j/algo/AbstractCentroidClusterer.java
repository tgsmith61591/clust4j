package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.metrics.SilhouetteScore;
import com.clust4j.metrics.UnsupervisedIndexAffinity;
import com.clust4j.utils.Convergeable;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public abstract class AbstractCentroidClusterer extends AbstractPartitionalClusterer 
		implements CentroidLearner, Convergeable, UnsupervisedClassifier {
	
	private static final long serialVersionUID = -424476075361612324L;
	final public static double DEF_MIN_CHNG = 0.005;
	final public static int DEF_K = Neighbors.DEF_K;
	
	final protected int maxIter;
	final protected double minChange;
	final protected int[] init_centroid_indices;
	final protected int m;
	
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
		this.m = data.getRowDimension();
		
		if(maxIter < 0)	throw new IllegalArgumentException("maxIter must exceed 0");
		if(minChange<0)	throw new IllegalArgumentException("minChange must exceed 0");
		
		meta("maxIter="+maxIter);
		meta("minChange="+minChange);
		
		this.init_centroid_indices = initCentroids();
	}

	
	
	public static abstract class CentroidClustererPlanner 
			extends BaseClustererPlanner 
			implements UnsupervisedClassifierPlanner {
		
		abstract public int getK();
		abstract public int getMaxIter();
		abstract public double getMinChange();
	}
	



	
	@Override
	public boolean didConverge() {
		return converged;
	}
	
	@Override
	public ArrayList<double[]> getCentroids() {
		final ArrayList<double[]> cent = new ArrayList<double[]>();
		for(double[] d : centroids)
			cent.add(VecUtils.copy(d));
		
		return cent;
	}
	
	/**
	 * Returns a copy of the classified labels
	 */
	@Override
	public int[] getLabels() {
		try {
			return VecUtils.copy(labels);
			
		} catch(NullPointerException npe) {
			String error = "model has not yet been fit";
			error(error);
			throw new ModelNotFitException(error);
		}
	}
	
	@Override
	public int getMaxIter() {
		return maxIter;
	}
	
	@Override
	public double getConvergenceTolerance() {
		return minChange;
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
		for(int i = 0; i < m; i++) 
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
	
	@Override
	public int itersElapsed() {
		return iter;
	}
	
	/** {@inheritDoc} */
	@Override
	public double indexAffinityScore(int[] labels) {
		// Propagates ModelNotFitException
		return UnsupervisedIndexAffinity.getInstance().evaluate(labels, getLabels());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore() {
		return silhouetteScore(getSeparabilityMetric());
	}

	/** {@inheritDoc} */
	@Override
	public double silhouetteScore(GeometricallySeparable dist) {
		// Propagates ModelNotFitException
		return SilhouetteScore.getInstance().evaluate(this, dist, getLabels());
	}
	
	
	abstract TreeMap<Integer, ArrayList<Integer>> assignClustersAndLabelsInPlace();
}
