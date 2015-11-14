package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.VecUtils;

public abstract class AbstractClusterer implements Classifier {
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	final public static boolean DEF_SCALE = false;
	
	/** Underlying data */
	final protected AbstractRealMatrix data;
	/** Similarity metric */
	final protected GeometricallySeparable dist;
	/** Seed for any shuffles */
	private Random seed = new Random();
	

	
	
	/**
	 * Base planner class many clustering algorithms
	 * will extend with static inner classes. Some clustering
	 * algorithms will require more parameters and must provide
	 * the interface for the getting/setting of such parameters.
	 * 
	 * @author Taylor G Smith
	 */
	abstract protected static class BaseClustererPlanner {
		abstract public GeometricallySeparable getDist();
		abstract public boolean getScale();
		abstract public BaseClustererPlanner setScale(final boolean b);
		abstract public BaseClustererPlanner setDist(final GeometricallySeparable dist);
	}
	
	
	
	
	/**
	 * Base clusterer constructor. Sets up the distance measure,
	 * and if necessary scales data.
	 * @param data
	 * @param planner
	 */
	public AbstractClusterer(
		AbstractRealMatrix data, 
		BaseClustererPlanner planner) 
	{
		this.dist = planner.getDist();
		
		// Handle data, now...
		if(!planner.getScale())
			this.data = (AbstractRealMatrix) data.copy();
		else this.data = scale(data, (AbstractRealMatrix) data.copy());
	} // End constructor
	
	
	
	
	/**
	 * Static method to scale a matrix
	 * @param data
	 * @return
	 */
	final static protected AbstractRealMatrix scale(AbstractRealMatrix data) {
		return scale(data, (AbstractRealMatrix) data.copy());
	}
	
	/**
	 * Static method to scale a matrix given a copy
	 * @param data
	 * @param copy
	 * @return
	 */
	final static protected AbstractRealMatrix scale(AbstractRealMatrix data, AbstractRealMatrix copy) {
		// Must iter by column
		for(int col = 0; col < data.getColumnDimension(); col++) {
			final double[] v = data.getColumn(col);
			final double mean = VecUtils.mean(v);
			final double sd = VecUtils.stdDev(v, mean);
			
			for(int row = 0; row < data.getRowDimension(); row++) {
				final double new_val = (v[row] - mean) / sd;
				copy.setEntry(row, col, new_val);
			}
		}
		
		return copy;
	}

	/**
	 * Copies the underlying AbstractRealMatrix datastructure
	 * and returns the clone so as to prevent accidental referential
	 * alterations of the data.
	 * @return copy of data
	 */
	public AbstractRealMatrix getData() {
		return (AbstractRealMatrix) data.copy();
	}
	
	/**
	 * Returns the distance metric used to assess vector similarity
	 * @return distance metric
	 */
	public GeometricallySeparable getDistanceMetric() {
		return dist;
	}
	
	/**
	 * Get the current seed being used
	 * @return
	 */
	public Random getSeed() {
		return seed;
	}
	
	/**
	 * Reset the current seed
	 * @param newSeed
	 * @return
	 */
	public Random setSeed(final Random newSeed) {
		final Random old = seed;
		this.seed = newSeed;
		return old;
	}
	
	@Override
	public String toString() {
		return getName() + " clusterer";
	}
	
	
	public abstract String getName();
	public abstract boolean isTrained();
	public abstract int predict(final double[] newRecord);
	public abstract void train();
}
