package com.clust4j.utils;
import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.MeanShift;

/**
 * Any cluster that does not force a prediction for every
 * single point is considered a "noisey clusterer." This interface
 * provides the method {@link #getNumberOfNoisePoints()}, which
 * returns the number of points that were not classified as
 * belonging to any clusters.
 * 
 * @author Taylor G Smith
 * @see {@link DBSCAN}
 * @see {@link MeanShift}
 */
public interface NoiseyClusterer extends SelfSegmenting {
	/**
	 * the number of points that were not classified as
	 * belonging to any clusters.
	 * @return how many points are considered noise
	 */
	public int getNumberOfNoisePoints();
}
