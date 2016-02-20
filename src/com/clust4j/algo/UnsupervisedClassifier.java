package com.clust4j.algo;

import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.metrics.SilhouetteScore;
import com.clust4j.metrics.UnsupervisedIndexAffinity;

public interface UnsupervisedClassifier extends BaseClassifier {
	/**
	 * Evaluate how the model performed against a truth set. This method
	 * utilizes the {@link UnsupervisedIndexAffinity} class
	 * @param actualLabels
	 * @return
	 */
	public double indexAffinityScore(int[] labels);
	
	
	/**
	 * Evaluate how the model performed via the {@link SilhouetteScore} metric
	 * @param actualLabels
	 * @return
	 */
	public double silhouetteScore();
	
	
	/**
	 * Evaluate how the model performed via the {@link SilhouetteScore} metric
	 * @param dist
	 * @param actualLabels
	 * @return
	 */
	public double silhouetteScore(GeometricallySeparable dist);
}
