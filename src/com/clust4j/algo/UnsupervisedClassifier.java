package com.clust4j.algo;

import com.clust4j.metrics.scoring.SilhouetteScore;
import com.clust4j.metrics.scoring.UnsupervisedIndexAffinity;

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
	 * @return
	 */
	public double silhouetteScore();
}
