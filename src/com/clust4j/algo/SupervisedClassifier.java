package com.clust4j.algo;

import com.clust4j.metrics.scoring.SupervisedEvaluationMetric;

public interface SupervisedClassifier extends BaseClassifier {
	public int[] getTrainingLabels();
	
	
	/**
	 * Evaluate how the model performed. Every classifier should
	 * have a default scoring method
	 * @param actualLabels
	 * @return
	 */
	public double score();
	
	/**
	 * Evaluate how the model performed
	 * @param actualLabels
	 * @return
	 */
	public double score(final SupervisedEvaluationMetric metric);
}
