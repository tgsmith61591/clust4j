package com.clust4j.algo;

import com.clust4j.metrics.ClassificationScoring;
import com.clust4j.metrics.SilhouetteScore;
import com.clust4j.metrics.SupervisedEvaluationMetric;
import com.clust4j.metrics.UnsupervisedEvaluationMetric;

/**
 * An interface for classifiers, both supervised and unsupervised.
 * @author Taylor G Smith
 */
public interface BaseClassifier extends java.io.Serializable {
	public final static SupervisedEvaluationMetric DEF_SUPERVISED_METRIC = 
		ClassificationScoring.ACCURACY;
	public final static UnsupervisedEvaluationMetric DEF_UNSUPERVISED_METRIC =
		SilhouetteScore.getInstance();
	
	/**
	 * Returns a copy of the assigned class labels in
	 * record order
	 * @return
	 */
	public int[] getLabels();
}
