package com.clust4j.utils;

import com.clust4j.metrics.ClassificationScoring;
import com.clust4j.metrics.EvaluationMetric;

/**
 * An interface for classifiers, both supervised and unsupervised.
 * @author Taylor G Smith
 */
public interface Classifier extends java.io.Serializable {
	public final static EvaluationMetric DEF_METRIC = ClassificationScoring.ACCURACY;
	
	/**
	 * Returns a copy of the assigned class labels in
	 * record order
	 * @return
	 */
	public int[] getLabels();
	
	/**
	 * Evaluate how the model performed. Every classifier should
	 * have a default scoring method
	 * @param actualLabels
	 * @return
	 */
	public double score(final int[] actualLabels);
	
	/**
	 * Evaluate how the model performed
	 * @param actualLabels
	 * @return
	 */
	public double score(final int[] actualLabels, final EvaluationMetric metric);
}
