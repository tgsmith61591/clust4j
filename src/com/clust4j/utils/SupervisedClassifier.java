package com.clust4j.utils;

public interface SupervisedClassifier extends PredictableClassifier {
	/**
	 * Return the predicted test set labels
	 * @return
	 */
	public int[] getPredictedLabels();
}
