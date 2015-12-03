package com.clust4j.utils;

public interface PredictableClassifier extends Classifier {
	/**
	 * Predicts the class label membership given a new record
	 * @param newRecord
	 * @return the class label
	 */
	public int predict(final double[] newRecord);
}
