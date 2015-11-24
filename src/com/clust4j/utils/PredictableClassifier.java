package com.clust4j.utils;

public interface PredictableClassifier extends Classifier {
	public int predict(final double[] newRecord);
}
