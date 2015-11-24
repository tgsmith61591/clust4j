package com.clust4j.utils;

public interface PredictableClassifier extends Classifier {
	public abstract int predict(final double[] newRecord);
}
