package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface SupervisedLearner {
	public AbstractRealMatrix testSet();
	public int[] truthSet();
}
