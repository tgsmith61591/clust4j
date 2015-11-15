package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface TrainTestSplit {
	public AbstractRealMatrix trainSet();
	public AbstractRealMatrix testSet();
}
