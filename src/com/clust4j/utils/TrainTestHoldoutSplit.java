package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface TrainTestHoldoutSplit extends TrainTestSplit {
	public AbstractRealMatrix holdoutSet();
}
