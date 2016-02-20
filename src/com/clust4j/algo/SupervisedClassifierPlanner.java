package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface SupervisedClassifierPlanner extends BaseClassifierPlanner {
	public AbstractClusterer buildNewModelInstance(AbstractRealMatrix data, int[] y);
}
