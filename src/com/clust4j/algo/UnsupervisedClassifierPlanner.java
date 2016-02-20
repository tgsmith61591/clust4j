package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface UnsupervisedClassifierPlanner extends BaseClassifierPlanner {
	public AbstractClusterer buildNewModelInstance(AbstractRealMatrix data);
}
