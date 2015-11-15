package com.clust4j.utils;

public interface SupervisedLearner extends TrainTestSplit {
	public int[] truthSet();
}
