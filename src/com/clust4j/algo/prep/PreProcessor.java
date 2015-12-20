package com.clust4j.algo.prep;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public interface PreProcessor {
	public AbstractRealMatrix operate(AbstractRealMatrix data);
	public double[][] operate(double[][] data);
}
