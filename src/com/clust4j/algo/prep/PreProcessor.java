package com.clust4j.algo.prep;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.DeepCloneable;

public interface PreProcessor extends DeepCloneable {
	@Override public PreProcessor copy();
	public AbstractRealMatrix operate(AbstractRealMatrix data);
	public double[][] operate(double[][] data);
}
