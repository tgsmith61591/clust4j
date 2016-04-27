package com.clust4j.algo.preprocess;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public abstract class Transformer extends PreProcessor {
	private static final long serialVersionUID = -2321706357919100725L;
	
	protected abstract void checkFit();
	abstract public AbstractRealMatrix inverseTransform(AbstractRealMatrix X);
}
