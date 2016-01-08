package com.clust4j.utils.parallel.reduce;

/**
 * A base class for distributed vector operations
 * @author Taylor G Smith
 */
public abstract class DistributedVectorOperator extends VectorReduceTask<Double> {
	private static final long serialVersionUID = 704439933447978232L;
	
	public DistributedVectorOperator(final double[] arr, int lo, int hi) {
        super(arr, lo, hi);
    }
}