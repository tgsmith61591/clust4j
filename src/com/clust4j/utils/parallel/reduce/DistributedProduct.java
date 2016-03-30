package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed products of vectors
 * @author Taylor G Smith
 */
@Deprecated
final public class DistributedProduct extends ReduceTaskOperator<Double> {
	private static final long serialVersionUID = -1038455192192012983L;

	private DistributedProduct(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}
	
	public static double operate(final double[] array) {
		return getThreadPool().invoke(new DistributedProduct(array,0,array.length));
	}

	@Override
	protected Double joinSides(Double left, Double right) {
		return left * right; // Product
	}

	@Override
	protected Double operate(int lo, int hi) {
		double prod = 1;
        for(int i=lo; i < hi; ++i) 
            prod *= array[i];
        return prod;
	}

	@Override
	protected DistributedProduct newInstance(double[] array, int low, int high) {
		return new DistributedProduct(array, low, high);
	}
	
    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDims(v);
    }
}