package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed summing of vectors
 * @author Taylor G Smith
 */
@Deprecated
final public class DistributedSum extends ReduceTaskOperator<Double> {
	private static final long serialVersionUID = -6086182277529660733L;

    private DistributedSum(final double[] arr, int lo, int hi) {
        super(arr, lo, hi);
    }

	@Override
	protected DistributedSum newInstance(double[] array, int low, int high) {
		return new DistributedSum(array, low, high);
	}

	@Override
	protected Double joinSides(Double left, Double right) {
		return left + right; // Sum
	}

	@Override
	protected Double operate(int lo, int hi) {
		double sum = 0;
		for(int i = lo; i < hi; i++)
			sum += array[i];
		return sum;
	}

    public static double operate(final double[] array) {
    	if(array.length == 0)
    		return 0;
        return getThreadPool().invoke(new DistributedSum(array,0,array.length));
    }
    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}