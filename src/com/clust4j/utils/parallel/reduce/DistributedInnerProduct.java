package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed inner products of vectors
 * @author Taylor G Smith
 */
final public class DistributedInnerProduct extends DualReduceTaskOperator<Double> {
	private static final long serialVersionUID = 9189105909360824409L;

    private DistributedInnerProduct(final double[] a, final double[] b, int lo, int hi) {
        super(a, b, lo, hi);
    }

	@Override
	protected Double joinSides(Double left, Double right) {
		return left + right; // Sum
	}

	@Override
	protected Double operate(int lo, int hi) {
		double s = 0;
		for(int i = lo; i < hi; i++)
			s += array[i] * array_b[i];
		return s;
	}

	@Override
	protected DistributedInnerProduct newInstance(double[] a, double[] b, int low, int high) {
		return new DistributedInnerProduct(a, b, low, high);
	}

    public static double operate(final double[] array, final double[] array_b) {
        return getThreadPool().invoke(new DistributedInnerProduct(array,array_b,0,array.length));
    }

    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}