package com.clust4j.utils.parallel.map;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed summing of vectors
 * @author Taylor G Smith
 */
final public class DistributedAdd extends DualMapTaskOperator {
	private static final long serialVersionUID = -6086182277529660733L;

    private DistributedAdd(final double[] arr, final double[] arr_b, final double[] arr_c, int lo, int hi) {
        super(arr, arr_b, arr_c, lo, hi);
    }

    @Override 
    protected DistributedAdd newInstance(final double[] a, final double[]b, final double[]c, final int low, final int high) {
    	return new DistributedAdd(a, b, c, low, high);
    }
    
    @Override
    protected double operate(final double a, final double b) {
    	return a + b;
    }

    public static double[] operate(final double[] array, final double[] array_b) {
    	 return getThreadPool().invoke(new DistributedAdd(array, array_b, new double[array.length], 0, array.length));
    }
    
    public static double[] scalarOperate(final double[] array, final double val) {
    	VecUtils.checkDims(array);
    	return getThreadPool().invoke(new DistributedAdd(array, VecUtils.rep(val, array.length), new double[array.length], 0, array.length));
    }
}