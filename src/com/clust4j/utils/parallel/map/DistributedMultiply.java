package com.clust4j.utils.parallel.map;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed summing of vectors
 * @author Taylor G Smith
 */
@Deprecated
final public class DistributedMultiply extends DualVectorMapTaskOperator {
	private static final long serialVersionUID = -6086182277529660733L;

    private DistributedMultiply(final double[] arr, final double[] arr_b, final double[] arr_c, int lo, int hi) {
        super(arr, arr_b, arr_c, lo, hi);
    }
    
    @Override
    protected double operate(final double a, final double b) {
    	return a * b;
    }
    
    @Override 
    protected DistributedMultiply newInstance(final double[] a, final double[]b, final double[]c, final int low, final int high) {
    	return new DistributedMultiply(a, b, c, low, high);
    }

    public static double[] operate(final double[] array, final double[] array_b) {
    	 return getThreadPool().invoke(new DistributedMultiply(array, array_b, new double[array.length], 0, array.length));
    }
    
    public static double[] scalarOperate(final double[] array, final double val) {
    	VecUtils.checkDims(array);
    	return getThreadPool().invoke(new DistributedMultiply(array, VecUtils.rep(val, array.length), new double[array.length], 0, array.length));
    }

    void dimCheck(double[] a, double[] b) {
    	VecUtils.checkDimsPermitEmpty(a, b);
    }
}