package com.clust4j.utils.parallel.reduce;
import com.clust4j.utils.parallel.VectorMRTask;

abstract class VectorReduceTask<T> extends VectorMRTask<T> {
	private static final long serialVersionUID = -7986981765361158408L;
	
	VectorReduceTask(double[] arr, int lo, int hi) {
		super(arr, lo, hi);
		checkDims(arr);
	}
    
    /**
     * How to join two values in the result
     */
    protected abstract T joinSides(final T left, final T right);

    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @return
     */
    abstract protected T operate(final int lo, final int hi);
    
    /**
     * Check dims
     * @param v
     */
    abstract void checkDims(double[] v);
}
