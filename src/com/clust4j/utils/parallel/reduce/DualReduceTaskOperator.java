package com.clust4j.utils.parallel.reduce;

/**
 * A package private class to handle all dual-vector operator
 * reducing tasks. The {@link #operate(double, double)} method must
 * be overridden or a {@link UnsupportedOperationException} will be
 * thrown.
 * @author Taylor G Smith
 */
@Deprecated
abstract class DualReduceTaskOperator<T> extends DualVectorReduceTask<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6003468790911249385L;

	DualReduceTaskOperator(double[] arr, double[] arr_b, int lo, int hi) {
		super(arr, arr_b, lo, hi);
	}

    @Override
    protected T compute() {
        if(high - low <= getChunkSize()) {
        	return operate(low, high);
         } else {
            int mid = low + (high - low) / 2;
            DualReduceTaskOperator<T> left  = newInstance(array, array_b, low, mid);
            DualReduceTaskOperator<T> right = newInstance(array, array_b, mid, high);
            left.fork();
            T rightAns = right.compute();
            T leftAns = left.join();
            
            return joinSides(leftAns, rightAns);
         }
    }
    
    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @param c
     * @param low
     * @param high
     * @return
     */
    abstract protected DualReduceTaskOperator<T> newInstance(final double[] a, final double[] b, final int low, final int high);
}
