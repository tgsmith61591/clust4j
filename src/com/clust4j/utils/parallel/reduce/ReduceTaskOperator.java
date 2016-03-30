package com.clust4j.utils.parallel.reduce;

@Deprecated
abstract class ReduceTaskOperator<T> extends VectorReduceTask<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8398439421996975256L;

	ReduceTaskOperator(double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

    @Override
    protected T compute() {
        if(high - low <= getChunkSize()) {
        	return operate(low, high);
        } else {
        	int mid = low + (high - low) / 2;
            ReduceTaskOperator<T> left  = newInstance(array, low, mid);
            ReduceTaskOperator<T> right = newInstance(array, mid, high);
            left.fork();
            T rightAns = right.compute();
            T leftAns = left.join();
            
            return joinSides(leftAns, rightAns);
        }
    }
	
	/**
     * Must be overridden by subclasses
     * @param array
     * @param low
     * @param high
     * @return
     */
	abstract protected ReduceTaskOperator<T> newInstance(final double[] array, final int low, final int high);
}
