package com.clust4j.utils.parallel.map;

/**
 * A package private class to handle all dual-vector operator
 * mapping tasks. The {@link #operate(double, double)} method must
 * be overridden or a {@link UnsupportedOperationException} will be
 * thrown.
 * @author Taylor G Smith
 */
abstract class DualMapTaskOperator extends DualVectorMapTask {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6003468790911249385L;

	DualMapTaskOperator(double[] arr, double[] arr_b, double[] result, int lo,
			int hi) {
		super(arr, arr_b, result, lo, hi);
	}

    @Override
    protected double[] compute() {
        if(high - low <= getChunkSize()) {
            for(int i=low; i < high; ++i) 
                array_c[i] = operate(array[i], array_b[i]);
            return array_c;
         } else {
            int mid = low + (high - low) / 2;
            DualMapTaskOperator left  = newInstance(array, array_b, array_c, low, mid);
            DualMapTaskOperator right = newInstance(array, array_b, array_c, mid, high);
            left.fork();
            right.compute();
            left.join();
            return array;
         }
    }

    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @return
     */
    abstract protected double operate(final double a, final double b);
    
    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @param c
     * @param low
     * @param high
     * @return
     */
    abstract protected DualMapTaskOperator newInstance(final double[] a, final double[] b, final double[] c, final int low, final int high);
}
