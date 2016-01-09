package com.clust4j.utils.parallel.map;

import com.clust4j.utils.VecUtils;

abstract class MapTaskOperator extends VectorMapTask {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8398439421996975256L;

	MapTaskOperator(double[] arr, int lo, int hi) {
		super(VecUtils.copy(arr), lo, hi);
	}

    @Override
    protected double[] compute() {
        if(high - low <= getChunkSize()) {
            for(int i=low; i < high; ++i) 
                array[i] = operate(array[i]);
            return array;
         } else {
            int mid = low + (high - low) / 2;
            MapTaskOperator left  = newInstance(array, low, mid);
            MapTaskOperator right = newInstance(array, mid, high);
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
	abstract protected double operate(final double a);
	
	/**
     * Must be overridden by subclasses
     * @param array
     * @param low
     * @param high
     * @return
     */
	abstract protected MapTaskOperator newInstance(final double[] array, final int low, final int high);
}
