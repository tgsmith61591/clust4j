package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

final public class DistributedNaNCount extends ReduceTaskOperator<Integer> {
	private static final long serialVersionUID = 5031788548523204436L;

	private DistributedNaNCount(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected DistributedNaNCount newInstance(double[] array, int low, int high) {
		return new DistributedNaNCount(array, low, high);
	}

	@Override
	protected Integer joinSides(Integer left, Integer right) {
		return left + right; // Sum of counts
	}

	@Override
	protected Integer operate(int lo, int hi) {
		int sum = 0;
        for(int i=lo; i < hi; ++i) 
            if(Double.isNaN(array[i]))
            	sum++;
        return sum;
	}
	
	public static int operate(double[] array) {
		if(array.length == 0)
			return 0;
		return getThreadPool().invoke(new DistributedNaNCount(array,0,array.length));
	}

    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}