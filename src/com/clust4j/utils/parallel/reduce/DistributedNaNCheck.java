package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed NaN checks
 * @author Taylor G Smith
 */
public class DistributedNaNCheck extends ReduceTaskOperator<Boolean> {
	private static final long serialVersionUID = -4107497709587691394L;

	private DistributedNaNCheck(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected DistributedNaNCheck newInstance(double[] array, int low, int high) {
		return new DistributedNaNCheck(array, low, high);
	}

	@Override
	protected Boolean joinSides(Boolean left, Boolean right) {
		return left || right;
	}

	@Override
	protected Boolean operate(int lo, int hi) {
		for(int i=lo; i < hi; ++i)
            if(Double.isNaN(array[i]))
        		return true;
        return false;
	}
	
	public static boolean operate(final double[] array) {
		return getThreadPool().invoke(new DistributedNaNCheck(array,0,array.length));
	}

    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}
