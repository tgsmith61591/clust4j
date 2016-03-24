package com.clust4j.utils.parallel.reduce;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.Precision;

import com.clust4j.utils.VecUtils;

public class DistributedEqualityTest extends VectorReduceTask<Boolean> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6929463475856249545L;
	final double tolerance;
	final double[] arr_b;

	DistributedEqualityTest(double[] arr_a, double[] arr_b, double tolerance, int lo, int hi) {
		super(arr_a, lo, hi);
		
		if(arr_a.length != arr_b.length)
			throw new DimensionMismatchException(arr_a.length, arr_b.length);
		if(tolerance < 0)
			throw new IllegalArgumentException("tolerance cannot be less than 0");
		
		this.tolerance = tolerance;
		this.arr_b = arr_b;
	}

	@Override
	protected Boolean compute() {
		if(high - low <= getChunkSize()) {
        	return operate(low, high);
        } else {
        	int mid = low + (high - low) / 2;
        	DistributedEqualityTest left  = new DistributedEqualityTest(array, arr_b, tolerance, low, mid);
        	DistributedEqualityTest right = new DistributedEqualityTest(array, arr_b, tolerance, mid, high);
            left.fork();
            Boolean rightAns = right.compute();
            Boolean leftAns = left.join();
            
            return joinSides(leftAns, rightAns);
        }
	}

	@Override
	protected Boolean joinSides(Boolean left, Boolean right) {
		return left && right; // Both need to be equal
	}

	@Override
	protected Boolean operate(int lo, int hi) {
		for(int i = lo; i < hi; i++) {
			if(Double.isNaN(array[i]) && Double.isNaN(arr_b[i]))
				continue;
			
			if( !Precision.equals(array[i], arr_b[i], tolerance) )
				return false;
		}
		
		return true;
	}
	
	public static Boolean operate(final double[] array, final double[] array_b, final double tol) {
		// corner...
		if(null == array && null == array_b)
			return true;
		
		return getThreadPool().invoke(new DistributedEqualityTest(array,array_b,tol,0,array.length));
	}

    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}
