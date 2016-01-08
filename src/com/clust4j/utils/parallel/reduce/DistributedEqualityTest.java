package com.clust4j.utils.parallel.reduce;

import org.apache.commons.math3.util.Precision;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed inner products of vectors
 * @author Taylor G Smith
 */
final public class DistributedEqualityTest extends VectorReduceTask<Boolean> {
	private static final long serialVersionUID = 9189105909360824409L;
	double[] array_b;
	double tolerance;

    private DistributedEqualityTest(final double[] a, final double[] b, int lo, int hi, final double tolerance) {
        super(a, lo, hi);
        array_b = b;
        this.tolerance = tolerance;
    }

    @Override
    protected Boolean compute() {
        if(high - low <= getChunkSize()) {
            for(int i=low; i < high; ++i) 
                if( !Precision.equals(array[i], array_b[i], tolerance) )
                	return false;
            return true;
         } else {
            int mid = low + (high - low) / 2;
            DistributedEqualityTest left  = new DistributedEqualityTest(array, array_b, low, mid, tolerance);
            DistributedEqualityTest right = new DistributedEqualityTest(array, array_b, mid, high,tolerance);
            left.fork();
            boolean rightAns = right.compute();
            boolean leftAns  = left.join();
            return leftAns && rightAns;
         }
     }

     public static Boolean equalsExactly(final double[] array, final double[] array_b) {
         return equalsWithTolerance(array, array_b, 0);
     }
     
     public static Boolean equalsWithTolerance(final double[] array, final double[] array_b, final double tolerance) {
    	 VecUtils.checkDims(array, array_b);
    	 return getThreadPool().invoke(new DistributedEqualityTest(array,array_b,0,array.length,tolerance));
     }
}