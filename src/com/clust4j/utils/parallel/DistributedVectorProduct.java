package com.clust4j.utils.parallel;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed products of vectors
 * @author Taylor G Smith
 */
final public class DistributedVectorProduct extends DistributedVectorOperator {
	private static final long serialVersionUID = -1038455192192012983L;

	private DistributedVectorProduct(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}
	
	@Override
	protected Double compute() {
		if(high - low <= getChunkSize()) {
            double prod = 1;
            for(int i=low; i < high; ++i) 
                prod *= array[i];
            return prod;
         } else {
            int mid = low + (high - low) / 2;
            DistributedVectorProduct left  = new DistributedVectorProduct(array, low, mid);
            DistributedVectorProduct right = new DistributedVectorProduct(array, mid, high);
            left.fork();
            double rightAns = right.compute();
            double leftAns  = left.join();
            return leftAns * rightAns;
         }
	}
	
	public static double prod(final double[] array) {
		VecUtils.checkDims(array);
		return getThreadPool().invoke(new DistributedVectorProduct(array,0,array.length));
	}
}