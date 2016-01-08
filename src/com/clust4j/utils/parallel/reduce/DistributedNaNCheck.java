package com.clust4j.utils.parallel.reduce;

/**
 * A class for distributed NaN checks
 * @author Taylor G Smith
 */
public class DistributedNaNCheck extends VectorReduceTask<Boolean> {
	private static final long serialVersionUID = -4107497709587691394L;

	private DistributedNaNCheck(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected Boolean compute() {
		if(high - low <= getChunkSize()) {
            for(int i=low; i < high; ++i)
                if(Double.isNaN(array[i]))
            		return true;
            
            return false;
         } else {
            int mid = low + (high - low) / 2;
            DistributedNaNCheck left  = new DistributedNaNCheck(array, low, mid);
            DistributedNaNCheck right = new DistributedNaNCheck(array, mid, high);
            left.fork();
            boolean rightAns = right.compute();
            boolean leftAns  = left.join();
            return leftAns || rightAns;
         }
	}
	
	public static boolean containsNaN(final double[] array) {
		return getThreadPool().invoke(new DistributedNaNCheck(array,0,array.length));
	}
}
