package com.clust4j.utils.parallel;

/**
 * A class for distributed NaN checks
 * @author Taylor G Smith
 */
public class DistributedVectorNaNCheck extends DistributedVectorTask<Boolean> {
	private static final long serialVersionUID = -4107497709587691394L;

	private DistributedVectorNaNCheck(final double[] arr, int lo, int hi) {
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
            DistributedVectorNaNCheck left  = new DistributedVectorNaNCheck(array, low, mid);
            DistributedVectorNaNCheck right = new DistributedVectorNaNCheck(array, mid, high);
            left.fork();
            boolean rightAns = right.compute();
            boolean leftAns  = left.join();
            return leftAns || rightAns;
         }
	}
	
	public static boolean containsNaN(final double[] array) {
		return getThreadPool().invoke(new DistributedVectorNaNCheck(array,0,array.length));
	}
}
