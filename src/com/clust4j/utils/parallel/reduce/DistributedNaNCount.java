package com.clust4j.utils.parallel.reduce;

final public class DistributedNaNCount extends VectorReduceTask<Integer> {
	private static final long serialVersionUID = 5031788548523204436L;

	private DistributedNaNCount(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected Integer compute() {
		if(high - low <= getChunkSize()) {
            int sum = 0;
            for(int i=low; i < high; ++i) 
                if(Double.isNaN(array[i]))
                	sum++;
            return sum;
         } else {
            int mid = low + (high - low) / 2;
            DistributedNaNCount left  = new DistributedNaNCount(array, low, mid);
            DistributedNaNCount right = new DistributedNaNCount(array, mid, high);
            left.fork();
            int rightAns = right.compute();
            int leftAns  = left.join();
            return leftAns + rightAns;
         }
	}
	
	public static int nanCount(double[] array) {
		if(array.length == 0)
			return 0;
		return getThreadPool().invoke(new DistributedNaNCount(array,0,array.length));
	}
}