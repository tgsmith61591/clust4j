package com.clust4j.utils.parallel;

final public class DistributedVectorNaNCount extends DistributedVectorTask<Integer> {
	private static final long serialVersionUID = 5031788548523204436L;

	private DistributedVectorNaNCount(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected Integer compute() {
		if(high - low <= MAX_CHUNK_SIZE) {
            int sum = 0;
            for(int i=low; i < high; ++i) 
                if(Double.isNaN(array[i]))
                	sum++;
            return sum;
         } else {
            int mid = low + (high - low) / 2;
            DistributedVectorNaNCount left  = new DistributedVectorNaNCount(array, low, mid);
            DistributedVectorNaNCount right = new DistributedVectorNaNCount(array, mid, high);
            left.fork();
            int rightAns = right.compute();
            int leftAns  = left.join();
            return leftAns + rightAns;
         }
	}
	
	public static int nanCount(double[] array) {
		if(array.length == 0)
			return 0;
		return ConcurrencyUtils.fjPool.invoke(new DistributedVectorNaNCount(array,0,array.length));
	}
}