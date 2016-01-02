package com.clust4j.utils.parallel;

/**
 * A class for distributed summing of vectors
 * @author Taylor G Smith
 */
final public class DistributedVectorSum extends DistributedVectorOperator {
	private static final long serialVersionUID = -6086182277529660733L;

    private DistributedVectorSum(final double[] arr, int lo, int hi) {
        super(arr, lo, hi);
    }

    @Override
    protected Double compute() {
        if(high - low <= MAX_CHUNK_SIZE) {
            double sum = 0;
            for(int i=low; i < high; ++i) 
                sum += array[i];
            return sum;
         } else {
            int mid = low + (high - low) / 2;
            DistributedVectorSum left  = new DistributedVectorSum(array, low, mid);
            DistributedVectorSum right = new DistributedVectorSum(array, mid, high);
            left.fork();
            double rightAns = right.compute();
            double leftAns  = left.join();
            return leftAns + rightAns;
         }
     }

     public static double sum(final double[] array) {
    	 if(array.length == 0)
    		 return 0;
         return ConcurrencyUtils.fjPool.invoke(new DistributedVectorSum(array,0,array.length));
     }
}