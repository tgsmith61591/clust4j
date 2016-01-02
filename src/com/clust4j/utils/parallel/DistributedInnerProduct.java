package com.clust4j.utils.parallel;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed inner products of vectors
 * @author Taylor G Smith
 */
final public class DistributedInnerProduct extends DistributedVectorOperator {
	private static final long serialVersionUID = 9189105909360824409L;
	final double[] array_b;

    private DistributedInnerProduct(final double[] a, final double[] b, int lo, int hi) {
        super(a, lo, hi);
        array_b = b;
    }

    @Override
    protected Double compute() {
        if(high - low <= MAX_CHUNK_SIZE) {
            double sum = 0;
            for(int i=low; i < high; ++i) 
                sum += array[i] * array_b[i];
            return sum;
         } else {
            int mid = low + (high - low) / 2;
            DistributedInnerProduct left  = new DistributedInnerProduct(array, array_b, low, mid);
            DistributedInnerProduct right = new DistributedInnerProduct(array, array_b, mid, high);
            left.fork();
            double rightAns = right.compute();
            double leftAns  = left.join();
            return leftAns + rightAns;
         }
     }

     public static double innerProd(final double[] array, final double[] array_b) {
    	 VecUtils.checkDims(array, array_b);
         return ConcurrencyUtils.fjPool.invoke(new DistributedInnerProduct(array,array_b,0,array.length));
     }
}