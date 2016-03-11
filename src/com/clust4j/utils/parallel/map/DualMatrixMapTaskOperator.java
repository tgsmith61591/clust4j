package com.clust4j.utils.parallel.map;

abstract class DualMatrixMapTaskOperator extends DualMatrixMapTask {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2401423511466814014L;

	DualMatrixMapTaskOperator(double[][] mat, double[][] mat_b, double[][] result, int lo, int hi) {
		super(mat, mat_b, result, lo, hi);
	}

	@Override
    protected double[][] compute() {
        if(high - low <= getChunkSize()) {
            return operate(low, high);
        } else {
            int mid = low + (high - low) / 2;
            DualMatrixMapTaskOperator left  = newInstance(matrix, matrix_b, matrix_c, low, mid);
            DualMatrixMapTaskOperator right = newInstance(matrix, matrix_b, matrix_c, mid, high);
            left.fork();
            right.compute();
            left.join();
            
            return matrix_c;
        }
    }
	
	/**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @return
     */
    abstract protected double[][] operate(final int lo, final int hi);
    
    /**
     * Must be overridden by subclasses
     * @param a
     * @param b
     * @param c
     * @param low
     * @param high
     * @return
     */
    abstract protected DualMatrixMapTaskOperator newInstance(final double[][] a, final double[][] b, final double[][] c, final int low, final int high);
}
