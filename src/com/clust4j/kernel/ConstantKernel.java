package com.clust4j.kernel;

abstract class ConstantKernel extends Kernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3376273063247220042L;
	public static final double DEFAULT_CONSTANT = 1;
	
	public ConstantKernel() {
		super();
	}
	
	abstract public double getConstant();
}
