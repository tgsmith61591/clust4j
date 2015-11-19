package com.clust4j.kernel;

public abstract class ConstantKernel extends Kernel {
	public static final double DEFAULT_CONSTANT = 1;
	
	public ConstantKernel() {
		super();
	}
	
	abstract public double getConstant();
}
