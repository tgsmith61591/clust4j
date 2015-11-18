package com.clust4j.kernel;

public abstract class AbstractConstantKernel extends AbstractKernel {
	public static final double DEFAULT_CONSTANT = 0;
	
	public AbstractConstantKernel() {
		super();
	}
	
	abstract public double getConstant();
}
