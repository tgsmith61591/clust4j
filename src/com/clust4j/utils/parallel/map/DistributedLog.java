package com.clust4j.utils.parallel.map;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

public class DistributedLog extends MapTaskOperator {
	private static final long serialVersionUID = -3885390722365779996L;

	DistributedLog(double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected double operate(double a) {
		return FastMath.log(a);
	}

	@Override
	protected MapTaskOperator newInstance(double[] array, int low, int high) {
		return new DistributedLog(array, low, high);
	}
	
	public static double[] operate(final double[] array) {
		VecUtils.checkDimsPermitEmpty(array);
		return getThreadPool().invoke(new DistributedLog(array, 0, array.length));
    }
}
