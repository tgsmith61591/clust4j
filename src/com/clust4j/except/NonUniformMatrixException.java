package com.clust4j.except;

import org.apache.commons.math3.exception.DimensionMismatchException;

public class NonUniformMatrixException extends DimensionMismatchException {
	private static final long serialVersionUID = 4638430875804061847L;

	public NonUniformMatrixException(int wrong, int expected) {
		super(wrong, expected);
	}
}
