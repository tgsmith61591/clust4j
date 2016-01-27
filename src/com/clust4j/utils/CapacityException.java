package com.clust4j.utils;

public class CapacityException extends IllegalStateException {
	private static final long serialVersionUID = -6001899328378324782L;

	public CapacityException() {
		super();
	}
	
	public CapacityException(final String msg) {
		super(msg);
	}
	
	public CapacityException(final Throwable thrown) {
		super(thrown);
	}
	
	public CapacityException(final String msg, final Throwable thrown) {
		super(msg, thrown);
	}
}
