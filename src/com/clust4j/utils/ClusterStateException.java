package com.clust4j.utils;

public class ClusterStateException extends RuntimeException {

	private static final long serialVersionUID = -2379108879459786857L;

	public ClusterStateException() {
		super();
	}
	
	public ClusterStateException(final String msg) {
		super(msg);
	}
	
	public ClusterStateException(final Throwable thrown) {
		super(thrown);
	}
	
	public ClusterStateException(final String msg, final Throwable thrown) {
		super(msg, thrown);
	}
}
